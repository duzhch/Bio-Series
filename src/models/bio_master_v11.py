#!/usr/bin/env python3
import os
import json
import math
import re
import hashlib
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from torch.utils.data import Dataset, DataLoader
from pandas_plink import read_plink
from pathlib import Path

from src.id_mapping import map_pheno_ids_to_plink_ids

SUPPORTED_ABLATIONS = {
    'full',
    'no_delta',
    'no_gene2vec',
    'no_bio_prior',
    'no_pca',
    'pca_only_prior_off',
}


def normalize_ablation(ablation):
    normalized = 'full' if ablation is None else str(ablation).strip().lower()
    normalized = re.sub(r'[\s-]+', '_', normalized)
    normalized = re.sub(r'_+', '_', normalized).strip('_')
    if normalized not in SUPPORTED_ABLATIONS:
        raise ValueError(
            f"Unsupported ablation='{ablation}'. Expected one of {sorted(SUPPORTED_ABLATIONS)}"
        )
    return normalized


@dataclass(frozen=True)
class AblationConfig:
    name: str
    prior_mode: str
    use_context: bool = True
    zero_delta: bool = False
    zero_gene: bool = False

    @classmethod
    def from_name(cls, ablation):
        name = normalize_ablation(ablation)
        if name == 'full':
            return cls(name=name, prior_mode='learned_two_channel')
        if name == 'no_delta':
            return cls(name=name, prior_mode='learned_two_channel', zero_delta=True)
        if name == 'no_gene2vec':
            return cls(name=name, prior_mode='learned_two_channel', zero_gene=True)
        if name == 'no_bio_prior':
            return cls(name=name, prior_mode='single_channel')
        if name == 'no_pca':
            return cls(name=name, prior_mode='learned_two_channel', use_context=False)
        if name == 'pca_only_prior_off':
            return cls(name=name, prior_mode='zero_two_channel')
        raise AssertionError(f"Unhandled ablation mode: {name}")

# ==========================================
# 1. Hybrid Loss Function (Robust Version)
# ==========================================
class HybridPCCLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.1):
        super().__init__()
        self.alpha = alpha 
        self.beta = beta   
        self.gamma = gamma 

    def pcc_loss(self, pred, target):
        """ 1 - PCC. Optimizes linearity. """
        vx = pred - torch.mean(pred)
        vy = target - torch.mean(target)
        
        # [Robust Fix] Stability protection
        sum_vx2 = torch.sum(vx ** 2)
        sum_vy2 = torch.sum(vy ** 2)
        denom = torch.sqrt(sum_vx2) * torch.sqrt(sum_vy2)
        denom = torch.clamp(denom, min=1e-6) 
        
        cost = torch.sum(vx * vy) / denom
        return 1 - cost

    def listnet_loss(self, pred, target):
        """ ListNet Ranking Loss. """
        P_y_true = F.softmax(target, dim=0)
        P_y_pred = F.softmax(pred, dim=0)
        # [Robust Fix] Epsilon in log
        return -torch.sum(P_y_true * torch.log(P_y_pred + 1e-8))

    def forward(self, pred, target, deep_feat, context_feat):
        p = pred.squeeze()
        t = target.squeeze()
        
        loss_pcc = self.pcc_loss(p, t)
        loss_rank = self.listnet_loss(p, t)
        
        if deep_feat is not None and context_feat is not None:
            orth_loss = torch.mean(
                torch.abs(F.cosine_similarity(deep_feat, context_feat.detach(), dim=1))
            )
        else:
            orth_loss = torch.tensor(0.0, device=pred.device)

        return self.alpha * loss_pcc + self.beta * loss_rank + self.gamma * orth_loss

# ==========================================
# 2. Components: Attention & Transformer
# ==========================================

class PriorGenerator(nn.Module):
    def __init__(self, delta_dim, gene_dim):
        super().__init__()
        self.delta_compress = nn.Linear(delta_dim, 16)
        self.gene_compress = nn.Linear(gene_dim, 16)
        self.fuse = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
    def forward(self, d_emb, g_emb, zero_delta=False, zero_gene=False):
        d = F.relu(self.delta_compress(d_emb), inplace=True)
        g = F.relu(self.gene_compress(g_emb), inplace=True)
        if zero_delta:
            d = torch.zeros_like(d)
        if zero_gene:
            g = torch.zeros_like(g)
        return self.fuse(torch.cat([d, g], dim=1))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [Batch, Seq_Len, D_Model] (because batch_first=True)
        return x + self.pe[:, :x.size(1), :]

class GenomicTransformer(nn.Module):
    def __init__(self, block_size, in_channels=2, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        
        # Local Conv: [Batch, 2, Block_Size] -> [Batch, d_model, 1]
        self.local_conv = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=block_size) # Pooling entire block to 1 token
        )
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=d_model*2, 
                                                   dropout=0.2, batch_first=True) # Increased dropout for GS
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x_folded):
        return self.local_conv(x_folded).squeeze(-1)

# ==========================================
# 3. Bio-Master V10 (The Transformer Shift)
# ==========================================

class BioMasterV10(nn.Module):
    def __init__(self, delta_E, gene_E, num_snps, num_pcs, block_size=100, ablation='full'):
        super().__init__()
        self.ablation = AblationConfig.from_name(ablation)
        self.register_buffer('delta_E', torch.tensor(delta_E, dtype=torch.float32))
        self.register_buffer('gene_E', torch.tensor(gene_E, dtype=torch.float32))
        
        # Dimensions
        self.block_size = block_size
        self.pad_len = (block_size - (num_snps % block_size)) % block_size
        self.n_blocks = (num_snps + self.pad_len) // block_size
        self.d_model = 64 
        
        # --- Deep Tower (Genomic Transformer) ---
        self.prior_gen = PriorGenerator(delta_E.shape[1], gene_E.shape[1])
        deep_in_channels = 1 if self.ablation.prior_mode == 'single_channel' else 2
        
        self.genomic_transformer = GenomicTransformer(
            block_size=block_size, 
            in_channels=deep_in_channels, 
            d_model=self.d_model, 
            nhead=4,
            num_layers=2 
        )
        
        # [ARCH IMPROVEMENT] Global Average Pooling Head
        # Instead of Flattening (N_Blocks * 64), we average over blocks.
        # drastically reduces parameters prevents overfitting.
        self.deep_head = nn.Sequential(
            nn.Linear(self.d_model, 64), 
            nn.LayerNorm(64), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True)
        )
        self.deep_out = nn.Linear(64, 1)
        
        # --- Context Tower ---
        if self.ablation.use_context:
            self.context_feat_extractor = nn.Sequential(
                nn.Linear(num_pcs, 64), 
                nn.ReLU(inplace=True)
            )
            self.context_out = nn.Linear(64, 1)
        else:
            self.context_feat_extractor = None
            self.context_out = None

        # --- Wide Tower ---
        self.wide_linear = nn.Linear(num_snps, 1)
        nn.init.normal_(self.wide_linear.weight, 0, 0.001)

    def _build_priors(self):
        if self.ablation.prior_mode != 'learned_two_channel':
            return torch.zeros(
                self.delta_E.shape[0], 1, device=self.delta_E.device, dtype=self.delta_E.dtype
            )

        return self.prior_gen(
            self.delta_E,
            self.gene_E,
            zero_delta=self.ablation.zero_delta,
            zero_gene=self.ablation.zero_gene,
        )

    def forward(self, X_snps, X_pcs):
        B, N = X_snps.shape
        
        # Path A: Wide
        out_wide = self.wide_linear(X_snps)
        
        # Path B: Context
        if self.context_feat_extractor is not None:
            feat_ctx = self.context_feat_extractor(X_pcs)
            out_ctx = self.context_out(feat_ctx)
        else:
            feat_ctx = None
            out_ctx = torch.zeros_like(out_wide)
        
        # Path C: Deep (Transformer)
        priors = self._build_priors()
        
        if self.pad_len > 0:
            X_p = F.pad(X_snps, (0, self.pad_len))
            P_p = F.pad(priors.t(), (0, self.pad_len)).t()
        else:
            X_p, P_p = X_snps, priors
            
        g_blocks = X_p.view(B, self.n_blocks, self.block_size)
        if self.ablation.prior_mode == 'single_channel':
            x_folded = g_blocks.reshape(-1, 1, self.block_size)
        else:
            p_blocks = P_p.view(1, self.n_blocks, self.block_size).expand(B, -1, -1)
            
            # [B * N_Blocks, 2, Block_Size]
            x_folded = torch.stack([g_blocks, p_blocks], dim=2).view(-1, 2, self.block_size)
        
        # 1. Local Compression -> [B*N_Blk, d_model]
        block_tokens = self.genomic_transformer.local_conv(x_folded).squeeze(-1) 
        
        # 2. Unfold -> [B, N_Blocks, d_model]
        seq_tokens = block_tokens.view(B, self.n_blocks, self.d_model)
        
        # 3. Positional Encoding
        seq_tokens = self.genomic_transformer.pos_encoder(seq_tokens)
        
        # 4. Global Self-Attention
        trans_out = self.genomic_transformer.transformer_encoder(seq_tokens) 
        
        # 5. [ARCH IMPROVEMENT] Global Average Pooling (GAP)
        # Collapse the sequence dimension (N_Blocks) by averaging
        gap_feat = torch.mean(trans_out, dim=1) # [B, d_model]
        
        feat_deep = self.deep_head(gap_feat) 
        out_deep = self.deep_out(feat_deep)
        
        return out_wide + out_ctx + out_deep, priors[:N], feat_deep, feat_ctx

# ==========================================
# 4. Training Utilities (Robust)
# ==========================================

class _DS(Dataset):
    def __init__(self, X, P, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.P = torch.tensor(P, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.P[i], self.y[i]

def load_pca(pca_file, valid_ids):
    if not os.path.exists(pca_file): return np.zeros((len(valid_ids), 10))
    try:
        df = pd.read_csv(pca_file)
        df['IID'] = df['IID'].astype(str)
        pc_cols = [c for c in df.columns if c.startswith('PC')]
        if not pc_cols: return np.zeros((len(valid_ids), 10))
        pca_map = df.set_index('IID')[pc_cols].to_dict('index')
        zero_vec = [0.0] * len(pc_cols)
        return np.array([list(pca_map.get(iid, zero_vec).values()) for iid in valid_ids])
    except: return np.zeros((len(valid_ids), 10))


def _make_train_val_split(train_ids, val_fraction=0.1, min_val_size=2):
    num_samples = len(train_ids)
    if num_samples <= 1:
        return list(range(num_samples)), []

    target_val = max(1, math.ceil(num_samples * val_fraction))
    if num_samples >= 4:
        target_val = max(target_val, min_val_size)
    target_val = min(target_val, num_samples - 1)

    ranked = sorted(
        range(num_samples),
        key=lambda idx: hashlib.md5(str(train_ids[idx]).encode("utf-8")).hexdigest(),
    )
    val_idx = sorted(ranked[:target_val])
    val_lookup = set(val_idx)
    train_idx = [idx for idx in range(num_samples) if idx not in val_lookup]
    return train_idx, val_idx


def _build_fold_pca_features(x_train, other_arrays, num_pcs=10):
    x_train = np.asarray(x_train, dtype=np.float32)
    other_arrays = [np.asarray(arr, dtype=np.float32) for arr in other_arrays]

    def zeros_for(arr):
        return np.zeros((arr.shape[0], num_pcs), dtype=np.float32)

    if x_train.ndim != 2:
        raise ValueError(f"Expected x_train to be 2D, got shape={x_train.shape}")
    if x_train.shape[0] == 0 or x_train.shape[1] == 0:
        return zeros_for(x_train), [zeros_for(arr) for arr in other_arrays]

    train_mean = np.mean(x_train, axis=0, keepdims=True)
    centered_train = x_train - train_mean
    if x_train.shape[0] < 2:
        return zeros_for(x_train), [zeros_for(arr) for arr in other_arrays]

    _u, _s, vt = np.linalg.svd(centered_train, full_matrices=False)
    effective_pcs = min(num_pcs, vt.shape[0])
    components = vt[:effective_pcs].T

    def transform(arr):
        centered = arr - train_mean
        scores = centered @ components
        if effective_pcs < num_pcs:
            scores = np.pad(scores, ((0, 0), (0, num_pcs - effective_pcs)))
        return scores.astype(np.float32, copy=False)

    return transform(x_train), [transform(arr) for arr in other_arrays]

def _load_pheno_map(pheno_file, trait, plink_prefix):
    df = pd.read_csv(pheno_file, sep='\t' if pheno_file.endswith('.tsv') else ',')
    id_col = next((c for c in df.columns if c.lower() in ['iid', 'sample_id', 'id']), df.columns[0])
    mapped_df = map_pheno_ids_to_plink_ids(df, plink_prefix, id_col)
    valid_df = mapped_df[mapped_df[trait].notna()].copy()
    return dict(zip(valid_df['IID'].astype(str), valid_df[trait]))

def train(plink_prefix, pheno_file, train_ids, test_ids, trait, delta_path, gene_path, out_dir, 
          lr=3e-4, batch_size=64, epochs=150, lambda_l1=0.005, device='auto', ablation='full'):
    
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True) # Ensure dir exists

    dev = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    ablation = normalize_ablation(ablation)
    
    if dev == 'cpu':
        torch.set_num_threads(4) 
    
    print(f"🚀 Training Bio-Master V10 (Transformer + Robust) on {dev} [ablation={ablation}]")

    # Load Data
    delta = np.load(delta_path); gene = np.load(gene_path)
    if gene.ndim != 2:
        raise ValueError(f"[Gene2Vec] Expected gene_knowledge to be 2D, got shape={gene.shape} from {gene_path}")
    print(f"[Gene2Vec] gene_knowledge dim = {gene.shape[1]}")
    if gene.shape[1] != 300:
        raise ValueError(f"[Gene2Vec] Expected gene_knowledge dim=300, got dim={gene.shape[1]} from {gene_path}")
    m = min(len(delta), len(gene)); delta, gene = delta[:m], gene[:m]
    (bim, fam, bed) = read_plink(plink_prefix, verbose=False)
    G = bed.compute().T
    if G.shape[1] > m: G = G[:, :m]
    
    ymap = _load_pheno_map(pheno_file, trait, plink_prefix)
    iid2idx = {str(iid): i for i, iid in enumerate(fam['iid'].astype(str))}
    tr_df = pd.read_csv(train_ids, sep='\t', names=['FID', 'IID'])
    te_df = pd.read_csv(test_ids, sep='\t', names=['FID', 'IID'])
    
    all_train_ids = [str(x) for x in tr_df['IID'] if str(x) in iid2idx and str(x) in ymap]
    te_ids = [str(x) for x in te_df['IID'] if str(x) in iid2idx and str(x) in ymap]

    train_split_idx, val_split_idx = _make_train_val_split(all_train_ids)
    fit_ids = [all_train_ids[idx] for idx in train_split_idx]
    val_ids = [all_train_ids[idx] for idx in val_split_idx]
    if not val_ids:
        val_ids = list(fit_ids)

    X_fit = G[[iid2idx[x] for x in fit_ids]]
    X_val = G[[iid2idx[x] for x in val_ids]]
    X_te = G[[iid2idx[x] for x in te_ids]]
    y_fit = np.array([ymap[x] for x in fit_ids])
    y_val = np.array([ymap[x] for x in val_ids])
    y_te = np.array([ymap[x] for x in te_ids])

    # [Leakage Fix] Fit genotype scaling and context PCA on the training subset only.
    sc_g = StandardScaler()
    X_fit = sc_g.fit_transform(X_fit)
    X_val = sc_g.transform(X_val)
    X_te = sc_g.transform(X_te)

    P_fit, (P_val, P_te) = _build_fold_pca_features(X_fit, [X_val, X_te], num_pcs=10)
    sc_p = StandardScaler()
    P_fit = sc_p.fit_transform(P_fit)
    P_val = sc_p.transform(P_val)
    P_te = sc_p.transform(P_te)

    X_fit = np.nan_to_num(X_fit)
    X_val = np.nan_to_num(X_val)
    X_te = np.nan_to_num(X_te)
    P_fit = np.nan_to_num(P_fit)
    P_val = np.nan_to_num(P_val)
    P_te = np.nan_to_num(P_te)
    
    model = BioMasterV10(
        delta,
        gene,
        num_snps=X_fit.shape[1],
        num_pcs=P_fit.shape[1],
        block_size=100,
        ablation=ablation,
    ).to(dev)
    
    base_params = [p for n, p in model.named_parameters() if 'wide' not in n]
    wide_params = [p for n, p in model.named_parameters() if 'wide' in n]
    opt = optim.AdamW([
        {'params': base_params, 'weight_decay': 1e-3},
        {'params': wide_params, 'weight_decay': 0.1} 
    ], lr=lr)
    
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    crit = HybridPCCLoss(alpha=1.0, beta=0.5, gamma=0.1)

    nw = 4 if dev=='cuda' else 0 
    loader = DataLoader(_DS(X_fit, P_fit, y_fit), batch_size=batch_size, shuffle=True, 
                        num_workers=nw, drop_last=True)
    
    best_pcc = -1.0; patience = 0
    # [Robust] Save initial fallback
    torch.save(model.state_dict(), out / 'best_model.pt')
    
    for ep in range(epochs):
        model.train()
        tot_loss = 0; cnt = 0
        
        for x, p, y in loader:
            x, p, y = x.to(dev), p.to(dev), y.to(dev)
            opt.zero_grad()
            pred, priors, feat_deep, feat_ctx = model(x, p)
            
            main_loss = crit(pred, y, feat_deep, feat_ctx)
            loss = main_loss + lambda_l1 * torch.norm(priors, 1)
            
            # [Robust] NaN Check
            if torch.isnan(loss):
                continue
            
            loss.backward()
            
            # [Robust] Gradient Clipping (Crucial for Transformer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            opt.step()
            tot_loss += loss.item(); cnt += 1
            
        avg_loss = tot_loss / max(1, cnt)
        
        # Validation
        model.eval()
        with torch.no_grad():
            vt_x = torch.tensor(X_val, dtype=torch.float32).to(dev)
            vt_p = torch.tensor(P_val, dtype=torch.float32).to(dev)
            vp, _, _, _ = model(vt_x, vt_p)
            vp = vp.cpu().numpy().flatten()
            
            # [Robust] Safe PCC
            if len(y_val) > 1:
                val_pcc = float(pearsonr(y_val, vp)[0])
                if np.isnan(val_pcc): val_pcc = -1.0
            else:
                val_pcc = 0.0
        
        sched.step(avg_loss)

        if val_pcc > best_pcc:
            best_pcc = val_pcc; patience = 0
            torch.save(model.state_dict(), out / 'best_model.pt')
        else:
            patience += 1
            if patience >= 15: break
            
    # Final Eval
    best_model_path = out / 'best_model.pt'
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=dev)) # Removed weights_only=True for compatibility
    
    model.eval()
    with torch.no_grad():
        final_x = torch.tensor(X_te, dtype=torch.float32).to(dev)
        final_p = torch.tensor(P_te, dtype=torch.float32).to(dev)
        preds, priors, _, _ = model(final_x, final_p)
        preds = preds.cpu().numpy().flatten()
        priors = priors.cpu().numpy().reshape(-1)

    # [Robust] Final Check
    if np.isnan(preds).any():
        preds = np.nan_to_num(preds, nan=np.nanmean(preds))

    pcc = float(pearsonr(y_te, preds)[0]) if len(preds) > 1 else 0.0
    try:
        mse = float(mean_squared_error(y_te, preds))
    except ValueError:
        mse = -1.0
        
    nz = int(np.sum(np.abs(priors) > 1e-3))
    
    pd.DataFrame({'IID': te_ids, 'True': y_te, 'Pred': preds}).to_csv(out / 'pred.csv', index=False)
    with open(out / 'stats.json', 'w') as f: 
        json.dump({'pcc_test': pcc, 'mse': mse, 'sparsity': nz}, f, indent=2)
    
    return {'pcc': pcc, 'mse': mse}
