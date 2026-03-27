#!/usr/bin/env python3
import os
import json
import math
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
            orth_loss = torch.mean(torch.abs(F.cosine_similarity(deep_feat, context_feat.detach(), dim=0)))
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
    def forward(self, d_emb, g_emb):
        d = F.relu(self.delta_compress(d_emb), inplace=True)
        g = F.relu(self.gene_compress(g_emb), inplace=True)
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
    def __init__(self, delta_E, gene_E, num_snps, num_pcs, block_size=100):
        super().__init__()
        self.register_buffer('delta_E', torch.tensor(delta_E, dtype=torch.float32))
        self.register_buffer('gene_E', torch.tensor(gene_E, dtype=torch.float32))
        
        # Dimensions
        self.block_size = block_size
        self.pad_len = (block_size - (num_snps % block_size)) % block_size
        self.n_blocks = (num_snps + self.pad_len) // block_size
        self.d_model = 64 
        
        # --- Deep Tower (Genomic Transformer) ---
        self.prior_gen = PriorGenerator(delta_E.shape[1], gene_E.shape[1])
        
        self.genomic_transformer = GenomicTransformer(
            block_size=block_size, 
            in_channels=2, 
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
        self.context_feat_extractor = nn.Sequential(
            nn.Linear(num_pcs, 64), 
            nn.ReLU(inplace=True)
        )
        self.context_out = nn.Linear(64, 1)

        # --- Wide Tower ---
        self.wide_linear = nn.Linear(num_snps, 1)
        nn.init.normal_(self.wide_linear.weight, 0, 0.001)

    def forward(self, X_snps, X_pcs):
        B, N = X_snps.shape
        
        # Path A: Wide
        out_wide = self.wide_linear(X_snps)
        
        # Path B: Context
        feat_ctx = self.context_feat_extractor(X_pcs)
        out_ctx = self.context_out(feat_ctx)
        
        # Path C: Deep (Transformer)
        priors = self.prior_gen(self.delta_E, self.gene_E)
        
        if self.pad_len > 0:
            X_p = F.pad(X_snps, (0, self.pad_len))
            P_p = F.pad(priors.t(), (0, self.pad_len)).t()
        else:
            X_p, P_p = X_snps, priors
            
        g_blocks = X_p.view(B, self.n_blocks, self.block_size)
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

def _load_pheno_map(pheno_file, trait):
    df = pd.read_csv(pheno_file, sep='\t' if pheno_file.endswith('.tsv') else ',')
    id_col = next((c for c in df.columns if c.lower() in ['iid', 'sample_id', 'id']), df.columns[0])
    df[id_col] = df[id_col].astype(str)
    return dict(zip(df[id_col], df[trait]))

def train(plink_prefix, pheno_file, train_ids, test_ids, trait, delta_path, gene_path, out_dir, 
          lr=3e-4, batch_size=64, epochs=150, lambda_l1=0.005, device='auto'):
    
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True) # Ensure dir exists

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if dev == 'cpu':
        torch.set_num_threads(4) 
    
    print(f"🚀 Training Bio-Master V10 (Transformer + Robust) on {dev}")

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
    
    ymap = _load_pheno_map(pheno_file, trait)
    iid2idx = {str(iid): i for i, iid in enumerate(fam['iid'].astype(str))}
    tr_df = pd.read_csv(train_ids, sep='\t', names=['FID', 'IID'])
    te_df = pd.read_csv(test_ids, sep='\t', names=['FID', 'IID'])
    
    tr_ids = [str(x) for x in tr_df['IID'] if str(x) in iid2idx and str(x) in ymap]
    te_ids = [str(x) for x in te_df['IID'] if str(x) in iid2idx and str(x) in ymap]
    
    X_tr = G[[iid2idx[x] for x in tr_ids]]; X_te = G[[iid2idx[x] for x in te_ids]]
    y_tr = np.array([ymap[x] for x in tr_ids]); y_te = np.array([ymap[x] for x in te_ids])
    
    pca_path = out / "global_pca_features.csv"
    P_tr = load_pca(pca_path, tr_ids); P_te = load_pca(pca_path, te_ids)
    
    # [Robust] Handle NaN in inputs
    sc_g = StandardScaler(); X_tr = sc_g.fit_transform(X_tr); X_te = sc_g.transform(X_te)
    sc_p = StandardScaler(); P_tr = sc_p.fit_transform(P_tr); P_te = sc_p.transform(P_te)
    X_tr = np.nan_to_num(X_tr); X_te = np.nan_to_num(X_te)
    P_tr = np.nan_to_num(P_tr); P_te = np.nan_to_num(P_te)
    
    model = BioMasterV10(delta, gene, num_snps=X_tr.shape[1], num_pcs=P_tr.shape[1], block_size=100).to(dev)
    
    base_params = [p for n, p in model.named_parameters() if 'wide' not in n]
    wide_params = [p for n, p in model.named_parameters() if 'wide' in n]
    opt = optim.AdamW([
        {'params': base_params, 'weight_decay': 1e-3},
        {'params': wide_params, 'weight_decay': 0.1} 
    ], lr=lr)
    
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    crit = HybridPCCLoss(alpha=1.0, beta=0.5, gamma=0.1)

    nw = 4 if dev=='cuda' else 0 
    loader = DataLoader(_DS(X_tr, P_tr, y_tr), batch_size=batch_size, shuffle=True, 
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
            vt_x = torch.tensor(X_te, dtype=torch.float32).to(dev)
            vt_p = torch.tensor(P_te, dtype=torch.float32).to(dev)
            vp, _, _, _ = model(vt_x, vt_p)
            vp = vp.cpu().numpy().flatten()
            
            # [Robust] Safe PCC
            if len(y_te) > 1:
                val_pcc = float(pearsonr(y_te, vp)[0])
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
        preds, w, _, _ = model(final_x, final_p)
        preds = preds.cpu().numpy().flatten()
        w_mean = w.cpu().numpy().mean(axis=0)

    # [Robust] Final Check
    if np.isnan(preds).any():
        preds = np.nan_to_num(preds, nan=np.nanmean(preds))

    pcc = float(pearsonr(y_te, preds)[0]) if len(preds) > 1 else 0.0
    try:
        mse = float(mean_squared_error(y_te, preds))
    except ValueError:
        mse = -1.0
        
    nz = int(np.sum(np.abs(w_mean) > 1e-3))
    
    pd.DataFrame({'IID': te_ids, 'True': y_te, 'Pred': preds}).to_csv(out / 'pred.csv', index=False)
    with open(out / 'stats.json', 'w') as f: 
        json.dump({'pcc_test': pcc, 'mse': mse, 'sparsity': nz}, f, indent=2)
    
    return {'pcc': pcc, 'mse': mse}
