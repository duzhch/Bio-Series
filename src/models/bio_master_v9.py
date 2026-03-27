#!/usr/bin/env python3
import os
import json
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
# 1. Hybrid Loss Function (The Soul of V9)
# ==========================================

class HybridPCCLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.1):
        super().__init__()
        self.alpha = alpha # PCC Weight (Correlation)
        self.beta = beta   # ListNet Weight (Ranking)
        self.gamma = gamma # Orthogonality Weight (Disentanglement)

    def pcc_loss(self, pred, target):
        """ 1 - PCC. Optimizes for linear correlation. """
        vx = pred - torch.mean(pred)
        vy = target - torch.mean(target)
        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
        return 1 - cost

    def listnet_loss(self, pred, target):
        """ 
        ListNet Top-k Ranking Loss. 
        Focuses on: "Who is the top performer?" rather than exact value.
        """
        # Temperature scaling can be added to softmax if needed, typically T=1 is fine
        P_y_true = F.softmax(target, dim=0)
        P_y_pred = F.softmax(pred, dim=0)
        return -torch.sum(P_y_true * torch.log(P_y_pred + 1e-8))

    def forward(self, pred, target, deep_feat, context_feat):
        # Flatten tensors for calculation
        p = pred.squeeze()
        t = target.squeeze()
        
        loss_pcc = self.pcc_loss(p, t)
        loss_rank = self.listnet_loss(p, t)
        
        # --- Orthogonality Penalty ---
        # Forces Deep features to be independent of Population Structure (Context)
        if deep_feat is not None and context_feat is not None:
            # We use .detach() on context_feat because we want Deep tower to move away from Context,
            # not Context to move away from Deep (Context is ground truth for structure).
            # Calculating cosine similarity along batch dimension (dim=0)
            orth_loss = torch.mean(torch.abs(F.cosine_similarity(deep_feat, context_feat.detach(), dim=0)))
        else:
            orth_loss = torch.tensor(0.0, device=pred.device)

        # Total Loss
        return self.alpha * loss_pcc + self.beta * loss_rank + self.gamma * orth_loss

# ==========================================
# 2. Bio-Master V9 Architecture
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

class LDBlockEncoder(nn.Module):
    def __init__(self, block_size, in_channels=2, embed_dim=32):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, embed_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(embed_dim)
        self.rare_project = nn.Linear(block_size * in_channels, embed_dim)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        B, C, L = x.shape
        h_conv = self.act(self.bn1(self.conv1(x))) 
        h_conv_pool = torch.max(h_conv, dim=2)[0]
        x_flat = x.view(B, -1)
        h_rare = self.rare_project(x_flat)
        return self.dropout(h_conv_pool + h_rare)

class BioMasterV9(nn.Module):
    def __init__(self, delta_E, gene_E, num_snps, num_pcs, block_size=100):
        super().__init__()
        self.register_buffer('delta_E', torch.tensor(delta_E, dtype=torch.float32))
        self.register_buffer('gene_E', torch.tensor(gene_E, dtype=torch.float32))
        
        # --- Deep Tower ---
        self.prior_gen = PriorGenerator(delta_E.shape[1], gene_E.shape[1])
        self.block_size = block_size
        self.pad_len = (block_size - (num_snps % block_size)) % block_size
        self.n_blocks = (num_snps + self.pad_len) // block_size
        self.embed_dim = 64
        self.encoder = LDBlockEncoder(block_size, 2, self.embed_dim)
        
        # [ARCH UPDATE] Deep Head: Output 64-dim feature for orthogonality check
        self.deep_feat_extractor = nn.Sequential(
            nn.Linear(self.embed_dim * self.n_blocks, 128),
            nn.LayerNorm(128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, 64), # Bottleneck for feature alignment
            nn.ReLU(inplace=True)
        )
        self.deep_out = nn.Linear(64, 1) # Final prediction layer
        
        # --- Context Tower ---
        # [ARCH UPDATE] Context Head: Also output 64-dim feature
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
        
        # 1. Wide Output
        out_wide = self.wide_linear(X_snps)
        
        # 2. Context Flow
        # Extract features first (for Loss), then predict
        feat_ctx = self.context_feat_extractor(X_pcs) # Shape: (B, 64)
        out_ctx = self.context_out(feat_ctx)
        
        # 3. Deep Flow
        priors = self.prior_gen(self.delta_E, self.gene_E)
        if self.pad_len > 0:
            X_p = F.pad(X_snps, (0, self.pad_len))
            P_p = F.pad(priors.t(), (0, self.pad_len)).t()
        else:
            X_p, P_p = X_snps, priors
            
        g_blocks = X_p.view(B, self.n_blocks, self.block_size)
        p_blocks = P_p.view(1, self.n_blocks, self.block_size).expand(B, -1, -1)
        
        x_folded = torch.stack([g_blocks, p_blocks], dim=2).view(-1, 2, self.block_size)
        enc_out = self.encoder(x_folded)
        
        # Extract features (for Loss)
        feat_deep = self.deep_feat_extractor(enc_out.view(B, -1)) # Shape: (B, 64)
        out_deep = self.deep_out(feat_deep)
        
        # Final Sum
        final_pred = out_wide + out_ctx + out_deep
        
        # Return 4 values now: Pred, Attention Priors, Deep Feat, Context Feat
        return final_pred, priors[:N], feat_deep, feat_ctx

# ==========================================
# 3. Data & Training Utilities
# ==========================================

class _DS(Dataset):
    def __init__(self, X, P, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.P = torch.tensor(P, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.P[i], self.y[i]

def load_pca(pca_file, valid_ids):
    if not os.path.exists(pca_file):
        print("⚠️  Warning: PCA file missing. Falling back to zero vectors.")
        return np.zeros((len(valid_ids), 10))
    try:
        df = pd.read_csv(pca_file)
        df['IID'] = df['IID'].astype(str)
        pc_cols = [c for c in df.columns if c.startswith('PC')]
        if not pc_cols: return np.zeros((len(valid_ids), 10))
        pca_map = df.set_index('IID')[pc_cols].to_dict('index')
        zero_vec = [0.0] * len(pc_cols)
        return np.array([list(pca_map.get(iid, zero_vec).values()) for iid in valid_ids])
    except:
        return np.zeros((len(valid_ids), 10))

def _load_pheno_map(pheno_file, trait):
    df = pd.read_csv(pheno_file, sep='\t' if pheno_file.endswith('.tsv') else ',')
    id_col = next((c for c in df.columns if c.lower() in ['iid', 'sample_id', 'id']), df.columns[0])
    df[id_col] = df[id_col].astype(str)
    return dict(zip(df[id_col], df[trait]))

def train(plink_prefix, pheno_file, train_ids, test_ids, trait, delta_path, gene_path, out_dir, 
          lr=5e-4, batch_size=64, epochs=150, lambda_l1=0.005, device='auto'):
    
    out = Path(out_dir)
    # Ensure output directory exists
    out.mkdir(parents=True, exist_ok=True)
    
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Training Bio-Master V9 (Wide&Deep + HybridLoss) on {dev}")

    # Load Data (Standard Process)
    delta = np.load(delta_path); gene = np.load(gene_path)
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
    
    # Load PCA
    pca_path = out / "global_pca_features.csv"
    P_tr = load_pca(pca_path, tr_ids); P_te = load_pca(pca_path, te_ids)
    
    # Scale
    sc_g = StandardScaler(); X_tr = sc_g.fit_transform(X_tr); X_te = sc_g.transform(X_te)
    sc_p = StandardScaler(); P_tr = sc_p.fit_transform(P_tr); P_te = sc_p.transform(P_te)
    
    # Init Model
    model = BioMasterV9(delta, gene, num_snps=X_tr.shape[1], num_pcs=P_tr.shape[1]).to(dev)
    
    # Optimizer
    base_params = [p for n, p in model.named_parameters() if 'wide' not in n]
    wide_params = [p for n, p in model.named_parameters() if 'wide' in n]
    opt = optim.AdamW([
        {'params': base_params, 'weight_decay': 1e-3},
        {'params': wide_params, 'weight_decay': 0.1} 
    ], lr=lr)
    
    # Scheduler
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    
    # [CRITICAL UPDATE] Using HybridPCCLoss instead of MSE
    # alpha=1.0 (PCC), beta=0.5 (Rank), gamma=0.1 (Orthogonality)
    crit = HybridPCCLoss(alpha=1.0, beta=0.5, gamma=0.1)

    loader = DataLoader(_DS(X_tr, P_tr, y_tr), batch_size=batch_size, shuffle=True, 
                        num_workers=4 if dev=='cuda' else 0, drop_last=True) 
                        # drop_last=True ensures consistent batch size for ranking loss stability

    best_pcc = -1.0; patience = 0 # Track PCC now, not MSE
    
    for ep in range(epochs):
        model.train()
        tot_loss = 0; cnt = 0
        
        for x, p, y in loader:
            x, p, y = x.to(dev), p.to(dev), y.to(dev)
            opt.zero_grad()
            
            # [UPDATE] Unpack 4 values
            pred, priors, feat_deep, feat_ctx = model(x, p)
            
            # [UPDATE] Calculate Hybrid Loss
            # Note: L1 penalty on priors is separate from HybridLoss logic
            main_loss = crit(pred, y, feat_deep, feat_ctx)
            l1_loss = lambda_l1 * torch.norm(priors, 1)
            loss = main_loss + l1_loss
            
            loss.backward()
            opt.step()
            tot_loss += loss.item(); cnt += 1
            
        avg_loss = tot_loss / max(1, cnt)
        
        # Validation
        model.eval()
        with torch.no_grad():
            vt_x = torch.tensor(X_te, dtype=torch.float32).to(dev)
            vt_p = torch.tensor(P_te, dtype=torch.float32).to(dev)
            # Forward pass validation
            vp, _, _, _ = model(vt_x, vt_p)
            vp = vp.cpu().numpy().flatten()
            
            # Calculate PCC for monitoring (primary metric)
            # FIX: Handle potential NaN returns from pearsonr or very small datasets
            if len(y_te) > 1:
                val_pcc = float(pearsonr(y_te, vp)[0])
                if np.isnan(val_pcc): 
                    val_pcc = -1.0
            else:
                val_pcc = 0.0
        
        # Scheduler steps on Loss
        sched.step(avg_loss)

        # Save Best Model based on PCC (Target Metric)
        if val_pcc > best_pcc:
            best_pcc = val_pcc; patience = 0
            torch.save(model.state_dict(), out / 'best_model.pt')
        else:
            patience += 1
            if patience >= 15: break
            
    # Final Eval - FIX APPLIED HERE
    best_model_path = out / 'best_model.pt'
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=dev))
        print(f"✅ Loaded best model from validation (PCC: {best_pcc:.4f})")
    else:
        print(f"⚠️  Warning: No best model saved (Validation PCC never > {best_pcc}). Using final epoch weights.")

    model.eval()
    with torch.no_grad():
        final_x = torch.tensor(X_te, dtype=torch.float32).to(dev)
        final_p = torch.tensor(P_te, dtype=torch.float32).to(dev)
        preds, w, _, _ = model(final_x, final_p)
        preds = preds.cpu().numpy().flatten()
        w_mean = w.cpu().numpy().mean(axis=0)

    pcc = float(pearsonr(y_te, preds)[0])
    mse = float(mean_squared_error(y_te, preds))
    nz = int(np.sum(np.abs(w_mean) > 1e-3))
    
    pd.DataFrame({'IID': te_ids, 'True': y_te, 'Pred': preds}).to_csv(out / 'pred.csv', index=False)
    with open(out / 'stats.json', 'w') as f: 
        json.dump({'pcc_test': pcc, 'mse': mse, 'sparsity': nz}, f, indent=2)
    
    return {'pcc': pcc, 'mse': mse}