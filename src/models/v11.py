#!/usr/bin/env python3
import os
from pathlib import Path
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
import multiprocessing

# ==========================================
# Bio-Master v11: Stable Scaled Linear
# Key Fix: Target Normalization (Standardizing Y) to fix MSE explosion
# ==========================================

class BioScaler(nn.Module):
    """
    Predicts SNP importance scaling factor.
    Initialized to be very close to 1.0.
    """
    def __init__(self, delta_dim, gene_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(delta_dim + gene_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Tanh() 
        )
        
    def forward(self, d_emb, g_emb):
        priors = torch.cat([d_emb, g_emb], dim=1)
        # Output range: 1.0 +/- 0.5
        # Allows suppressing a SNP (0.5x) or boosting it (1.5x)
        return 1.0 + (self.net(priors) * 0.5)

class StableBioLinear(nn.Module):
    def __init__(self, delta_E: np.ndarray, gene_E: np.ndarray):
        super().__init__()
        
        self.register_buffer('delta_E', torch.tensor(delta_E, dtype=torch.float32))
        self.register_buffer('gene_E', torch.tensor(gene_E, dtype=torch.float32))
        
        n_snps = self.delta_E.shape[0]
        delta_dim = self.delta_E.shape[1]
        gene_dim = self.gene_E.shape[1]
        
        # 1. The Deep Scaler
        self.scaler = BioScaler(delta_dim, gene_dim)
        
        # 2. The Base Linear Model (Standard nn.Linear for stability)
        # We turn off bias here because we handle bias globally or via centering
        self.base_linear = nn.Linear(n_snps, 1, bias=False)
        
        # Global bias
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, X):
        # 1. Get Scales from Biology [N_SNPs, 1]
        scales = self.scaler(self.delta_E, self.gene_E)
        
        # 2. Apply Scales to the Weights
        # w_effective: [1, N_SNPs] * [N_SNPs, 1] (Broadcasting correctly?)
        # nn.Linear weights are [Out, In] -> [1, N_SNPs]
        # scales are [N_SNPs, 1]. Transpose scales to match weight shape.
        w_effective = self.base_linear.weight * scales.t()
        
        # 3. Predict: y = X @ w.T + b
        return F.linear(X, w_effective) + self.bias, w_effective.t()

class _DS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

def _load_pheno_map(pheno_file, trait):
    df = pd.read_csv(pheno_file, sep='\t' if pheno_file.endswith('.tsv') else ',')
    id_col = next((c for c in df.columns if c.lower() in ['id', 'iid', 'sample_id', 'sample']), df.columns[0])
    df[id_col] = df[id_col].astype(str)
    return dict(zip(df[id_col], df[trait]))

def train(plink_prefix, pheno_file, train_ids, test_ids, trait, delta_path, gene_path, out_dir,
          lr=1e-3, batch_size=64, epochs=150, lambda_l1=0.005, device='auto', **kwargs):
    
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    if device == 'auto': dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    else: dev = device
    print(f"Training Bio-Master v11 (Stable) on {dev}")
    
    # CPU Optimization
    num_workers, persistent_workers, pin_memory = 0, False, False
    if dev == 'cpu':
        try:
            n_cores = multiprocessing.cpu_count()
            num_workers = min(8, max(2, n_cores // 2))
            persistent_workers = True
        except: num_workers = 2
    else:
        num_workers, pin_memory = 4, True

    # Load Data
    delta = np.load(delta_path); gene = np.load(gene_path)
    if delta.shape[0] != gene.shape[0]:
        m = min(delta.shape[0], gene.shape[0])
        delta, gene = delta[:m], gene[:m]

    (bim, fam, bed) = read_plink(plink_prefix, verbose=False)
    G = bed.compute().T
    if G.shape[1] != delta.shape[0]:
        G = G[:, :min(G.shape[1], delta.shape[0])]
        delta, gene = delta[:G.shape[1]], gene[:G.shape[1]]

    ymap = _load_pheno_map(pheno_file, trait)
    iid2idx = dict(zip(fam['iid'].astype(str), range(len(fam))))
    tr_df = pd.read_csv(train_ids, sep='\t', names=['FID', 'IID'])
    te_df = pd.read_csv(test_ids, sep='\t', names=['FID', 'IID'])
    
    tr_idx = [iid2idx[str(x)] for x in tr_df['IID'] if str(x) in iid2idx]
    te_idx = [iid2idx[str(x)] for x in te_df['IID'] if str(x) in iid2idx]
    
    X_tr_raw, y_tr_raw = G[tr_idx], np.array([ymap.get(str(x), np.nan) for x in tr_df['IID'] if str(x) in iid2idx])
    X_te_raw, y_te_raw = G[te_idx], np.array([ymap.get(str(x), np.nan) for x in te_df['IID'] if str(x) in iid2idx])
    
    mtr, mte = ~np.isnan(y_tr_raw), ~np.isnan(y_te_raw)
    X_tr, y_tr = X_tr_raw[mtr], y_tr_raw[mtr]
    X_te, y_te = X_te_raw[mte], y_te_raw[mte]

    # --- CRITICAL FIX: SCALING Y ---
    # We must scale Y to have mean=0, std=1 for the optimizer to work reliably
    # otherwise MSE=142 kills the gradients.
    y_scaler = StandardScaler()
    y_tr_scaled = y_scaler.fit_transform(y_tr.reshape(-1, 1)).flatten()
    # Note: We do NOT scale X_te's Y, we will inverse_transform predictions instead
    
    # Scale X
    x_scaler = StandardScaler()
    X_tr = x_scaler.fit_transform(X_tr)
    X_te = x_scaler.transform(X_te)

    # Init Model
    model = StableBioLinear(delta, gene).to(dev)
    
    # Force scaler to start neutral (Scale ~ 1.0)
    nn.init.uniform_(model.scaler.net[-2].weight, -0.001, 0.001)
    nn.init.zeros_(model.scaler.net[-2].bias)

    # Optimizer (Increased LR slightly because data is normalized)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    crit = nn.MSELoss()

    loader = DataLoader(_DS(X_tr, y_tr_scaled), batch_size=batch_size, shuffle=True, 
                        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    
    best = float('inf'); patience = 0
    
    for ep in range(epochs):
        model.train(); tot = 0; cnt = 0
        for xb, yb in loader:
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            opt.zero_grad()
            
            # Predict (in Scaled Y space)
            pred, w_eff = model(xb)
            mse = crit(pred, yb)
            
            # L1 on weights
            l1 = torch.norm(w_eff, 1)
            loss = mse + (lambda_l1 * l1)
            
            loss.backward(); opt.step()
            tot += mse.item(); cnt += 1
            
        avg = tot / max(1, cnt); sched.step(avg)
        if avg < best:
            best = avg; patience = 0
            torch.save(model.state_dict(), out / 'best_model.pt')
        else:
            patience += 1
        if patience >= 15: break

    # Eval
    model.load_state_dict(torch.load(out / 'best_model.pt', map_location=dev))
    model.eval()
    test_loader = DataLoader(_DS(X_te, y_te), batch_size=256, shuffle=False, num_workers=num_workers)
    preds, final_weights = [], None
    
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(dev)
            p_scaled, w = model(xb)
            
            # Inverse Transform predictions back to original scale (e.g. Backfat mm)
            p_orig = y_scaler.inverse_transform(p_scaled.cpu().numpy())
            preds.append(p_orig.flatten())
            
            if final_weights is None: 
                final_weights = w.cpu().numpy().flatten()
            
    p_te = np.concatenate(preds)
    
    # Calculate Metrics on Original Scale
    pcc = float(pearsonr(y_te, p_te)[0]) if len(y_te) > 1 else 0.0
    mse = float(mean_squared_error(y_te, p_te))
    nz = int(np.sum(np.abs(final_weights) > 1e-3))
    
    pd.DataFrame({'IID': [str(x) for x in te_df['IID'][mte]], 'True': y_te, 'Pred': p_te}).to_csv(out / 'df_gsf_v5_pred.csv', index=False)
    with open(out / 'DF_GSF_v5_stats.json', 'w') as f:
        json.dump({'pcc_test': pcc, 'mse': mse, 'sparsity': nz, 'model': 'bio_master_v11'}, f, indent=2)
    
    return {'pcc_test': pcc}
