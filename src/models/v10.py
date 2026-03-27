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
from scipy.stats import pearsonr
from torch.utils.data import Dataset, DataLoader
from pandas_plink import read_plink
import multiprocessing

# ==========================================
# Bio-Master v11: Fixed Scaling Issues
# ==========================================
# FIX LOG:
# 1. Standardize Phenotypes (y): Crucial for Ridge Regression convergence.
# 2. Monitor Output Mean: Check if model predicts 0 or actual values.
# 3. Revert to MSELoss: Standardized data works best with MSE.

class BioFeatureScaler(nn.Module):
    """
    The 'Advisor' Network.
    Output: A scaling factor for each SNP.
    Initialized to strict 1.0.
    """
    def __init__(self, delta_dim, gene_dim, hidden_dim=32):
        super().__init__()
        input_dim = delta_dim + gene_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU()
        )
        self.to_scale = nn.Linear(hidden_dim // 2, 1)
        
        # Zero Init -> Output 0 -> Tanh(0)=0 -> Scale=1.0
        nn.init.zeros_(self.to_scale.weight)
        nn.init.zeros_(self.to_scale.bias)

    def forward(self, delta_E, gene_E):
        x = torch.cat([delta_E, gene_E], dim=1)
        feat = self.net(x)
        raw_score = self.to_scale(feat)
        # Scale range: (0.5, 1.5) - constrained range to prevent explosion
        scale = 1.0 + 0.5 * torch.tanh(raw_score) 
        return scale

class ScaledRidgeModel(nn.Module):
    def __init__(self, delta_E, gene_E):
        super().__init__()
        self.register_buffer('delta_E', torch.tensor(delta_E, dtype=torch.float32))
        self.register_buffer('gene_E', torch.tensor(gene_E, dtype=torch.float32))
        num_snps = delta_E.shape[0]
        
        self.scaler = BioFeatureScaler(delta_E.shape[1], gene_E.shape[1])
        self.linear = nn.Linear(num_snps, 1)
        
        # Init linear weights small
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        # x: [Batch, N_SNPs]
        snp_scales = self.scaler(self.delta_E, self.gene_E) # [N, 1]
        x_scaled = x * snp_scales.t() 
        y_pred = self.linear(x_scaled)
        return y_pred, snp_scales

class _DS_Simple(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        # Standardize Genotypes
        mean = self.X.mean(dim=0)
        std = self.X.std(dim=0) + 1e-6
        self.X = (self.X - mean) / std
        
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

def _load_pheno_map(pheno_file, trait):
    df = pd.read_csv(pheno_file, sep='\t' if pheno_file.endswith('.tsv') else ',')
    id_col = next((c for c in df.columns if c.lower() in ['id', 'iid', 'sample_id', 'sample']), df.columns[0])
    df[id_col] = df[id_col].astype(str)
    return dict(zip(df[id_col], df[trait]))

def train(plink_prefix, pheno_file, train_ids, test_ids, trait, delta_path, gene_path, out_dir,
          lr=1e-3, batch_size=64, epochs=150, lambda_l1=0.0, device='auto', **kwargs):
    
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    if device == 'auto': dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    else: dev = device
    print(f"[Bio-Master v11] Strategy: Y-Standardization Fix. Device: {dev}")
    
    # --- Load Data ---
    delta = np.load(delta_path)
    gene = np.load(gene_path)
    (bim, fam, bed) = read_plink(plink_prefix, verbose=False)
    G = bed.compute().T
    
    # Align
    min_len = min(G.shape[1], delta.shape[0])
    G = G[:, :min_len]
    delta, gene = delta[:min_len], gene[:min_len]

    # Pheno
    ymap = _load_pheno_map(pheno_file, trait)
    iid2idx = dict(zip(fam['iid'].astype(str), range(len(fam))))
    tr_df = pd.read_csv(train_ids, sep='\t', names=['FID', 'IID'])
    te_df = pd.read_csv(test_ids, sep='\t', names=['FID', 'IID'])
    
    tr_idx = [iid2idx[str(x)] for x in tr_df['IID'] if str(x) in iid2idx]
    te_idx = [iid2idx[str(x)] for x in te_df['IID'] if str(x) in iid2idx]
    
    X_tr = G[tr_idx]
    y_tr = np.array([ymap.get(str(x), np.nan) for x in tr_df['IID'] if str(x) in iid2idx])
    X_te = G[te_idx]
    y_te = np.array([ymap.get(str(x), np.nan) for x in te_df['IID'] if str(x) in iid2idx])
    
    mtr, mte = ~np.isnan(y_tr), ~np.isnan(y_te)
    X_tr, y_tr = X_tr[mtr], y_tr[mtr]
    X_te, y_te = X_te[mte], y_te[mte]
    X_tr[np.isnan(X_tr)] = 0; X_te[np.isnan(X_te)] = 0

    # --- CRITICAL FIX: STANDARDIZE Y ---
    print(f"Raw Phenotype Mean: {y_tr.mean():.4f}, Std: {y_tr.std():.4f}")
    y_mean = y_tr.mean()
    y_std = y_tr.std() + 1e-6
    
    y_tr_scaled = (y_tr - y_mean) / y_std
    y_te_scaled = (y_te - y_mean) / y_std # Use Train stats to scale Test
    print("Phenotypes standardized to Mean=0, Std=1")
    # -----------------------------------

    train_ds = _DS_Simple(X_tr, y_tr_scaled)
    test_ds = _DS_Simple(X_te, y_te_scaled) # Pass scaled y to test ds for loss calculation
    
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)

    model = ScaledRidgeModel(delta, gene).to(dev)
    
    # Optimizer
    opt = optim.AdamW([
        {'params': model.linear.parameters(), 'lr': 2e-3, 'weight_decay': 0.05}, # Higher decay for Ridge
        {'params': model.scaler.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4}
    ])
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    
    # Use MSELoss on standardized data
    crit = nn.MSELoss()

    best = float('inf'); patience = 0
    
    for ep in range(epochs):
        model.train(); tot = 0; cnt = 0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            
            opt.zero_grad()
            pred, scales = model(xb)
            
            reg_loss = crit(pred, yb)
            # Regularize scales to stay near 1.0
            scale_loss = 0.05 * torch.mean((scales - 1.0)**2)
            
            loss = reg_loss + scale_loss
            loss.backward()
            opt.step()
            tot += reg_loss.item(); cnt += 1
            
        avg = tot / max(1, cnt); sched.step(avg)
        
        if ep % 10 == 0:
            s_std = scales.std().item()
            # If Loss is around 1.0, it means model is guessing mean.
            # If Loss < 1.0 (e.g., 0.7), model is learning.
            print(f"Ep {ep}: MSE={avg:.4f} (Base=1.0) | Scale_Std={s_std:.4f}")

        if avg < best:
            best = avg; patience = 0
            torch.save(model.state_dict(), out / 'best_model.pt')
        else:
            patience += 1
        if patience >= 15: break

    # Eval
    model.load_state_dict(torch.load(out / 'best_model.pt', map_location=dev))
    model.eval()
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    preds = []
    
    with torch.no_grad():
        final_scales = model.scaler(model.delta_E, model.gene_E).cpu().numpy().flatten()
        for xb, yb in test_loader:
            xb = xb.to(dev)
            p, _ = model(xb)
            preds.append(p.cpu().numpy().flatten())
            
    p_scaled = np.concatenate(preds)
    
    # Calculate PCC on SCALED values (Invariant)
    pcc = float(pearsonr(y_te_scaled, p_scaled)[0]) if len(y_te) > 1 else 0.0
    
    # Calculate MSE on ORIGINAL scale (Recovered)
    p_orig = p_scaled * y_std + y_mean
    mse_orig = float(mean_squared_error(y_te, p_orig))
    
    print(f"[Bio-Master v11] PCC={pcc:.4f}, MSE(Original)={mse_orig:.2f}")
    
    # Save results
    pd.DataFrame({'IID': [str(x) for x in te_df['IID'][mte]], 'True': y_te, 'Pred': p_orig}).to_csv(out / 'df_gsf_v11_pred.csv', index=False)
    
    with open(out / 'DF_GSF_v11_stats.json', 'w') as f:
        json.dump({
            'pcc_test': pcc, 
            'mse': mse_orig, 
            'model': 'bio_master_v11_standardized'
        }, f, indent=2)
    
    return {'pcc_test': pcc}