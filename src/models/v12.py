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
# Bio-Master v12: Bio-Factorization Machine (BioFM)
# 
# Philosophy: 
# 1. Linear Baseline (GBLUP)
# 2. Factorization Machine for 2nd-order Epistasis (using Biological Latent Vectors)
# 3. Deep Component for High-order Non-linearity
# ==========================================

class PriorProjector(nn.Module):
    """
    Projects high-dimensional biological priors (Delta + Gene) 
    into a low-dimensional Latent Vector space for FM interactions.
    """
    def __init__(self, input_dim, latent_dim=16, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, latent_dim),
            nn.Tanh() # Bound latent vectors to [-1, 1] for stability
        )
        
    def forward(self, priors):
        # Input: [N_SNPs, Prior_Dim]
        # Output: [N_SNPs, Latent_Dim] -> Matrix V
        return self.net(priors)

class BioFM(nn.Module):
    def __init__(self, delta_E: np.ndarray, gene_E: np.ndarray, latent_dim=16):
        super().__init__()
        
        # --- 1. Register Priors ---
        self.register_buffer('delta_E', torch.tensor(delta_E, dtype=torch.float32))
        self.register_buffer('gene_E', torch.tensor(gene_E, dtype=torch.float32))
        
        n_snps = self.delta_E.shape[0]
        prior_dim = self.delta_E.shape[1] + self.gene_E.shape[1]
        
        # --- 2. Linear Component (GBLUP Baseline) ---
        # y = w * x + b
        # We perform standard linear regression here.
        self.linear = nn.Linear(n_snps, 1)
        
        # --- 3. FM Component (2nd Order Interactions) ---
        # Latent vectors V are generated from Priors, not learned from scratch.
        # This injects biological inductive bias into the interaction term.
        self.prior_projector = PriorProjector(prior_dim, latent_dim=latent_dim)
        
        # Learnable scale for FM part (Initialize to 0 to start as Linear model)
        self.fm_scale = nn.Parameter(torch.zeros(1))
        
        # --- 4. Deep Component (High-order Non-linearity) ---
        # Compresses the latent representation of the individual
        self.deep_mlp = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
        
        # Learnable scale for Deep part (Initialize to 0)
        self.deep_scale = nn.Parameter(torch.zeros(1))

    def forward(self, X):
        """
        X: [Batch, N_SNPs]
        """
        # --- Part A: Linear (1st Order) ---
        out_linear = self.linear(X)
        
        # --- Part B: Factorization Machine (2nd Order) ---
        # 1. Generate Latent Matrix V from Priors: [N_SNPs, K]
        #    concat priors along feature dimension
        priors = torch.cat([self.delta_E, self.gene_E], dim=1)
        V = self.prior_projector(priors)
        
        # 2. Compute FM term using O(N) trick:
        #    sum_interactions = 0.5 * [ (sum(vx))^2 - sum(v^2 x^2) ]
        
        # Term 1: (X @ V)^2 -> [Batch, N] @ [N, K] -> [Batch, K] -> Square
        term1 = torch.pow(torch.matmul(X, V), 2)
        
        # Term 2: (X^2) @ (V^2) -> [Batch, N] @ [N, K] -> [Batch, K]
        term2 = torch.matmul(torch.pow(X, 2), torch.pow(V, 2))
        
        # FM Output: Sum over latent dimension K -> [Batch, 1]
        out_fm = 0.5 * torch.sum(term1 - term2, dim=1, keepdim=True)
        
        # --- Part C: Deep Component (High Order) ---
        # We reuse the "Sum(VX)" embedding as a compact representation of the individual
        # Interaction Embedding: [Batch, K]
        interaction_emb = torch.matmul(X, V) 
        out_deep = self.deep_mlp(interaction_emb)
        
        # --- Final Sum ---
        # Y = Linear + alpha * FM + beta * Deep
        return out_linear + (self.fm_scale * out_fm) + (self.deep_scale * out_deep)

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
          lr=5e-4, batch_size=64, epochs=150, lambda_l1=0.005, device='auto', **kwargs):
    
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    if device == 'auto': dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    else: dev = device
    print(f"Training Bio-Master v12 (BioFM) on {dev}")
    
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

    print(f"Feature Dims: Delta={delta.shape[1]}, Gene={gene.shape[1]}")

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

    # --- CRITICAL: STANDARDIZE Y ---
    # Fixes MSE explosion and gradient instability
    y_scaler = StandardScaler()
    y_tr_scaled = y_scaler.fit_transform(y_tr.reshape(-1, 1)).flatten()
    
    # Scale X
    x_scaler = StandardScaler()
    X_tr = x_scaler.fit_transform(X_tr)
    X_te = x_scaler.transform(X_te)

    # Init Model
    model = BioFM(delta, gene, latent_dim=16).to(dev)
    
    # Optimizer
    # We use a slightly higher LR for the linear part, lower for deep? 
    # For simplicity, global LR with weight decay
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
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
            
            pred = model(xb)
            mse = crit(pred, yb)
            
            # Regularization:
            # 1. L1 on Linear Weights (Sparsity)
            # 2. L2 on Latent Vectors (Prevent large interactions) - handled by weight_decay
            l1_linear = torch.norm(model.linear.weight, 1)
            
            loss = mse + (lambda_l1 * l1_linear)
            
            loss.backward()
            opt.step()
            tot += mse.item(); cnt += 1
            
        avg = tot / max(1, cnt); sched.step(avg)
        
        # Monitoring the scales to see if model is using FM/Deep
        if ep % 10 == 0:
            fm_s = model.fm_scale.item()
            dp_s = model.deep_scale.item()
            # print(f"Ep {ep}: Loss={avg:.4f} | FM_Scale={fm_s:.4f} | Deep_Scale={dp_s:.4f}")

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
    preds = []
    
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(dev)
            p_scaled = model(xb)
            # Inverse Transform
            p_orig = y_scaler.inverse_transform(p_scaled.cpu().numpy())
            preds.append(p_orig.flatten())
            
    p_te = np.concatenate(preds)
    
    pcc = float(pearsonr(y_te, p_te)[0]) if len(y_te) > 1 else 0.0
    mse = float(mean_squared_error(y_te, p_te))
    
    # Check contribution of interaction
    fm_contribution = model.fm_scale.item()
    
    pd.DataFrame({'IID': [str(x) for x in te_df['IID'][mte]], 'True': y_te, 'Pred': p_te}).to_csv(out / 'df_gsf_v5_pred.csv', index=False)
    with open(out / 'DF_GSF_v5_stats.json', 'w') as f:
        json.dump({
            'pcc_test': pcc, 
            'mse': mse, 
            'model': 'bio_master_v12_biofm',
            'fm_scale': fm_contribution
        }, f, indent=2)
    
    return {'pcc_test': pcc}