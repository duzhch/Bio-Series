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
# Bio-Master v9: Wide & Deep Residual Architecture
# Philosophy: "Trust the Data (Wide), Consult the Priors (Deep)"
# ==========================================

class PriorCompressor(nn.Module):
    """
    Compresses high-dim priors (Delta=768, Gene=64) into a compact
    weight vector for each SNP.
    """
    def __init__(self, delta_dim, gene_dim, out_dim=32):
        super().__init__()
        # Compress priors to a manageable subspace
        self.compress = nn.Sequential(
            nn.Linear(delta_dim + gene_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_dim),
            nn.Tanh() # Tanh allows negative weights (suppression) and positive (activation)
        )
    
    def forward(self, d_emb, g_emb):
        # Concatenate raw priors
        raw_priors = torch.cat([d_emb, g_emb], dim=1)
        # Return [N_SNPs, out_dim]
        return self.compress(raw_priors)

class WideAndDeepModel(nn.Module):
    def __init__(self, delta_E: np.ndarray, gene_E: np.ndarray):
        super().__init__()
        
        # --- 1. Buffer Setup ---
        self.register_buffer('delta_E', torch.tensor(delta_E, dtype=torch.float32))
        self.register_buffer('gene_E', torch.tensor(gene_E, dtype=torch.float32))
        
        n_snps = self.delta_E.shape[0]
        delta_dim = self.delta_E.shape[1]
        gene_dim = self.gene_E.shape[1]
        
        # --- 2. Wide Component (The GBLUP Killer) ---
        # Direct linear mapping from SNPs to Phenotype.
        # This ensures the model performs AT LEAST as well as a linear model.
        # No activation, no dropout here. Raw additive genetics.
        self.wide = nn.Linear(n_snps, 1)
        
        # --- 3. Deep Component (The Enhancer) ---
        prior_embed_dim = 16
        data_embed_dim = 16
        
        # A. Prior Processing
        self.prior_compressor = PriorCompressor(delta_dim, gene_dim, out_dim=prior_embed_dim)
        
        # B. Data-Driven Bottleneck (Learn from X directly, ignoring priors)
        # Allows model to correct itself if priors are wrong.
        self.data_projector = nn.Linear(n_snps, data_embed_dim)
        
        # C. MLP for Non-linear Interactions (Epistasis)
        # Input: [Batch, prior_embed_dim + data_embed_dim]
        # We assume interactions happen in the latent space
        hidden_dim = 64
        self.deep_mlp = nn.Sequential(
            nn.Linear(prior_embed_dim + data_embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), # Stability
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), # High dropout to prevent overfitting on small data
            nn.Linear(hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )
        
        # Learnable scaling factor for the deep component
        # Initialize small so training starts linear-dominant
        self.deep_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, X):
        """
        X: [Batch, N_SNPs]
        """
        # --- Path 1: Wide (Linear) ---
        # The baseline prediction
        out_wide = self.wide(X)
        
        # --- Path 2: Deep (Non-linear) ---
        # A. Prepare Prior Weights [N_SNPs, 16]
        w_priors = self.prior_compressor(self.delta_E, self.gene_E)
        
        # B. Compute "Biological Scores" for each individual
        # [Batch, N] @ [N, 16] -> [Batch, 16]
        # This aggregates SNP effects based on Prior knowledge
        h_prior_driven = X @ w_priors
        
        # C. Compute "Data Scores" (Blind to priors)
        # [Batch, N] -> [Batch, 16]
        h_data_driven = self.data_projector(X)
        
        # D. Fuse and Non-linear Process
        h_combined = torch.cat([h_prior_driven, h_data_driven], dim=1)
        out_deep = self.deep_mlp(h_combined)
        
        # --- Final Sum ---
        # Residual connection: Linear + alpha * NonLinear
        return out_wide + (self.deep_scale * out_deep)

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

# Standard Training Interface
def train(plink_prefix, pheno_file, train_ids, test_ids, trait, delta_path, gene_path, out_dir,
          lr=1e-3, batch_size=64, epochs=150, lambda_l1=0.001, device='auto', **kwargs):
    
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    if device == 'auto': dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    else: dev = device
    print(f"Training Bio-Master v9 (Wide & Deep) on {dev}")
    
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

    # Load Features
    delta = np.load(delta_path); gene = np.load(gene_path)
    if delta.shape[0] != gene.shape[0]:
        m = min(delta.shape[0], gene.shape[0])
        delta, gene = delta[:m], gene[:m]

    # Load Genotypes
    (bim, fam, bed) = read_plink(plink_prefix, verbose=False)
    G = bed.compute().T
    if G.shape[1] != delta.shape[0]:
        G = G[:, :min(G.shape[1], delta.shape[0])]
        delta, gene = delta[:G.shape[1]], gene[:G.shape[1]]

    # Data Splits
    ymap = _load_pheno_map(pheno_file, trait)
    iid2idx = dict(zip(fam['iid'].astype(str), range(len(fam))))
    tr_df = pd.read_csv(train_ids, sep='\t', names=['FID', 'IID'])
    te_df = pd.read_csv(test_ids, sep='\t', names=['FID', 'IID'])
    
    tr_idx = [iid2idx[str(x)] for x in tr_df['IID'] if str(x) in iid2idx]
    te_idx = [iid2idx[str(x)] for x in te_df['IID'] if str(x) in iid2idx]
    
    X_tr, y_tr = G[tr_idx], np.array([ymap.get(str(x), np.nan) for x in tr_df['IID'] if str(x) in iid2idx])
    X_te, y_te = G[te_idx], np.array([ymap.get(str(x), np.nan) for x in te_df['IID'] if str(x) in iid2idx])
    
    mtr, mte = ~np.isnan(y_tr), ~np.isnan(y_te)
    X_tr, y_tr = X_tr[mtr], y_tr[mtr]
    X_te, y_te = X_te[mte], y_te[mte]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    # Init Model
    model = WideAndDeepModel(delta, gene).to(dev)
    
    # Optimizer: Weight decay helps prevent overfitting in the Deep part
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    crit = nn.MSELoss()

    loader = DataLoader(_DS(X_tr, y_tr), batch_size=batch_size, shuffle=True, 
                        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    
    best = float('inf'); patience = 0
    
    for ep in range(epochs):
        model.train(); tot = 0; cnt = 0
        for xb, yb in loader:
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            opt.zero_grad()
            pred = model(xb)
            mse = crit(pred, yb)
            
            # L1 Regularization specifically on the Wide weights (Sparsity)
            # This mimics Lasso / BayesC
            l1_wide = torch.norm(model.wide.weight, 1)
            loss = mse + (lambda_l1 * l1_wide)
            
            loss.backward(); opt.step()
            tot += mse.item(); cnt += 1
            
        avg = tot / max(1, cnt); sched.step(avg)
        if avg < best:
            best = avg; patience = 0
            torch.save(model.state_dict(), out / 'best_model.pt')
        else:
            patience += 1
        if patience >= 20: break # Increased patience slightly

    # Eval
    model.load_state_dict(torch.load(out / 'best_model.pt', map_location=dev))
    model.eval()
    test_loader = DataLoader(_DS(X_te, y_te), batch_size=256, shuffle=False, num_workers=num_workers)
    preds = []
    
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(dev)
            p = model(xb)
            preds.append(p.cpu().numpy().flatten())
            
    p_te = np.concatenate(preds)
    pcc = float(pearsonr(y_te, p_te)[0]) if len(y_te) > 1 else 0.0
    mse = float(mean_squared_error(y_te, p_te))
    
    # Sparsity check on Wide weights
    wide_w = model.wide.weight.data.cpu().numpy().flatten()
    nz = int(np.sum(np.abs(wide_w) > 1e-3))
    
    pd.DataFrame({'IID': [str(x) for x in te_df['IID'][mte]], 'True': y_te, 'Pred': p_te}).to_csv(out / 'df_gsf_v5_pred.csv', index=False)
    with open(out / 'DF_GSF_v5_stats.json', 'w') as f:
        json.dump({'pcc_test': pcc, 'mse': mse, 'sparsity': nz, 'model': 'bio_master_v9'}, f, indent=2)
    
    return {'pcc_test': pcc}