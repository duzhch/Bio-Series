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
# Bio-Master v8 Optimized Architecture
# Optimized for CPU Efficiency: ReLU instead of GELU, In-place ops
# ==========================================

class PriorGenerator(nn.Module):
    """
    Internal module to fuse Delta and Gene embeddings.
    Optimization: Uses ReLU(inplace=True) for speed and memory efficiency.
    """
    def __init__(self, delta_dim, gene_dim):
        super().__init__()
        self.delta_compress = nn.Linear(delta_dim, 16)
        self.gene_compress = nn.Linear(gene_dim, 16)
        self.fuse = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid() 
        )
    
    def forward(self, d_emb, g_emb):
        # In-place ReLU saves memory and avoids creating new tensors
        d = F.relu(self.delta_compress(d_emb), inplace=True)
        g = F.relu(self.gene_compress(g_emb), inplace=True)
        return self.fuse(torch.cat([d, g], dim=1))

class LDBlockEncoder(nn.Module):
    """
    Hierarchical Encoder: Conv1D + DeepSet.
    Optimization: Replaced GELU with ReLU for faster CPU execution.
    """
    def __init__(self, block_size, in_channels=2, embed_dim=32):
        super().__init__()
        # Path A: Local Linkage (Convolutional)
        self.conv1 = nn.Conv1d(in_channels, embed_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(embed_dim) # BatchNorm is often faster than LayerNorm on CPU
        
        # Path B: Rare Variants
        self.rare_project = nn.Linear(block_size * in_channels, embed_dim)
        
        # Optimization: ReLU is significantly faster on CPU than GELU
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        B, C, L = x.shape
        
        # Path A
        h_conv = self.act(self.bn1(self.conv1(x))) 
        h_conv_pool = torch.max(h_conv, dim=2)[0]
        
        # Path B
        x_flat = x.view(B, -1)
        h_rare = self.rare_project(x_flat)
        
        # Fusion
        return self.dropout(h_conv_pool + h_rare)

class PriorGuidedSE(nn.Module):
    """
    Gated Attention.
    Optimization: ReLU inplace.
    """
    def __init__(self, input_dim, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(input_dim // reduction, input_dim, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.fc2(self.relu(self.fc1(x)))
        scale = self.sigmoid(y)
        return x * scale

# ==========================================
# Main Model Wrapper
# ==========================================

class DualTowerNeuralBayes(nn.Module):
    def __init__(self, delta_E: np.ndarray, gene_E: np.ndarray, block_size=100):
        super().__init__()
        
        self.register_buffer('delta_E', torch.tensor(delta_E, dtype=torch.float32))
        self.register_buffer('gene_E', torch.tensor(gene_E, dtype=torch.float32))
        
        delta_dim = self.delta_E.shape[1]
        gene_dim = self.gene_E.shape[1]
        n_snps = self.delta_E.shape[0]

        # Padding logic
        self.block_size = block_size
        self.pad_len = (block_size - (n_snps % block_size)) % block_size
        self.n_blocks = (n_snps + self.pad_len) // block_size
        
        self.prior_generator = PriorGenerator(delta_dim, gene_dim)
        
        self.embed_dim = 64
        self.block_encoder = LDBlockEncoder(block_size, in_channels=2, embed_dim=self.embed_dim)
        self.global_gate = PriorGuidedSE(self.embed_dim)
        
        # Optimization: Lightweight Head
        self.regressor = nn.Sequential(
            nn.Linear(self.embed_dim * self.n_blocks, 128),
            nn.LayerNorm(128), # Keep LN here for stability
            nn.ReLU(inplace=True), # Faster activation
            nn.Linear(128, 1)
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, X: torch.Tensor):
        B, N = X.shape
        
        # 1. Generate Priors
        prior_scores = self.prior_generator(self.delta_E, self.gene_E)
        
        # 2. Reshape (Optimized view operations)
        if self.pad_len > 0:
            X_padded = F.pad(X, (0, self.pad_len))
            priors_padded = F.pad(prior_scores.t(), (0, self.pad_len)).t()
        else:
            X_padded = X
            priors_padded = prior_scores
            
        g_blocks = X_padded.view(B, self.n_blocks, self.block_size)
        p_blocks = priors_padded.view(1, self.n_blocks, self.block_size).expand(B, -1, -1)
        
        # Stack & Encode
        x_folded = torch.stack([g_blocks, p_blocks], dim=2).view(B * self.n_blocks, 2, self.block_size)
        block_embs = self.block_encoder(x_folded)
        
        # Attention & Predict
        block_seq = block_embs.view(B, self.n_blocks, self.embed_dim)
        flat_features = self.global_gate(block_seq).view(B, -1)
        
        y = self.regressor(flat_features) + self.bias
        return y, prior_scores[:N]


class _DS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]


def _load_pheno_map(pheno_file: str, trait: str):
    df = pd.read_csv(pheno_file, sep='\t' if pheno_file.endswith('.tsv') else ',')
    id_col = None
    for c in df.columns:
        if c.lower() in ['id', 'iid', 'sample_id', 'sample']:
            id_col = c
            break
    if id_col is None:
        id_col = df.columns[0]
    df[id_col] = df[id_col].astype(str)
    return dict(zip(df[id_col], df[trait]))


def train(plink_prefix: str, pheno_file: str, train_ids: str, test_ids: str,
          trait: str, delta_path: str, gene_path: str, out_dir: str,
          lr: float = 3e-4, batch_size: int = 64, epochs: int = 150, lambda_l1: float = 0.005,
          device: str = 'auto'):
    
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # 1. Device Selection
    if device == 'auto':
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        dev = device
    
    print(f"Training Bio-Master v8 on device: {dev}")
    
    # 2. CPU Performance Optimization Config
    # Detect CPU cores to optimize DataLoader
    num_workers = 0
    persistent_workers = False
    pin_memory = False
    
    if dev == 'cpu':
        # On CPU, multiple workers are crucial to feed data faster
        try:
            # Use 50% of available cores, max 8
            n_cores = multiprocessing.cpu_count()
            num_workers = min(8, max(2, n_cores // 2))
            persistent_workers = True # Keep workers alive to avoid setup overhead
            print(f"CPU Optimization: Using {num_workers} data loader workers")
        except:
            num_workers = 2
    else:
        # On GPU, standard settings
        num_workers = 4
        pin_memory = True

    # 3. Load Features & Data (Same logic as before)
    delta = np.load(delta_path)
    gene = np.load(gene_path)
    if delta.shape[0] != gene.shape[0]:
        m = min(delta.shape[0], gene.shape[0])
        delta, gene = delta[:m], gene[:m]

    (bim, fam, bed) = read_plink(plink_prefix, verbose=False)
    G = bed.compute().T
    if G.shape[1] != delta.shape[0]:
        m = min(G.shape[1], delta.shape[0])
        G = G[:, :m]
        delta, gene = delta[:m], gene[:m]

    ymap = _load_pheno_map(pheno_file, trait)
    iid2idx = dict(zip(fam['iid'].astype(str), range(len(fam))))
    tr_df = pd.read_csv(train_ids, sep='\t', names=['FID', 'IID'])
    te_df = pd.read_csv(test_ids, sep='\t', names=['FID', 'IID'])
    
    tr_idx = [iid2idx[str(x)] for x in tr_df['IID'] if str(x) in iid2idx]
    te_idx = [iid2idx[str(x)] for x in te_df['IID'] if str(x) in iid2idx]
    
    X_tr, X_te = G[tr_idx], G[te_idx]
    y_tr = np.array([ymap.get(str(x), np.nan) for x in tr_df['IID'] if str(x) in iid2idx])
    y_te = np.array([ymap.get(str(x), np.nan) for x in te_df['IID'] if str(x) in iid2idx])
    
    mtr, mte = ~np.isnan(y_tr), ~np.isnan(y_te)
    X_tr, y_tr = X_tr[mtr], y_tr[mtr]
    X_te, y_te = X_te[mte], y_te[mte]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    # 4. Initialize Optimized Model
    model = DualTowerNeuralBayes(delta, gene, block_size=100).to(dev)
    
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    crit = nn.MSELoss()

    # 5. Optimized DataLoader
    loader = DataLoader(
        _DS(X_tr, y_tr), 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    
    best = float('inf'); patience = 0
    
    # 6. Training
    for ep in range(epochs):
        model.train()
        tot = 0; cnt = 0
        
        for xb, yb in loader:
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            opt.zero_grad()
            
            pred, priors = model(xb)
            
            mse = crit(pred, yb)
            l1 = torch.norm(priors, 1)
            loss = mse + lambda_l1 * l1
            
            loss.backward()
            opt.step()
            
            tot += mse.item(); cnt += 1
            
        avg = tot / max(1, cnt)
        sched.step(avg)
        
        if avg < best:
            best = avg; patience = 0
            torch.save(model.state_dict(), out / 'best_model_v5.pt')
        else:
            patience += 1
        
        if patience >= 15:
            break

    # 7. Eval (Optimized for Inference)
    model.load_state_dict(torch.load(out / 'best_model_v5.pt', map_location=dev))
    model.eval()
    
    test_loader = DataLoader(_DS(X_te, y_te), batch_size=256, shuffle=False, num_workers=num_workers)
    preds = []
    final_priors = None
    
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(dev)
            p, w = model(xb)
            preds.append(p.cpu().numpy().flatten())
            if final_priors is None:
                final_priors = w.cpu().numpy().flatten()
    
    p_te = np.concatenate(preds)
    pcc = float(pearsonr(y_te, p_te)[0]) if len(y_te) > 1 else 0.0
    mse = float(mean_squared_error(y_te, p_te))
    nz = int(np.sum(np.abs(final_priors) > 1e-3))
    
    pd.DataFrame({'IID': [str(x) for x in te_df['IID'][mte]], 'True': y_te, 'Pred': p_te}) \
        .to_csv(out / 'df_gsf_v5_pred.csv', index=False)
    
    with open(out / 'DF_GSF_v5_stats.json', 'w') as f:
        json.dump({'pcc_test': pcc, 'mse': mse, 'sparsity': nz}, f, indent=2)
    
    return {'pcc_test': pcc, 'mse': mse, 'sparsity': nz}