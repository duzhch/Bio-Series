src/model.py
#!/usr/bin/env python3
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from torch.utils.data import Dataset, DataLoader
from pandas_plink import read_plink


class DualTowerNeuralBayes(nn.Module):
    def __init__(self, delta_E: np.ndarray, gene_E: np.ndarray):
        super().__init__()
        self.register_buffer('delta_E', torch.tensor(delta_E, dtype=torch.float32))
        self.register_buffer('gene_E', torch.tensor(gene_E, dtype=torch.float32))
        delta_dim = self.delta_E.shape[1]
        gene_dim = self.gene_E.shape[1]
        hidden = 64
        self.micro = nn.Sequential(
            nn.Linear(delta_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.2)
        )
        self.macro = nn.Sequential(
            nn.Linear(gene_dim, 32), nn.ReLU(), nn.Linear(32, hidden), nn.Sigmoid()
        )
        self.head = nn.Linear(hidden, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, X: torch.Tensor):
        h = self.micro(self.delta_E) * self.macro(self.gene_E)
        w = self.head(h)
        y = X @ w + self.bias
        return y, w


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
          lr: float = 5e-4, batch_size: int = 64, epochs: int = 150, lambda_l1: float = 0.005,
          device: str = 'auto'):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # Use CPU for training to avoid GPU conflicts
    dev = 'cpu'
    print(f"Training on device: {dev}")
    # Load features
    delta = np.load(delta_path)
    gene = np.load(gene_path)
    if delta.shape[0] != gene.shape[0]:
        m = min(delta.shape[0], gene.shape[0])
        delta, gene = delta[:m], gene[:m]

    # Load genotypes
    (bim, fam, bed) = read_plink(plink_prefix, verbose=False)
    G = bed.compute().T
    if G.shape[1] != delta.shape[0]:
        m = min(G.shape[1], delta.shape[0])
        G = G[:, :m]
        delta, gene = delta[:m], gene[:m]

    # Align phenotypes
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

    # Scale
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    # Device (already set above)
    model = DualTowerNeuralBayes(delta, gene).to(dev)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    crit = nn.MSELoss()

    loader = DataLoader(_DS(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    best = float('inf'); patience = 0
    for ep in range(epochs):
        model.train(); tot = 0; cnt = 0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            pred, w = model(xb)
            mse = crit(pred, yb)
            l1 = torch.norm(w, 1)
            loss = mse + lambda_l1 * l1
            loss.backward(); opt.step()
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

    # Eval
    model.load_state_dict(torch.load(out / 'best_model_v5.pt', map_location=dev))
    model.eval()
    with torch.no_grad():
        p_te, w_te = model(torch.tensor(X_te, dtype=torch.float32).to(dev))
        p_te = p_te.cpu().numpy().flatten()
        w = w_te.cpu().numpy().flatten()
    pcc = float(pearsonr(y_te, p_te)[0]) if len(y_te) > 1 else 0.0
    mse = float(mean_squared_error(y_te, p_te))
    nz = int(np.sum(np.abs(w) > 1e-3))
    pd.DataFrame({'IID': [str(x) for x in te_df['IID'][mte]], 'True': y_te, 'Pred': p_te}) \
        .to_csv(out / 'df_gsf_v5_pred.csv', index=False)
    with open(out / 'DF_GSF_v5_stats.json', 'w') as f:
        json.dump({'pcc_test': pcc, 'mse': mse, 'sparsity': nz}, f, indent=2)
    return {'pcc_test': pcc, 'mse': mse, 'sparsity': nz}
