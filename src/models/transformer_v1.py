#!/usr/bin/env python3
# 废弃效果太差，运行时间非常长，cat transformer_v1/rep_01/DF_GSF_v5_stats.json 
# {
#   "pcc_test": 0.18220408262757587,
#   "mse": 10.208342461409966,
#   "sparsity": -1,
#   "model": "Transformer_v1"
# }(multigs) [zyqgroup02@r03c03n07 LargeWhite_Pop1_BF]$ cat transformer_v1/rep_02/DF_GSF_v5_stats.json 
# {
#   "pcc_test": -0.10135252300765614,
#   "mse": 8.974993685111132,
#   "sparsity": -1,
#   "model": "Transformer_v1"
# }(multigs) [zyqgroup02@r03c03n07 LargeWhite_Pop1_BF]$ cat transformer_v1/rep_03/DF_GSF_v5_stats.json 
# {
#   "pcc_test": 0.19232210243604558,
#   "mse": 9.898519279533955,
#   "sparsity": -1,
#   "model": "Transformer_v1"
# }(multigs) [zyqgroup02@r03c03n07 LargeWhite_Pop1_BF]$ cat transformer_v1/rep_04/DF_GSF_v5_stats.json 
# {
#   "pcc_test": 0.15061130816253931,
#   "mse": 9.839240565625548,
#   "sparsity": -1,
#   "model": "Transformer_v1"

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

# --- 核心架构：Bio-Informed Feature Tokenizer Transformer ---

class BioFeatureTokenizer(nn.Module):
    """
    将 SNP 基因型 (0/1/2) 转换为富含生物学信息的 Token。
    逻辑：
    Token[i] = Base_Embedding[i] + Genotype_Embedding[Val]
    其中 Base_Embedding 由 PigBERT (Sequence) 和 Gene2Vec (Function) 融合而成。
    """
    def __init__(self, delta_E: torch.Tensor, gene_E: torch.Tensor, d_model: int):
        super().__init__()
        self.n_snps = delta_E.shape[0]
        
        # 1. 融合先验信息 (Bio-Priors)
        # 将 768维的 PigBERT 和 200维的 Gene2Vec 投影到 d_model 维度
        self.seq_proj = nn.Linear(delta_E.shape[1], d_model)
        self.func_proj = nn.Linear(gene_E.shape[1], d_model)
        
        # 注册为 buffer，不参与梯度更新 (或者可以微调，视数据量而定，这里建议冻结以防过拟合)
        self.register_buffer('delta_E', delta_E)
        self.register_buffer('gene_E', gene_E)

        # 2. 基因型状态嵌入 (0, 1, 2, Missing)
        # 0: Homo Ref, 1: Het, 2: Homo Alt, 3: Missing (padding)
        self.genotype_emb = nn.Embedding(4, d_model)
        
        # 3. LayerNorm 用于稳定融合
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x_cat: torch.Tensor):
        """
        x_cat: (Batch, N_SNPs) 整数矩阵，值为 0, 1, 2
        """
        # Step A: 构建 SNP 的生物学身份基底 (Static per SNP)
        # (N_SNPs, D_model)
        bio_identity = self.seq_proj(self.delta_E) + self.func_proj(self.gene_E)
        
        # Step B: 获取基因型状态嵌入 (Dynamic per Sample)
        # (Batch, N_SNPs, D_model)
        geno_state = self.genotype_emb(x_cat.long())
        
        # Step C: 融合
        # 广播机制: (1, N, D) + (B, N, D) -> (B, N, D)
        # 这里的加法假设：生物身份是底色，基因型是修饰
        tokens = bio_identity.unsqueeze(0) + geno_state
        
        return self.ln(tokens)

class EpistaticTransformer(nn.Module):
    def __init__(self, delta_E: np.ndarray, gene_E: np.ndarray, 
                 d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.2):
        super().__init__()
        
        # 转换为 Tensor
        delta_T = torch.tensor(delta_E, dtype=torch.float32)
        gene_T = torch.tensor(gene_E, dtype=torch.float32)
        
        # 1. Tokenizer
        self.tokenizer = BioFeatureTokenizer(delta_T, gene_T, d_model)
        
        # 2. CLS Token (用于聚合全局表型信息)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 3. Transformer Encoder
        # batch_first=True 是必须的
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True,
            norm_first=True # Pre-LN 收敛更稳
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Prediction Head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: (Batch, N_SNPs)
        
        # Tokenize -> (Batch, N_SNPs, D_model)
        x_emb = self.tokenizer(x)
        
        # Prepend CLS token
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_emb = torch.cat((cls_tokens, x_emb), dim=1)
        
        # Transformer Pass
        # x_out -> (Batch, N_SNPs + 1, D_model)
        x_out = self.transformer(x_emb)
        
        # Take CLS token output only
        cls_out = x_out[:, 0, :]
        
        # Predict
        return self.head(cls_out), cls_out # Return feature for hook/vis

# --- 基础设施代码 (与 DF_GSF_v5 兼容) ---

class _DS(Dataset):
    def __init__(self, X, y):
        # 注意：Transformer 需要 int 类型的输入来做 Embedding (0,1,2)
        # 但如果输入已经是归一化的 float，我们需要反推或者在 data.py 里不做 scaler
        # 鉴于 GBLUP 需要数值矩阵，而 Transformer 需要离散矩阵
        # 这里为了兼容性，假设传入的是标准化后的 float，我们在 Dataset 里做一个简单的离散化 trick
        # 或者更理想的是，train 函数里传入原始 0/1/2 矩阵
        
        # 这里假设 train 函数已经做了适配，传入的是 0/1/2 的原始 genotype
        self.X = torch.tensor(X, dtype=torch.float32) # 保持 float 接口兼容，内部转 long
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
          lr: float = 1e-4, batch_size: int = 32, epochs: int = 100, lambda_l1: float = 0.0,
          device: str = 'auto'):
    
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # 智能设备选择：Transformer 最好用 GPU
    if device == 'auto':
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        dev = device
    print(f"[Transformer_v1] Training on device: {dev}")

    # Load Features (Priors)
    delta = np.load(delta_path)
    gene = np.load(gene_path)
    # 对齐维度
    if delta.shape[0] != gene.shape[0]:
        m = min(delta.shape[0], gene.shape[0])
        delta, gene = delta[:m], gene[:m]

    # Load Genotypes
    # 注意：Transformer 最好直接吃 0/1/2 整数，不要 StandardScaler
    (bim, fam, bed) = read_plink(plink_prefix, verbose=False)
    G = bed.compute().T # (Samples, SNPs)
    
    # 填充 NaN (PLINK 中 NaN 通常是 -9 或 3，这里填 0 或 众数，简单起见填 0)
    G = np.nan_to_num(G, nan=0.0)
    
    # 确保 SNP 数量一致
    if G.shape[1] != delta.shape[0]:
        m = min(G.shape[1], delta.shape[0])
        G = G[:, :m]
        delta, gene = delta[:m], gene[:m]

    # Align Phenotypes
    ymap = _load_pheno_map(pheno_file, trait)
    iid2idx = dict(zip(fam['iid'].astype(str), range(len(fam))))
    tr_df = pd.read_csv(train_ids, sep='\t', names=['FID', 'IID'])
    te_df = pd.read_csv(test_ids, sep='\t', names=['FID', 'IID'])
    
    tr_idx = [iid2idx[str(x)] for x in tr_df['IID'] if str(x) in iid2idx]
    te_idx = [iid2idx[str(x)] for x in te_df['IID'] if str(x) in iid2idx]
    
    X_tr, X_te = G[tr_idx], G[te_idx]
    y_tr = np.array([ymap.get(str(x), np.nan) for x in tr_df['IID'] if str(x) in iid2idx])
    y_te = np.array([ymap.get(str(x), np.nan) for x in te_df['IID'] if str(x) in iid2idx])
    
    # Filter missing phenotypes
    mtr, mte = ~np.isnan(y_tr), ~np.isnan(y_te)
    X_tr, y_tr = X_tr[mtr], y_tr[mtr]
    X_te, y_te = X_te[mte], y_te[mte]

    # IMPORTANT: Do NOT scale X for this Transformer model. 
    # The Tokenizer expects 0, 1, 2 values.
    # We only assume inputs are reasonably close to 0, 1, 2.
    # If using imputed data (dosage 0-2 float), we round it for embedding lookup
    X_tr = np.round(X_tr)
    X_te = np.round(X_te)

    # Initialize Model
    # d_model=64 对于 3000 个 SNP 来说是比较轻量级的，适合 2000 样本
    model = EpistaticTransformer(delta, gene, d_model=64, nhead=4, num_layers=2, dropout=0.3).to(dev)
    
    # Optimizer (Transformer 需要 AdamW + Weight Decay 防止过拟合)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)
    crit = nn.MSELoss()

    loader = DataLoader(_DS(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    
    best_mse = float('inf')
    patience = 0
    
    for ep in range(epochs):
        model.train()
        tot_loss = 0
        cnt = 0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            
            opt.zero_grad()
            pred, _ = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            
            # 梯度裁剪，防止 Transformer 梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            opt.step()
            tot_loss += loss.item()
            cnt += 1
            
        avg_loss = tot_loss / max(1, cnt)
        sched.step() # Cosine Annealing per epoch
        
        # Validation logic
        model.eval()
        with torch.no_grad():
            # Process test set in chunks to avoid OOM
            preds_te = []
            # Simple manual batching for test
            test_bs = 100
            for i in range(0, len(X_te), test_bs):
                batch_x = torch.tensor(X_te[i:i+test_bs], dtype=torch.float32).to(dev)
                p, _ = model(batch_x)
                preds_te.append(p.cpu().numpy())
            
            if len(preds_te) > 0:
                p_te = np.concatenate(preds_te).flatten()
                val_mse = float(mean_squared_error(y_te, p_te))
            else:
                val_mse = float('inf')

        print(f"Epoch {ep+1}/{epochs} | Train Loss: {avg_loss:.4f} | Val MSE: {val_mse:.4f}")

        if val_mse < best_mse:
            best_mse = val_mse
            patience = 0
            torch.save(model.state_dict(), out / 'best_model_v5.pt')
        else:
            patience += 1
            
        if patience >= 10:
            print("Early stopping triggered.")
            break

    # Final Eval
    model.load_state_dict(torch.load(out / 'best_model_v5.pt', map_location=dev))
    model.eval()
    with torch.no_grad():
        preds_te = []
        attentions = [] # Optional: capture attention if needed later
        for i in range(0, len(X_te), 100):
            batch_x = torch.tensor(X_te[i:i+100], dtype=torch.float32).to(dev)
            p, _ = model(batch_x)
            preds_te.append(p.cpu().numpy())
        p_te = np.concatenate(preds_te).flatten()

    pcc = float(pearsonr(y_te, p_te)[0]) if len(y_te) > 1 else 0.0
    mse = float(mean_squared_error(y_te, p_te))
    
    # Transformer 没有像线性模型那样明确的 "sparsity" (zero weights)
    # 但我们可以计算 Attention map 的稀疏度作为代理，或者设为 -1 表示不适用
    nz = -1 

    # Save outputs
    pd.DataFrame({'IID': [str(x) for x in te_df['IID'][mte]], 'True': y_te, 'Pred': p_te}) \
        .to_csv(out / 'df_gsf_v5_pred.csv', index=False)
        
    with open(out / 'DF_GSF_v5_stats.json', 'w') as f:
        json.dump({'pcc_test': pcc, 'mse': mse, 'sparsity': nz, 'model': 'Transformer_v1'}, f, indent=2)
        
    return {'pcc_test': pcc, 'mse': mse, 'sparsity': nz}