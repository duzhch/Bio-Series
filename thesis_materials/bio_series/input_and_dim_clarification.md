# 代码细节确认：输入规模与注意力维度

根据您的提问，我核查了 `src/models/bio_master_v13.py` 和 `DF_GSF_v5.py` 的代码实现。以下是确切的代码事实和论文写作建议。

## 1. 关于 Wide Tower 的输入规模 (M vs N)

**问题**：Wide Tower 的输入是全基因组位点，还是和 Deep Tower 一样仅为 Top-3000 位点？

**代码事实**：
在 `BioMasterV12` 类的 `forward` 函数中：
```python
# src/models/bio_master_v13.py Line 170-174
def forward(self, X_snps, X_pcs):
    B, N = X_snps.shape
    
    # Path A: Wide
    out_wide = self.wide_linear(X_snps)
```
*   `X_snps` 是同时输入给 `wide_linear` (Wide Tower) 和后续 `prior_gen` (Deep Tower) 的同一个张量。
*   这个 `X_snps` 是在 `DF_GSF_v5.py` 的 Step 4 中通过 `plink_extract` 提取的，其 SNP 列表来自 `gwas_results['plink_snps']`（即 Clumping 后的结果）。

**结论**：
Wide Tower 和 Deep Tower **使用的是完全相同的输入**，即经过 GWAS 筛选后的 Top-N (如 3000) 个位点。Wide Tower **并没有**使用全基因组数据。

**论文修改建议**：
在 2.5.1 中**不应**声称 Wide Tower 使用了全基因组数据，否则会被认为是虚假陈述。
建议这样表述以保持严谨性：
> “在输入层，本模型采用共享特征策略。经过 GWAS 统计筛选后的 $N$ 个关键位点（Top-N SNPs）被同时输入到 Wide Tower 和 Deep Tower。Wide Tower 利用这些显著位点构建稳健的线性基准，而 Deep Tower 则进一步挖掘这些位点间的非线性上位性互作。Context Tower 则利用全基因组主成分（Global PCA）提供宏观的群体结构校正。”

## 2. 关于注意力维度 $d_k$ (公式 2-16)

**问题**：注意力公式 $Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$ 中的 $d_k$ 大小是多少？

**代码事实**：
在 `BioMasterV12` 的初始化中：
```python
# src/models/bio_master_v13.py Line 136
self.d_model = 64 

# Line 142
self.genomic_transformer = GenomicTransformerV2(..., d_model=self.d_model, nhead=4, ...)
```
Transformer 的 `d_model` 被设为 64，且有 4 个头 (`nhead=4`)。
在 PyTorch 的 `MultiheadAttention` 实现中，$d_k$ (每个头的维度) 通常等于 `d_model / nhead`。
$$ d_k = 64 / 4 = 16 $$

**结论**：
*   模型的隐藏层维度 $d_{model} = 64$。
*   每个注意力头的维度 $d_k = 16$。

**论文修改建议**：
建议在 2.5.3 节公式附近标注：
> “其中 $d_k$ 为注意力头的维度。在本研究中，设定模型隐藏层维度 $d_{model}=64$，采用 4 头注意力机制（Head=4），因此缩放因子 $d_k = d_{model} / \text{Head} = 16$。”
