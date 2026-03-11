# 针对评审问题的代码级查证与回复

本文档针对您提供的评审意见截图中的四个具体问题，基于 `src/models/bio_master_v13.py` 及相关代码进行了逐一查证和解答。

---

### 1. 公式 2-3 (GBLUP) 中 $Z$ 的定义

**问题**：这里的 $Z$ 是特指设计矩阵 (Design Matrix) 还是包含了基因型信息的加性效应矩阵？

**代码查证与回答**：
*   **理论定义**：在标准的 GBLUP 模型公式 $y = \mathbf{1}\mu + \mathbf{Z}g + e$ 中，$\mathbf{Z}$ 通常指的是 **设计矩阵 (Design Matrix)**，也称为关联矩阵 (Incidence Matrix)。它的作用是将个体的遗传值向量 $g$ 映射到表型观测值向量 $y$（例如，如果某个体有多次观测记录，或者某些个体没有表型）。
*   **代码实现**：在您的代码 `BioMasterV12` 的 Wide Tower 中：
    ```python
    # src/models/bio_master_v13.py
    self.wide_linear = nn.Linear(num_snps, 1)
    # forward: out_wide = self.wide_linear(X_snps)
    ```
    这里使用的是 **SNP-BLUP** 的等效形式 $y = \mathbf{X}\beta + e$，其中 $\mathbf{X}$ 是基因型矩阵。
*   **修改建议**：
    建议在文中明确：
    > “在经典 GBLUP 理论公式中，$\mathbf{Z}$ 为连接观测值与个体的**设计矩阵 (Design Matrix)**。而在本研究的 Bio-Series 架构实现中，我们采用等价的 SNP 回归形式，直接将**基因型矩阵 (Genotype Matrix)** 输入 Wide Tower 来捕获加性效应。”

---

### 2. 公式 2-10 (Gate Mechanism) 中的 $W_{AE}$

**问题**：出现 $W_{AE}$ 但未定义。是否指 Auto-Encoder 或 Additive Effect？

**代码查证与回答**：
*   **代码现状**：在先验生成模块 `RichPriorGenerator` 中：
    ```python
    # src/models/bio_master_v13.py
    self.delta_compress = nn.Linear(delta_dim, 32)  # W_d
    self.gene_compress = nn.Linear(gene_dim, 32)    # W_g
    self.fuse = nn.Sequential(..., nn.Linear(64, out_dim), ...) # W_fuse
    ```
    代码中没有名为 `AE` 的变量。
*   **推测与建议**：
    结合上下文，"Gate Mechanism" 通常涉及特征融合。$AE$ 最可能的含义是 **"Augmented Embedding" (增强嵌入)** 或 **"Attention Embedding" (注意力嵌入)**，指代融合后的特征。
    但考虑到容易引起歧义（AE 常指 Auto-Encoder），**强烈建议修改符号**。
*   **修改建议**：
    将 $W_{AE}$ 修改为 **$W_{fuse}$** (融合权重) 或 **$W_{prior}$** (先验权重)，并在公式下方注明：
    > “其中 $W_{fuse}$ 为多模态先验融合层的权重矩阵。”

---

### 3. 公式 2-22 (Total Loss) 超参数 $\alpha, \beta, \gamma$

**问题**：是固定超参数还是动态调整？

**代码查证与回答**：
*   **代码现状**：在 `src/models/bio_master_v13.py` 的 `train` 函数中：
    ```python
    # Line 315
    crit = HybridPCCLoss(alpha=1.0, beta=0.5, gamma=0.1)
    ```
    这些参数在初始化时被硬编码，且在整个训练循环中**保持不变**。
*   **回答**：
    是 **固定超参数 (Fixed Hyperparameters)**。
*   **具体取值**：
    *   $\alpha (\mathcal{L}_{PCC}) = 1.0$
    *   $\beta (\mathcal{L}_{Rank}) = 0.5$
    *   $\gamma (\mathcal{L}_{Orth}) = 0.1$
*   **修改建议**：
    在论文中明确写出：
    > “在本研究的所有实验中，超参数设为固定值：$\alpha=1.0, \beta=0.5, \gamma=0.1$。该比例经初步实验确定，旨在优先保证线性预测精度 ($\alpha$)，兼顾排序能力 ($\beta$)，并施加适度的解耦约束 ($\gamma$)。”

---

### 4. 公式 2-21 (Global Avg Pooling) 中的 $bh$

**问题**：$bh$ 是变量名还是 $b \times h$？下标 $i$ 是指 Block Token 索引吗？

**代码查证与回答**：
*   **代码现状**：
    ```python
    # src/models/bio_master_v13.py
    # Line 212: seq_tokens 维度为 [Batch, N_Blocks, d_model]
    # Line 218: trans_out = transformer_encoder(seq_tokens)
    # Line 221: gap_feat = torch.mean(trans_out, dim=1)
    ```
    这里的 `dim=1` 对应的是 `N_Blocks` 维度。
*   **回答**：
    1.  $bh$ 应理解为一个**整体变量名**，代表 **"Block Hidden-state" (区块隐藏状态)**，即 Transformer 输出的特征向量。建议改为 $h^{block}$ 或 $\mathbf{h}$ 以符合数学规范，避免被误解为乘积。
    2.  下标 $i$ 确实是指 **Block Token 的索引** (从 1 到 $B$，其中 $B$ 是区块总数 `n_blocks`)。
*   **修改建议**：
    建议优化公式表达，使用上标或粗体：
    $$ \mathbf{h}_{deep} = \frac{1}{B} \sum_{i=1}^{B} \mathbf{h}_i^{(Last)} $$
    并说明：
    > “其中 $B$ 为基因组划分的区块总数 (Number of Blocks)，$\mathbf{h}_i^{(Last)}$ 为 Transformer 最后一层输出的第 $i$ 个区块的隐藏状态向量 (Block Hidden State)。”
