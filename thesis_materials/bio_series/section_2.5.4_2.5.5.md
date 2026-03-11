### 2.5.4 基因组 Transformer 与局部-全局交互机制

为了有效解决全基因组高维 SNP 特征（$p \gg 50,000$）带来的维度灾难以及长距离连锁不平衡（Linkage Disequilibrium, LD）问题，Bio-Series 架构设计了一种 **Local-to-Global** 的两级特征处理机制。该机制通过分层抽象，将微观的 SNP 变异逐步聚合为宏观的遗传互作网络。

#### (1) 局部区块卷积 (Local Block Convolution)

依据数量遗传学中的连锁不平衡原理，物理位置相邻的 SNP 往往作为一个遗传单元（单倍型块，Haplotype Block）共同遗传。因此，我们首先将全基因组基因型向量 $\mathbf{X} \in \mathbb{R}^{M}$ 切分为 $B$ 个长度为 $L$ 的连续区块（在本研究中 $L=100$）：
$$ \mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_B], \quad \text{where } \mathbf{x}_i \in \mathbb{R}^{L} $$

在进行卷积操作之前，为了融入生物学先验知识，我们利用上一节生成的动态先验权重向量 $\mathbf{P}$ 对基因型进行特征增强。这一步通过哈达玛积（Hadamard Product）实现：
$$ \mathbf{x}'_i = \mathbf{x}_i \odot \mathbf{p}_i $$
其中 $\mathbf{p}_i$ 是对应区块的先验权重。该操作相当于在输入层即对功能性位点（如位于外显子区域或具有高 Delta 分数的 SNP）进行信号放大。

随后，利用一维卷积层（Conv1d）配合最大池化（Max Pooling）提取每个区块的局部特征：
$$ \mathbf{z}_i = \text{MaxPool}(\sigma(\mathbf{W}_{conv} * \mathbf{x}'_i + \mathbf{b}_{conv})) $$
经此步骤，原始的高维 SNP 序列被压缩为一系列低维的 **Block Tokens** 序列 $\mathbf{Z} = [\mathbf{z}_1, \mathbf{z}_2, \dots, \mathbf{z}_B]$，每个 Token 代表了一个局部染色体区域的综合遗传效应。

#### (2) 全局自注意力 (Global Self-Attention)

为了捕捉跨染色体的长距离上位性效应（Epistasis），压缩后的特征块序列 $\mathbf{Z}$ 被输入到 Transformer 编码层。首先，引入位置编码（Positional Encoding, PE）以保留基因组的线性空间结构信息：
$$ \mathbf{H}^{(0)} = \mathbf{Z} + \text{PE} $$

接着，利用多头自注意力机制（Multi-Head Self-Attention, MHSA）动态计算全基因组范围内任意两个区域之间的关联权重：
$$ \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V} $$
其中查询矩阵 $\mathbf{Q}$、键矩阵 $\mathbf{K}$ 和值矩阵 $\mathbf{V}$ 均由输入特征线性投影得到。注意力权重矩阵 $\mathbf{A} = \text{softmax}(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}})$ 中的元素 $a_{ij}$ 直观地反映了第 $i$ 个基因组区域与第 $j$ 个区域之间的互作强度。这使得模型能够自动发现并聚焦于对表型有显著贡献的基因-基因相互作用网络，而无需人工定义交互项。

最后，采用全局平均池化（Global Average Pooling, GAP）聚合所有 Token 的信息，生成最终的基因组深度特征向量 $\mathbf{h}_{deep}$：
$$ \mathbf{h}_{deep} = \frac{1}{B} \sum_{i=1}^{B} \mathbf{h}_i^{(Last)} $$
GAP 操作不仅显著降低了模型参数量（避免了全连接层带来的参数爆炸），还增强了模型对输入长度变化的鲁棒性，有效防止了在小样本数据上的过拟合。

---

### 2.5.5 模型优化策略与损失函数设计

针对基因组预测任务中样本量远小于特征数（$N \ll p$）且信噪比（Heritability）通常较低的挑战，本研究从损失函数设计与训练机制两个维度进行了针对性优化。

#### 1. 复合损失函数构建

为了兼顾数值预测的准确性、个体排序的精确度以及特征的生物学解耦性，本研究构建了如下的三元复合损失函数：

$$ \mathcal{L}_{total} = \alpha \cdot \mathcal{L}_{PCC} + \beta \cdot \mathcal{L}_{Rank} + \gamma \cdot \mathcal{L}_{Orth} $$

*   **皮尔逊相关损失 ($\mathcal{L}_{PCC}$)**：
    旨在直接优化预测育种值（GEBV, $\hat{y}$）与真实表型值（$y$）之间的线性相关性，这是育种实践中最关注的指标。
    $$ \mathcal{L}_{PCC} = 1 - \frac{\text{Cov}(y, \hat{y})}{\sigma_y \sigma_{\hat{y}} + \epsilon} $$
    其中 $\epsilon = 10^{-6}$ 为平滑项，防止分母为零导致的数值不稳定。

*   **ListNet 排序损失 ($\mathcal{L}_{Rank}$)**：
    通过将排序问题转化为概率分布的差异问题，提升模型对前 $K$ 个优良个体的筛选精度。我们使用 Softmax 函数将真实值和预测值映射为概率分布，并计算交叉熵：
    $$ \mathcal{L}_{Rank} = - \sum_{i=1}^{N} P(y_i) \log(P(\hat{y}_i)) $$
    $$ \text{where } P(y_i) = \frac{\exp(y_i)}{\sum_j \exp(y_j)} $$
    该损失函数迫使模型关注样本间的相对顺序，而非绝对数值误差，特别适用于选育顶端个体的场景。

*   **正交解耦损失 ($\mathcal{L}_{Orth}$)**：
    通过最小化深度交互流提取的特征 $\mathbf{h}_{deep}$ 与种群校正流特征 $\mathbf{h}_{ctx}$ 之间的余弦相似度，强制模型剔除种群结构干扰，深度挖掘纯净的遗传变异。
    $$ \mathcal{L}_{Orth} = \frac{1}{N} \sum_{i=1}^{N} \left| \frac{\mathbf{h}_{deep, i} \cdot \mathbf{h}_{ctx, i}}{\|\mathbf{h}_{deep, i}\| \|\mathbf{h}_{ctx, i}\|} \right| $$

#### 2. 训练机制与正则化策略

在训练过程中，本研究引入了多重策略以保证模型的泛化性与稳定性：

*   **动态学习率调度**：采用 **AdamW** 优化器配合 **ReduceLROnPlateau** 策略。初始学习率设为 $5 \times 10^{-4}$，当验证集 Loss 在 5 个 Epoch 内未下降时，学习率自动衰减为原来的 0.5 倍，实现训练后期的精细化调优。
*   **先验稀疏性约束**：考虑到基因组中大部分区域为非功能区域，我们在总损失中引入针对先验权重向量 $\mathbf{P}$ 的 $L_1$ 正则化项：
    $$ \mathcal{L}_{final} = \mathcal{L}_{total} + \lambda \|\mathbf{P}\|_1 $$
    这促使模型生成的先验权重具有稀疏性（Sparsity），从而有效抑制非功能区域的噪声信号，提高生物学可解释性。
*   **稳定性控制**：利用 **梯度裁剪 (Gradient Clipping)**（阈值设为 1.0）防止 Transformer 深层网络训练过程中的梯度爆炸。同时设置基于验证集 PCC 的 **早停机制 (Early Stopping)**（Patience = 15），确保模型在有限样本下具备良好的鲁棒性，避免过拟合。
