### 2.4.3 Bio 系列增强架构详细设计

本节将深入阐述 Bio-Series 架构中三个核心数据流（Streams）的数学构建与运作机理。该架构采用“分而治之”的策略，将基因组预测任务解耦为加性效应、种群结构效应与非线性互作效应三个正交分量。

#### (1) 加性效应流 (The Additive Stream)

在数量遗传学中，加性效应（Additive Effects）通常解释了表型变异的主要部分（即狭义遗传力 $h^2$）。为了确保模型在引入深度神经网络处理复杂非线性信号的同时，不丢失基础的预测能力与稳健性，本研究设立了直接连接输入与输出的线性路径（Wide Component）。

该路径在数学上等效于一个带有 $L_2$ 正则化的线性回归模型（Ridge Regression），其前向传播过程定义为：

$$ \hat{y}_{add} = \mathbf{w}_{add}^T \mathbf{x} + b_{add} $$

其中：
*   $\mathbf{x} \in \{0, 1, 2\}^{M}$ 表示个体包含 $M$ 个 SNP 位点的基因型向量。
*   $\mathbf{w}_{add} \in \mathbb{R}^{M}$ 为加性效应权重向量，对应于每个 SNP 的独立效应值（SNP Effects）。
*   $b_{add}$ 为截距项，捕获群体的平均表型水平。

为了防止在高维特征（$M \gg N$）下出现过拟合，我们对权重 $\mathbf{w}_{add}$ 施加 $L_2$ 范数约束。这使得该流能够模拟传统的 GBLUP 或 RR-BLUP 模型，为整个架构提供一个稳健的性能下限（Baseline）。在反向传播中，该流的梯度更新独立于深度网络，确保了即使深度模块训练不稳定，模型整体仍能输出合理的加性预测值。

#### (2) 校正流 (The Correction Stream)

种群分层（Population Stratification）是基因组关联分析与预测中的主要混杂因素。不同亚群（Sub-populations）之间的等位基因频率差异可能导致伪关联。为了消除这种偏差，本架构将传统的 PCA 校正步骤内嵌至端到端的训练流程中。

输入向量 $\mathbf{p} \in \mathbb{R}^{K}$ 由全基因组基因型矩阵的前 $K$ 个主成分（Principal Components, 取 $K=10$）组成。校正流采用一个轻量级的多层感知机（MLP）来非线性地拟合由背景因素导致的表型差异：

$$ \mathbf{h}_{ctx} = \sigma(\mathbf{W}_{ctx}^{(1)} \mathbf{p} + \mathbf{b}_{ctx}^{(1)}) $$
$$ \hat{y}_{ctx} = \mathbf{w}_{ctx}^{(2)T} \mathbf{h}_{ctx} + b_{ctx}^{(2)} $$

其中：
*   $\mathbf{W}_{ctx}^{(1)}$ 和 $\mathbf{b}_{ctx}^{(1)}$ 分别为隐藏层的权重矩阵与偏置。
*   $\sigma(\cdot)$ 为非线性激活函数（ReLU），用于捕捉种群结构与表型之间的非线性关系（如某些性状在特定血统组合中表现出的非加性偏差）。
*   $\hat{y}_{ctx}$ 代表由种群结构解释的表型分量。

通过在总损失函数中引入正交约束（详见 2.4.4 节），迫使校正流与深度交互流提取的特征保持正交，从而确保 $\hat{y}_{ctx}$ 吸收了大部分环境与结构噪音，使后续模块能专注于学习纯净的遗传变异。

#### (3) 深度交互流 (The Deep Interaction Stream)

作为架构的创新核心，深度交互流（Deep Tower）摒弃了全连接网络的暴力拟合，采用了**先验引导的基因组 Transformer (Prior-Guided Genomic Transformer)** 结构。该流负责挖掘 SNP 序列内部的复杂非加性效应（如显性 Dominance）以及跨区域的上位性效应（Epistasis）。其构建包含以下三个关键阶段：

**A. 多模态先验融合与加权 (Prior Fusion & Reweighting)**

首先，我们将序列语义特征（PigBERT Delta, $\mathbf{E}_{\Delta} \in \mathbb{R}^{M \times d_1}$）与基因功能特征（Gene2Vec, $\mathbf{E}_{func} \in \mathbb{R}^{M \times d_2}$）通过一个非线性门控机制融合，生成每个 SNP 的综合重要性权重 $\mathbf{\Pi}$：

$$ \mathbf{\Pi} = \tanh \left( \mathbf{W}_{fuse} \cdot \left[ \text{ReLU}(\mathbf{W}_{\Delta} \mathbf{E}_{\Delta}) \parallel \text{ReLU}(\mathbf{W}_{func} \mathbf{E}_{func}) \right] + \mathbf{b}_{fuse} \right) $$

其中 $\parallel$ 表示张量拼接操作，$\mathbf{\Pi} \in \mathbb{R}^{M \times d_{model}}$ 为融合后的先验特征矩阵。接着，利用该先验矩阵对原始基因型嵌入 $\mathbf{X}_{emb}$ 进行逐元素的哈达玛积（Hadamard Product）调制：

$$ \mathbf{X}' = \mathbf{X}_{emb} \odot \mathbf{\Pi} $$

这一步在数学上实现了“生物学注意力”机制：具有显著序列突变影响或位于关键功能通路上的 SNP 位点，其特征幅值被放大；反之，位于功能荒漠区的噪声位点被抑制。

**B. 局部连锁不平衡卷积 (Local LD Convolution)**

考虑到连锁不平衡（LD）的局部性，相邻 SNP 往往作为一个遗传单元（单倍型块）共同作用。我们将加权后的特征序列 $\mathbf{X}'$ 划分为长度为 $L$ 的区块，并应用一维卷积进行局部特征聚合：

$$ \mathbf{z}_i = \text{MaxPool} \left( \text{ReLU} \left( \text{Conv1d}(\mathbf{X}'_{[i:i+L]}) \right) \right) $$

该操作将高维的 SNP 序列压缩为一系列紧凑的 **Block Tokens**序列 $\mathbf{Z} = \{\mathbf{z}_1, \mathbf{z}_2, \dots, \mathbf{z}_{M/L}\}$，有效降低了计算复杂度，同时提取了局部区域的遗传模式。

**C. 全局上位性自注意力 (Global Epistatic Self-Attention)**

为了捕捉远距离染色体区域之间的上位性互作，我们将 Block Tokens 输入到 Transformer 编码器中。利用多头自注意力机制（Multi-Head Self-Attention, MHSA），模型动态计算不同基因组区域之间的依赖关系：

$$ \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax} \left( \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} \right) \mathbf{V} $$

其中查询（Query）、键（Key）、值（Value）矩阵均由 Block Tokens 线性投影生成。注意力权重矩阵 $\mathbf{A} = \text{softmax}(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}})$ 显式地刻画了基因组中任意两个区域 $i$ 和 $j$ 之间的互作强度。最终，通过全局平均池化（GAP）聚合所有区域的信息，输出深度预测值：

$$ \hat{y}_{deep} = \text{MLP}_{out} \left( \frac{1}{T} \sum_{t=1}^{T} \text{Transformer}(\mathbf{Z})_t \right) $$

综上所述，Bio-Series 架构的最终预测输出为三流之和：
$$ \hat{y}_{total} = \hat{y}_{add} + \hat{y}_{ctx} + \hat{y}_{deep} $$
这种加性分解的设计，不仅保证了模型的可解释性，更实现了统计遗传学经典理论与现代深度学习技术的有机融合。
