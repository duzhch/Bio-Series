# 2.4.3 Bio 系列增强架构设计 (Bio-Series Enhanced Architecture Design)

尽管上一节所述的基于随机森林的 DF-GSF 框架通过张量融合在一定程度上捕捉了基因型与生物学先验的交互，但作为浅层机器学习模型，其在处理高维稀疏数据时仍存在局限性：(1) **特征交互的静态性**：随机森林依赖于预先计算的交互张量，难以在训练过程中动态调整不同生物学特征的权重；(2) **上位性效应的局部性**：基于树的模型虽然能捕捉非线性，但在模拟全基因组范围内跨染色体的长距离互作（Epistasis）方面能力有限；(3) **种群结构的隐式处理**：缺乏明确的机制将种群分层造成的伪关联与真实遗传效应解耦。

为了克服上述瓶颈，本研究进一步提出了 **Bio-Series 增强架构**（具体实现为 BioMaster 模型）。该架构采用了 **"Wide & Deep & Context"** 的混合深度学习范式，旨在构建一个端到端的、具有生物学可解释性的深度基因组预测系统。该系统不仅继承了线性模型在加性效应上的稳健性，更引入了基于 Transformer 的深度网络来动态学习复杂的上位性网络，并显式地集成了更深层次的生物学先验——序列突变语义（PigBERT Delta）与基因功能网络（Gene2Vec）。

#### 2.4.3.1 混合多塔架构 (Hybrid Multi-Tower Architecture)

Bio-Series 架构的核心思想是将基因组预测分解为三个并行的任务流，分别由三个独立的神经网络“塔”（Tower）处理，最终在输出层进行加权融合。其数学形式可以表示为：

$$ \hat{y}_i = \mathcal{F}_{Wide}(\mathbf{g}_i) + \mathcal{F}_{Context}(\mathbf{p}_i) + \mathcal{F}_{Deep}(\mathbf{g}_i, \mathbf{E}_{prior}) $$

其中，$\mathbf{g}_i$ 为个体 $i$ 的 SNP 基因型向量，$\mathbf{p}_i$ 为种群结构主成分向量，$\mathbf{E}_{prior}$ 为全局共享的生物学先验矩阵。

1.  **Wide Tower (加性效应流)**：
    *   **设计逻辑**：尽管深度学习擅长捕捉非线性，但在数量遗传学中，加性效应（Additive Effects）仍然解释了大部分的遗传方差。为了防止深度网络在小样本下过拟合而丢失基础预测能力，我们保留了一个直接连接输入与输出的线性层。
    *   **实现细节**：该部分等效于一个大规模的线性回归模型（类似于 GBLUP 或 RR-BLUP）。它直接对原始的 $\{0, 1, 2\}$ 基因型编码进行加权求和。通过 L2 正则化，该塔能够快速收敛并提供稳健的性能下限（Baseline）。

2.  **Context Tower (种群结构流)**：
    *   **设计逻辑**：种群分层（Population Stratification）是基因组选择中的主要混杂因素。传统的校正方法通常是在数据预处理阶段进行，而本架构将其纳入端到端的训练流程。
    *   **实现细节**：输入为前 $k$ 个（本研究取 $k=10$）基因组主成分（PCs）。采用轻量级的多层感知机（MLP）结构（Linear $\rightarrow$ ReLU $\rightarrow$ Linear），专门用于拟合由品种、家系等背景因素导致的表型差异。这一设计使得模型能够显式地“减去”环境与结构偏差，迫使 Deep Tower 专注于学习真实的遗传效应。

3.  **Deep Tower (深度交互流)**：
    *   **设计逻辑**：这是架构的创新核心，旨在利用生物学先验指导深度神经网络探索复杂的基因型-表型图谱。不同于简单的全连接网络，Deep Tower 采用了一种 **"Prior-Guided Genomic Transformer"** 结构，包含先验融合、局部卷积压缩和全局自注意力机制三个阶段。

#### 2.4.3.2 多模态生物学先验的深度融合 (Deep Fusion of Multi-modal Priors)

在 2.4.2 节中，我们仅使用了单一的 GWAS 统计权重。在 Bio-Series 架构中，我们引入了更本质的生物学特征，从“序列”和“功能”两个维度对 SNP 进行重新定义。

1.  **序列语义先验 (Sequence Semantic Prior - PigBERT Delta)**：
    *   传统的 SNP 编码（0/1/2）丢失了位点周围的序列上下文信息。本研究利用在大规模猪基因组序列上预训练的 **PigBERT** 模型，提出了一种 **Delta Embedding** 策略。
    *   对于每个 SNP 位点 $j$，我们提取其参考等位基因序列 $S_{ref}$ 和突变等位基因序列 $S_{alt}$（窗口大小 512bp）。将两者分别输入 PigBERT，提取 `[CLS]` 处的上下文向量 $v_{ref}$ 和 $v_{alt}$。
    *   计算差异向量 $\mathbf{e}_{\Delta, j} = v_{alt} - v_{ref}$。该向量在 768 维的高维语义空间中，精确刻画了该单核苷酸突变对局部 DNA 序列语义（如转录因子结合亲和力、染色质开放性倾向）的扰动程度。

2.  **基因功能先验 (Functional Prior - Gene2Vec)**：
    *   为了捕捉基因间的共表达与调控关系，我们引入了 **Gene2Vec** 嵌入。通过 GTF 注释文件将 SNP 映射到其所在的基因区域。
    *   利用预训练的 Gene2Vec 模型（在猪的多组织转录组数据上训练），获取该基因的 300 维功能向量 $\mathbf{e}_{func, j}$。这使得模型能够理解“虽然 SNP A 和 SNP B 物理距离很远，但它们位于同一代谢通路的基因上，因此可能具有相似的效应模式”。

3.  **动态先验生成器 (Rich Prior Generator)**：
    *   上述两种先验被输入到一个非线性融合模块中。该模块由两个独立的压缩层（Linear）和一个融合层组成：
        $$ \mathbf{P}_j = \text{Tanh}(\mathbf{W}_f \cdot [\text{ReLU}(\mathbf{W}_d \mathbf{e}_{\Delta, j}) \oplus \text{ReLU}(\mathbf{W}_g \mathbf{e}_{func, j})]) $$
    *   最终生成的先验向量 $\mathbf{P}_j \in \mathbb{R}^{64}$ 是一个紧凑的、融合了序列微观语义与基因宏观功能的特征表示，它将作为后续 Transformer 的“位置特征”或“权重调节器”。

#### 2.4.3.3 基因组 Transformer 机制 (Genomic Transformer Mechanism)

为了解决全基因组范围内 SNP 数量巨大（通常 > 50k）导致的“维度灾难”以及长距离连锁不平衡（LD）问题，我们设计了 **Local-to-Global** 的两级处理机制。

1.  **局部区块卷积 (Local Block Convolution)**：
    *   **LD Block 建模**：依据连锁不平衡原理，相邻的 SNP 往往共同遗传。我们将基因型向量切分为长度为 $L$（如 100）的区块（Blocks）。
    *   **先验加权**：在卷积之前，利用生成的先验向量 $\mathbf{P}$ 对基因型 $\mathbf{X}$ 进行逐元素的哈达玛积（Hadamard Product）缩放：$\mathbf{X}' = \mathbf{X} \odot \mathbf{P}$。这一步相当于利用生物学知识对原始基因型进行了“特征增强”，放大了潜在功能位点的信号。
    *   **特征压缩**：使用一维卷积层（Conv1d）配合最大池化（MaxPool）处理每个区块。这不仅将高维的 SNP 序列压缩为一系列低维的 **"Block Tokens"**，还有效地提取了局部单倍型（Haplotype）特征。

2.  **全局自注意力 (Global Self-Attention)**：
    *   **上位性建模**：经过压缩的 Block Tokens 序列保留了基因组的物理顺序。我们引入位置编码（Positional Encoding），并将其输入到标准的 Transformer Encoder 层。
    *   **动态互作**：利用多头自注意力机制（Multi-Head Self-Attention），模型能够动态计算任意两个染色体区域之间的注意力权重。这意味着模型可以自动发现并聚焦于对表型有显著贡献的基因-基因相互作用（即上位性效应），而无需人工定义交互项。这对于解析如“加性 $\times$ 加性”或“显性 $\times$ 上位性”等复杂遗传机制至关重要。

3.  **全局平均池化 (Global Average Pooling, GAP)**：
    *   为了进一步降低过拟合风险，我们在 Transformer 输出层之后摒弃了传统的 Flatten 操作，转而采用 GAP。无论输入序列多长，GAP 都会将其聚合为固定维度的全局特征向量。这使得 Bio-Series 架构对 SNP 数量的变化具有鲁棒性，并且极大减少了模型参数量。

---

# 2.4.4 模型优化机制 (Model Optimization Mechanism)

在构建了复杂的 Bio-Series 深度神经网络架构后，如何有效地训练该模型成为关键挑战。基因组预测任务具有独特的数学特性：样本量远小于特征数（$N \ll p$）、信噪比低、且育种实践更关注个体的相对排序而非绝对预测值。针对这些特性，本研究设计了一套包含混合损失函数、正交约束及特定训练策略的优化机制。

#### 2.4.4.1 鲁棒混合损失函数 (Robust Hybrid Loss Function)

传统的均方误差（MSE）损失函数在基因组预测中往往表现不佳，因为它容易受到离群点的影响，且过度关注数值拟合而忽略了育种值的分布规律。Bio-Series 架构采用了一种三元混合损失函数，旨在同时优化线性相关性、排序准确性和特征解耦性：

$$ \mathcal{L}_{total} = \alpha \cdot \mathcal{L}_{PCC} + \beta \cdot \mathcal{L}_{Rank} + \gamma \cdot \mathcal{L}_{Orth} $$

1.  **皮尔逊相关性损失 ($\mathcal{L}_{PCC}$)**：
    *   **动机**：在育种中，预测准确性通常由预测育种值（GEBV）与真实表型值之间的皮尔逊相关系数（PCC）来衡量。直接优化 PCC 可以最大化模型在育种评价指标上的表现。
    *   **公式**：
        $$ \mathcal{L}_{PCC} = 1 - \frac{\sum (y_i - \bar{y})(\hat{y}_i - \bar{\hat{y}})}{\sqrt{\sum (y_i - \bar{y})^2} \cdot \sqrt{\sum (\hat{y}_i - \bar{\hat{y}})^2} + \epsilon} $$
    *   **数值稳定性**：为了防止在训练初期因预测值方差极小而导致的除零错误，我们在分母中引入了平滑项 $\epsilon = 10^{-6}$。该损失函数迫使模型的预测趋势与真实值保持高度一致的线性关系。

2.  **ListNet 排序损失 ($\mathcal{L}_{Rank}$)**：
    *   **动机**：基因组选择的最终目标是选留排名靠前的优良个体。MSE 或 PCC 关注的是全局拟合，而可能忽略了局部的相对顺序。ListNet 是一种成对排序学习（Learning to Rank）算法，它将排序问题转化为概率分布的差异问题。
    *   **公式**：我们使用 Softmax 函数将真实值和预测值映射为概率分布 $P(y)$ 和 $P(\hat{y})$，然后计算两者之间的交叉熵：
        $$ \mathcal{L}_{Rank} = - \sum P(y_i) \log(P(\hat{y}_i) + \epsilon) $$
    *   **作用**：该损失函数对 Top-K 个体的排序极其敏感。它引导模型将注意力集中在区分高表型值的优良个体上，这与育种实践中“只选最好的”这一需求完美契合。

3.  **正交解耦损失 ($\mathcal{L}_{Orth}$)**：
    *   **动机**：深度神经网络具有极强的拟合能力，容易通过拟合种群结构（如品种间的系统性差异）来“作弊”降低 Loss，而非真正学习到 QTL（数量性状位点）的效应。为了迫使 Deep Tower 学习到独立于种群结构的生物学特征，我们引入了正交约束。
    *   **公式**：计算 Deep Tower 提取的特征向量 $\mathbf{f}_{deep}$ 与 Context Tower 提取的种群特征 $\mathbf{f}_{context}$ 之间的余弦相似度：
        $$ \mathcal{L}_{Orth} = \frac{1}{N} \sum_{i=1}^N | \cos(\mathbf{f}_{deep}^{(i)}, \mathbf{f}_{context}^{(i)}) | $$
    *   **作用**：最小化该损失迫使两个特征空间相互正交。这意味着 Deep Tower 必须在剔除了 PCA 能解释的变异之后，去挖掘剩余的、更深层的遗传变异，从而提高了模型的泛化能力和生物学可解释性。

#### 2.4.4.2 训练策略与正则化 (Training Strategy and Regularization)

除了损失函数的设计，训练过程的动态控制对于深度模型的收敛同样至关重要。

1.  **动态学习率调度 (Dynamic Learning Rate Scheduling)**：
    *   采用了 **ReduceLROnPlateau** 策略。初始学习率设定为 $5 \times 10^{-4}$，使用 AdamW 优化器。在训练过程中，持续监控验证集的 Loss。当验证集性能在连续 5 个 Epoch 内未提升时，将学习率衰减为原来的 0.5 倍。这种机制允许模型在训练初期快速下降，在后期微调参数以逼近全局最优解。

2.  **先验稀疏性正则化 (Prior Sparsity Regularization)**：
    *   考虑到全基因组中大部分区域可能是功能缺失的“垃圾 DNA”，我们期望模型生成的先验权重 $\mathbf{P}$ 具有稀疏性。因此，我们在总损失中加入了一项针对先验向量的 L1 正则化项：
        $$ \mathcal{L}_{final} = \mathcal{L}_{total} + \lambda \|\mathbf{P}\|_1 $$
    *   这迫使模型将注意力权重集中在少数具有强生物学证据（如高 Delta 分数或特定 Gene2Vec 模式）的区域，抑制了噪声信号的干扰。

3.  **梯度裁剪与早停机制 (Gradient Clipping & Early Stopping)**：
    *   由于 Transformer 结构较深，且涉及高阶交互，反向传播时容易出现梯度爆炸。我们在训练循环中实施了范数梯度裁剪（Max Norm = 1.0），确保参数更新的稳定性。
    *   同时，为了防止过拟合，我们设置了基于验证集 PCC 的早停机制（Patience = 15）。如果模型在 15 个 Epoch 内无法在未见过的验证集上取得更高的相关系数，则立即停止训练，并回滚至最佳模型权重。

通过上述多维度的优化机制，Bio-Series 架构不仅在数学上保证了收敛的稳定性，更在生物学意义上确保了模型学习到的特征是鲁棒的、可解释的且符合育种需求的。这种深度学习与统计遗传学原理的深度耦合，正是本研究区别于传统“黑箱”模型的关键所在。
