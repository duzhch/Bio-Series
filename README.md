# DF_GSF_v5: Deep Genomic Prediction Framework with Biological Priors

## 1. Abstract (摘要)

**DF_GSF_v5** (Deep Feature Genomic Selection Framework v5) 是一个前沿的基因组预测（Genomic Prediction, GP）深度学习框架。旨在解决传统线性模型（如 GBLUP）在捕获非加性效应（显性、上位性）方面的局限性，以及常规深度学习模型容易过拟合且缺乏生物学可解释性的问题。

本项目创新性地提出了一种 **"Biologically-Informed Wide & Deep Transformer"** 架构（即 `BioMaster` 模型），通过融合多模态数据来提升预测精度：
1.  **Genotype Data**: 传统的 SNP 基因型矩阵。
2.  **Sequence Context (Delta Embeddings)**: 利用 **PigBERT** 预训练语言模型提取 SNP 位点突变引起的序列语义变化。
3.  **Functional Context (Gene2Vec)**: 利用 **Gene2Vec** 嵌入捕获 SNP 所在基因的共表达和功能互作网络信息。
4.  **Population Structure (PCA)**: 显式校正种群分层结构。

---

## 2. Methodology (原理详解)

本框架的核心在于将基因组预测视为一个**多模态特征融合**问题，而非单纯的矩阵回归。

### 2.1 The BioMaster Model Architecture
模型采用 **Wide & Deep** 的范式，由三个并行的 "Tower" 组成，分别处理不同类型的信息：

#### A. Wide Tower (Additive Effects)
*   **输入**: 原始 SNP 基因型编码 (0, 1, 2)。
*   **结构**: 单层线性网络 (`nn.Linear`)。
*   **作用**: 模拟 GBLUP/Bayes 线性模型，高效捕获全基因组范围内的**加性效应**（Additive Effects）。这是表型变异的主要来源，保留 Wide 部分保证了模型的基础性能下限。

#### B. Context Tower (Population Structure)
*   **输入**: 基因型数据的前 $k$ 个主成分 (PCA)。
*   **结构**: 轻量级 MLP (`Linear -> ReLU -> Linear`)。
*   **作用**: 显式地建模种群结构（Population Structure），消除因种群分层导致的伪关联，相当于传统关联分析中的协变量校正。

#### C. Deep Tower (Genomic Transformer)
这是模型的核心创新点，用于捕获**非线性效应**（Non-linear Effects）和**上位性效应**（Epistasis）。

1.  **Prior Knowledge Integration (先验融合)**:
    *   输入：Delta Embeddings ($E_{\Delta}$) 和 Gene2Vec Embeddings ($E_{Gene}$)。
    *   操作：通过 `PriorGenerator` 子模块将两种生物学先验映射到同一潜在空间并融合，生成每个 SNP 的先验特征向量 $P$。
2.  **Local Convolution (局部特征提取)**:
    *   将基因组划分为长度为 $L$ 的 Block（如 100 SNPs）。
    *   使用 1D 卷积层和 MaxPool 处理每个 Block，将高维的 SNP 序列压缩为紧凑的 **Block Tokens**。这不仅降低了计算复杂度，还利用了连锁不平衡（LD）的局部特性。
3.  **Transformer Encoder (全局互作)**:
    *   使用标准的 Transformer Encoder 处理 Block Tokens 序列。
    *   **Self-Attention 机制**允许模型在全基因组范围内动态关注不同染色体区域之间的相互作用，从而有效建模**上位性效应**（Epistasis）。
4.  **Global Average Pooling (GAP)**:
    *   替代传统的 Flatten 操作，对 Transformer 输出的所有 Token 取平均。
    *   **作用**: 显著减少模型参数量，极大降低了在小样本（$N \ll p$）场景下的过拟合风险。

### 2.2 Hybrid Loss Function
为了提升模型的泛化能力和训练稳定性，我们设计了混合损失函数 `HybridPCCLoss`：
$$ \mathcal{L} = \alpha \cdot \mathcal{L}_{PCC} + \beta \cdot \mathcal{L}_{Rank} + \gamma \cdot \mathcal{L}_{Orth} $$

*   **$\mathcal{L}_{PCC}$ (1 - Pearson Correlation)**: 直接优化预测值与真实值的线性相关性，这是育种中最重要的指标。
*   **$\mathcal{L}_{Rank}$ (ListNet Loss)**: 学习样本的相对排序，关注选育顶端个体的准确性。
*   **$\mathcal{L}_{Orth}$ (Orthogonality Loss)**: 强制 Deep Tower 和 Context Tower 提取的特征正交，确保深度网络学习到的是种群结构之外的真实遗传效应，而非简单的混杂因素。

### 2.3 Feature Engineering Engines

#### Delta Engine (PigBERT)
*   **原理**: 类似 NLP 中的情感分析。对于每个 SNP，提取其 Reference 和 Alternative 等位基因的上下游序列（Window=512bp）。
*   **计算**: 输入到预训练的 **PigBERT** 模型，提取 `[CLS]` 或最后一层 Hidden States。
*   **Delta**: 计算 $\Delta = E(Seq_{Alt}) - E(Seq_{Ref})$。这个差值向量代表了该突变对局部序列语义（如转录因子结合位点亲和力）的扰动。

#### Gene2Vec Engine
*   **原理**: 基于“共表达基因具有相似功能”的假设。
*   **流程**: 
    1. 使用 GTF 注释文件将 SNP 映射到基因（Intergenic SNP 映射到最近基因或标记为未知）。
    2. 查询在大规模转录组数据上预训练的 Gene2Vec 模型。
    3. 赋予 SNP 所在基因的功能向量，使模型能利用基因通路层面的信息。

### 2.4 Model Evolution: BioMaster v9 vs v10
为了进一步提升模型性能和稳定性，我们对架构进行了重要升级。以下是 v9 到 v10 的核心改进对比：

| 特性 (Feature) | BioMaster v9 | BioMaster v10 | 改进意义 (Improvement) |
| :--- | :--- | :--- | :--- |
| **核心编码器 (Encoder)** | **CNN (LDBlockEncoder)** | **Transformer (GenomicTransformer)** | v9 仅通过 1D 卷积捕获局部连锁不平衡 (LD)；v10 引入 Transformer 捕获全基因组范围的长程上位性效应 (Epistasis)。 |
| **输入处理 (Input)** | 局部卷积 + 投影 | 局部卷积 (Token化) + 位置编码 | v10 将 SNP Block 视为 Token 序列，结合位置编码保留了染色体上的空间结构信息。 |
| **特征聚合 (Pooling)** | **Flatten + Linear** | **Global Average Pooling (GAP)** | v9 展平后参数量随 SNP 数量线性膨胀，易过拟合；v10 使用 GAP 聚合，大幅减少参数量，提升泛化能力。 |
| **稳定性 (Stability)** | 标准 Loss | **Robust Hybrid Loss** | v10 增加了数值稳定性保护 (epsilon, gradient clipping)，有效防止 Transformer 训练中的梯度爆炸或消失。 |

---

## 3. Project Structure (项目结构)

```text
DF_GSF_v5/
├── DF_GSF_v5.py                # [Main] 项目主入口，串联全流程
├── submit_jobs.py              # [Slurm] 批量作业生成与提交工具
├── config/                     # [Config] 实验配置文件
│   ├── config.yaml             # 通用配置
│   └── v11_config.yaml         # v11 模型特定配置 (推荐)
├── src/                        # [Source] 核心源代码
│   ├── data.py                 # PLINK 数据处理与提取工具
│   ├── features.py             # 特征工程 (PigBERT Delta, Gene2Vec)
│   ├── gwas.py                 # GWAS 筛选与 LD Clumping 流程
│   └── models/                 # 模型定义
│       ├── bio_master_v11.py   # 核心模型代码 (BioMasterV10 类)
│       └── ...                 # 其他历史版本
├── scripts/                    # [Scripts] 辅助分析脚本
│   └── compare_v5_vs_baselines.py # 结果对比与绘图数据生成
└── jobs.sh                     # Slurm 作业提交示例
```

---

## 4. Usage Guide (使用指南)

### 4.1 Environment Setup
需要 Python 3.8+ 及 PyTorch 环境。依赖库包括 `pandas_plink`, `transformers`, `pysam`, `intervaltree` 等。

### 4.2 Configuration
修改 `config/v11_config.yaml` 以适配你的数据：

```yaml
resources:
  ref_genome: "/path/to/pig.fa"
  gtf: "/path/to/annotation.gtf.gz"
  pigbert_model: "/path/to/pigbert"
  gene2vec_model: "/path/to/gene2vec.bin"

datasets:
  - name: "LargeWhite_Pop"
    plink: "/path/to/plink_data"
    pheno: "/path/to/phenotypes.csv"
    traits: ["ADG", "FCR"]
```

### 4.3 Running Single Experiment
使用主入口脚本运行单次实验：

```bash
# 运行默认配置 (使用 config/v11_config.yaml)
python DF_GSF_v5.py

# 或者指定配置文件
python DF_GSF_v5.py --config config/my_custom_config.yaml
```

**程序执行流程 (`step_run_all`):**
1.  **GWAS Screening**: 对每个性状运行 GWAS，进行 PCA 校正，计算 P-value。
2.  **LD Clumping**: 筛选显著且独立的 Top-N SNPs (默认 3000)。
3.  **Feature Extraction**:
    *   调用 `src.features.extract_delta` 计算 Delta Embeddings。
    *   调用 `src.features.annotate_snps_with_gtf` 和 `extract_gene2vec` 获取基因功能向量。
4.  **Data Preparation**: 使用 PLINK 提取筛选后 SNP 的基因型矩阵。
5.  **Model Training**: 启动 `BioMaster` 模型训练，自动进行 Train/Test 划分和评估。

### 4.4 Batch Experiments (Slurm)
对于大规模实验，建议使用 `submit_jobs.py` 生成并提交 Slurm 作业。

```bash
# 生成并提交作业，同时等待所有任务完成 (--wait)
python submit_jobs.py \
    --config config/v11_config.yaml \
    --datasets LargeWhite_Pop \
    --traits ADG,FCR \
    --reps 10 \
    --out-sh run_experiments.sh \
    --wait
```
此命令会自动：
1. 为每个数据集、每个性状、每个重复（Rep）生成单独的 `.sh` 脚本。
2. 提交这些脚本到 Slurm 集群。
3. 实时监控任务状态，直到所有任务完成。

### 4.5 Result Analysis
训练完成后，结果将保存在 `results/<dataset>/<trait>/` 目录下：
*   `best_model.pt`: 最佳模型权重。
*   `stats.json`: 测试集指标 (PCC, MSE)。
*   `pred.csv`: 详细的个体预测值 vs 真实值。

运行对比脚本生成汇总报告：
```bash
python scripts/compare_v5_vs_baselines.py
```
该脚本会汇总所有重复实验的结果，计算平均 PCC 和标准差，并生成 CSV 报告。

---

## 5. Script Details (脚本详解)

### `src/models/bio_master_v11.py`
*   **`BioMasterV10` 类**: 定义了完整的神经网络结构。
    *   `__init__`: 初始化 Embedding 层，构建 Wide, Context, Deep 三个塔。
    *   `forward`: 定义前向传播路径，包括 Transformer 的 Fold/Unfold 操作。
*   **`HybridPCCLoss` 类**: 实现了鲁棒的混合损失函数，包含防止除零错误的 epsilon 和梯度裁剪保护。
*   **`train` 函数**: 封装了完整的训练循环、验证逻辑、Early Stopping 和模型保存。

### `src/features.py`
*   **`_DeltaEngine`**: 封装了 `HuggingFace Transformers` 库，加载 PigBERT 模型。包含 `_paired` 方法用于构建 Reference/Alternative 序列对。
*   **`_load_gtf_optimized`**: 使用 `IntervalTree` 数据结构加速 GTF 文件的区间查询，比传统遍历快数倍。
*   **`extract_gene2vec`**: 智能处理 Gene ID 版本号差异（如去除小数点后的版本号），保证最大的匹配率。

### `DF_GSF_v5.py`
*   **`step_run_all`**: 核心调度器。它利用文件指纹（Checksum）或存在性检查来跳过已完成的步骤（断点续传功能）。
*   它动态加载模型类（通过 `importlib`），使得切换不同版本的模型架构非常灵活，无需修改主逻辑。

### `submit_jobs.py`
*   **功能**: 自动化作业管理工具。
*   **Job Generation**: 根据配置文件自动生成 Slurm `.sh` 脚本，支持 CPU/GPU 分区选择。
*   **Job Monitoring**: 提供 `--wait` 模式，通过轮询 `squeue` 监控作业状态，适合在工作流中阻塞等待实验完成。

---

## 6. Citation
If you use this code for your research, please cite:
> [Author Name], et al. "Integrating Biological Priors with Deep Learning for Genomic Prediction in Pigs." *Journal of Animal Science/Genetics*, 202X.
