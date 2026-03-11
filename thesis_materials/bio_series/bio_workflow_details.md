# BioMaster Framework: Deep Genomic Prediction with Biological Priors

## 1. System Architecture Overview

The **BioMaster** framework (DF_GSF_v5/v6) is a hybrid "Wide & Deep" deep learning system designed for genomic prediction in livestock (specifically pigs). It addresses the limitations of traditional linear models (GBLUP) by integrating:
1.  **Additive Effects**: Via a linear "Wide" component.
2.  **Population Structure**: Via a "Context" component (PCA).
3.  **Non-linear & Epistatic Effects**: Via a "Deep" Transformer component enhanced with biological priors.

The core innovation lies in the **Deep Tower**, which transforms SNP genotypes into biologically meaningful embeddings using **PigBERT** (sequence context) and **Gene2Vec** (functional context), then processes them through a **Genomic Transformer** to capture long-range interactions.

---

## 2. Data Processing Workflow

The pipeline processes raw genotype and phenotype data into model-ready tensors.

### 2.1 Input Data
*   **Genotype**: PLINK binary files (`.bed`, `.bim`, `.fam`).
*   **Phenotype**: CSV/TSV files containing trait values (e.g., ADG, FCR) and sample IDs.
*   **Reference Genome**: FASTA file (`Sus_scrofa.Sscrofa11.1.dna.toplevel.fa`) for sequence extraction.
*   **Annotation**: GTF file (`Sus_scrofa.Sscrofa11.1.111.gtf.gz`) for gene mapping.

### 2.2 Preprocessing Pipeline (`src/gwas.py`)
1.  **Quality Control & GWAS Screening**:
    *   **Tool**: GCTA (Genome-wide Complex Trait Analysis).
    *   **Step**: Calculate Genomic Relationship Matrix (GRM) to account for relatedness.
    *   **Step**: Run MLMA (Mixed Linear Model Association) to obtain P-values for each SNP, correcting for population structure.
2.  **LD Clumping (Feature Selection)**:
    *   **Tool**: PLINK.
    *   **Goal**: Select independent, significant SNPs to reduce dimensionality and redundancy.
    *   **Parameters**:
        *   P-value threshold (`--clump-p1 0.05`).
        *   $r^2$ threshold (`--clump-r2 0.2`) to remove highly correlated SNPs (linkage disequilibrium).
        *   Physical window (`--clump-kb 250`).
    *   **Output**: Top-N SNPs (e.g., 3000) used for downstream modeling.
3.  **Population Structure Analysis**:
    *   **Tool**: PLINK (`--pca`).
    *   **Output**: Top 10 Principal Components (PCs) used as input for the Context Tower.

---

## 3. Biological Feature Engineering (`src/features.py`)

This module generates the "Biological Priors" that enrich the genotype data.

### 3.1 Delta Embeddings (Sequence Context via PigBERT)
*   **Hypothesis**: SNPs affect phenotypes by altering local sequence semantics (e.g., transcription factor binding affinity).
*   **Process**:
    1.  **Window Extraction**: For each selected SNP, extract a 512bp window from the reference genome centered on the SNP.
    2.  **Sequence Pairing**: Generate two sequences:
        *   **Reference Sequence**: Original sequence from the genome.
        *   **Alternative Sequence**: Same sequence but with the SNP site mutated to the alternative allele.
    3.  **PigBERT Encoding**: Feed both sequences into **PigBERT** (a BERT model pre-trained on pig genomes).
    4.  **Delta Calculation**: Extract the embedding vector (last hidden state, index 0) for both sequences ($E_{ref}$, $E_{alt}$).
    5.  **Result**: $\Delta E = E_{alt} - E_{ref}$. This vector represents the *semantic shift* caused by the mutation.

### 3.2 Gene2Vec Embeddings (Functional Context)
*   **Hypothesis**: SNPs within functionally related genes should have similar representations.
*   **Process**:
    1.  **SNP-to-Gene Mapping**: Use `IntervalTree` for efficient geometric querying of the GTF file to map each SNP to its overlapping gene. Intergenic SNPs are mapped to "Intergenic" or nearest gene.
    2.  **Embedding Lookup**: Query a pre-trained **Gene2Vec** model (trained on co-expression networks) to retrieve the functional vector for the mapped gene.
    3.  **Result**: A functional embedding vector $E_{gene}$ for each SNP.

---

## 4. BioMaster Model Architecture (`src/models/bio_master_v13.py`)

The model (specifically `BioMasterV12`) is a multi-modal neural network.

### 4.1 Input Layers
*   **Genotypes ($X_{snp}$)**: shape $[Batch, N_{SNPs}]$, values $\{0, 1, 2\}$.
*   **PCs ($X_{pca}$)**: shape $[Batch, 10]$.
*   **Priors**:
    *   Delta Embeddings ($E_{\Delta}$): shape $[N_{SNPs}, 768]$.
    *   Gene2Vec Embeddings ($E_{gene}$): shape $[N_{SNPs}, 300]$.

### 4.2 Tower 1: Wide Tower (Additive)
*   **Type**: Linear Regression.
*   **Operation**: $y_{wide} = W_{wide} \cdot X_{snp} + b_{wide}$.
*   **Purpose**: Captures simple additive effects (equivalent to GBLUP), ensuring baseline performance.

### 4.3 Tower 2: Context Tower (Population)
*   **Type**: Multi-Layer Perceptron (MLP).
*   **Structure**: Linear $\rightarrow$ ReLU $\rightarrow$ Linear.
*   **Operation**: $y_{context} = MLP(X_{pca})$.
*   **Purpose**: Explicitly models and corrects for population stratification (batch effects, breed differences).

### 4.4 Tower 3: Deep Tower (Epistatic/Non-linear)
This is the core "Genomic Transformer" component.

1.  **Rich Prior Generation**:
    *   Fuses $E_{\Delta}$ and $E_{gene}$ into a unified prior tensor $P$ via a non-linear projection (`RichPriorGenerator`).
    *   $P = \text{Fuse}(\text{ReLU}(W_1 E_{\Delta}), \text{ReLU}(W_2 E_{gene}))$.

2.  **Genotype-Prior Interaction**:
    *   The genotype input is reshaped into blocks (e.g., 100 SNPs per block) to handle Linkage Disequilibrium (LD).
    *   Genotypes scale the priors: $X_{interaction} = X_{snp\_blocked} \odot P_{blocked}$.

3.  **Local Convolution (LD Modeling)**:
    *   **Layer**: `Conv1d` + `MaxPool1d`.
    *   **Action**: Compresses each block of 100 SNPs into a single "Block Token".
    *   **Intuition**: Aggregates local genetic information, mimicking haplotype blocks.

4.  **Transformer Encoder (Global Epistasis)**:
    *   **Input**: Sequence of Block Tokens + Positional Encodings.
    *   **Mechanism**: Multi-Head Self-Attention.
    *   **Purpose**: Allows the model to "attend" to interactions between distant chromosomal regions (epistasis).

5.  **Global Average Pooling (GAP)**:
    *   Averages all output tokens to produce a single graph-level embedding.
    *   Reduces overfitting compared to Flatten layers.

6.  **Prediction**:
    *   $y_{deep} = MLP(GAP\_Output)$.

### 4.5 Final Prediction
$$ \hat{y} = y_{wide} + y_{context} + y_{deep} $$

---

## 5. Loss Function: HybridPCCLoss

The model optimizes a composite objective to balance accuracy, ranking, and independence.

$$ \mathcal{L} = \alpha \cdot \mathcal{L}_{PCC} + \beta \cdot \mathcal{L}_{Rank} + \gamma \cdot \mathcal{L}_{Orth} $$

1.  **PCC Loss ($\mathcal{L}_{PCC}$)**:
    *   Formula: $1 - \text{PearsonCorr}(y, \hat{y})$.
    *   Goal: Directly maximizes the correlation, which is the gold standard in genomic prediction.
    *   Robustness: Includes epsilon terms to prevent division by zero.

2.  **Rank Loss ($\mathcal{L}_{Rank}$)**:
    *   Type: ListNet Loss.
    *   Formula: Cross-entropy between softmax-normalized true and predicted distributions.
    *   Goal: Focuses on correctly ordering individuals (crucial for selecting top candidates in breeding).

3.  **Orthogonality Loss ($\mathcal{L}_{Orth}$)**:
    *   Formula: Cosine similarity between Deep features and Context features.
    *   Goal: Forces the Deep Tower to learn genetic effects *orthogonal* to population structure, preventing it from just learning breed labels.

---

## 6. Software & Dependencies

### Core Libraries
*   **PyTorch**: Deep learning framework (Model construction, autograd).
*   **Transformers (HuggingFace)**: Loading and running PigBERT.
*   **Pandas/NumPy**: Data manipulation and numerical operations.
*   **Scikit-learn**: Metrics (MSE), Preprocessing (StandardScaler).
*   **SciPy**: Statistics (Pearson correlation).

### Bioinformatics Tools (External)
*   **PLINK (v1.9/2.0)**: Binary genotype file handling, PCA, Clumping.
*   **GCTA**: GRM calculation, MLMA (GWAS).
*   **PySAM**: FASTA file indexing and sequence extraction.
*   **IntervalTree**: Efficient genomic coordinate queries for annotation.

### Custom Scripts
*   `pandas_plink`: Reading PLINK binary files into Python.

