# Bio-Series

Bio-Series is a genomic prediction repository centered on the `DF_GSF_v5` pipeline and a family of biologically informed deep models such as `bio_master_v8`, `bio_master_v10`, `bio_master_v11`, and `bio_master_v13`.

The model and file names are kept as-is for experiment compatibility. This repository has been reorganized so it can be cloned and configured on a new machine without editing hardcoded paths inside the source code.

## What This Repository Contains

- `DF_GSF_v5.py`: main pipeline entrypoint
- `src/gwas.py`: GWAS, PCA, and LD clumping
- `src/features.py`: PigBERT delta embeddings, GTF annotation, and Gene2Vec loading
- `src/data.py`: PLINK extraction helpers
- `src/models/`: model implementations, version names preserved
- `submit_jobs.py`: Slurm wrapper generation
- `test_gpu.py`: local environment and DeltaEngine smoke test
- `scripts/compare_v5_vs_baselines.py`: summarize experiment outputs

## What Is Not Included

This repository does not ship the large external resources required for training:

- genotype PLINK files
- phenotype tables
- train/test split files
- reference genome FASTA
- GTF annotation
- PigBERT model files
- Gene2Vec model files

You need to prepare those locally and point the config file to them.

## Requirements

Python dependencies:

```bash
pip install -r requirements.txt
```

System tools expected in `PATH`:

- `plink`
- `gcta64`

## Repository Layout

```text
Bio-Series/
├── DF_GSF_v5.py
├── config/
│   ├── config.yaml
│   ├── global_config.yaml
│   ├── v11_config.yaml
│   └── v12_config.yaml
├── scripts/
│   └── compare_v5_vs_baselines.py
├── src/
│   ├── data.py
│   ├── features.py
│   ├── gwas.py
│   └── models/
├── test_gpu.py
├── submit_jobs.py
└── thesis_materials/
```

## Configuration

The files under `config/` are now templates. All machine-specific absolute paths have been removed.

Recommended workflow:

```bash
cp config/global_config.yaml config/local.yaml
```

Then edit `config/local.yaml` to match your machine.

### Path Rules

- Relative paths are supported and are resolved from the repository root.
- `exp_root: "."` means outputs are written under the cloned repository.
- `plink_bin`, `gcta_bin`, and `python_bin` can be command names such as `plink`, `gcta64`, and `python`.

### Required Config Sections

You must set:

- `resources.reference_genome`
- `resources.gtf_file`
- `resources.pigbert_model`
- `resources.gene2vec_model`
- each dataset's `plink`
- each dataset's `pheno`

The dataset keys and trait names in the templates are kept close to the original experiments so old commands still make sense.

## Required Local File Structure

This code expects split files at:

```text
data/splits/<dataset>_<trait>/<rep>/train.ids
data/splits/<dataset>_<trait>/<rep>/test.ids
```

Each split file should contain two columns without a header:

```text
FID IID
```

Example:

```text
1001 1001
1002 1002
```

## Quick Start

Clone the repository:

```bash
git clone https://github.com/duzhch/Bio-Series.git
cd Bio-Series
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create and edit your local config:

```bash
cp config/global_config.yaml config/local.yaml
```

Run a smoke test for device selection and DeltaEngine initialization:

```bash
python test_gpu.py --config config/local.yaml
```

Run only the GWAS stage first:

```bash
python DF_GSF_v5.py run-gwas \
  --config config/local.yaml \
  --dataset LargeWhite_Pop1 \
  --trait BF \
  --rep rep_01 \
  --model bio_master_v11
```

Run the full pipeline:

```bash
python DF_GSF_v5.py run-all \
  --config config/local.yaml \
  --dataset LargeWhite_Pop1 \
  --trait BF \
  --rep rep_01 \
  --model bio_master_v11
```

## Pipeline Stages

`run-all` executes the following steps:

1. genome-wide PCA and GCTA MLMA
2. LD clumping and SNP selection
3. PigBERT delta embedding extraction
4. GTF-based gene annotation and Gene2Vec loading
5. PLINK extraction of selected SNP genotypes
6. model training and evaluation

## Batch Jobs

Generate Slurm jobs with a chosen model:

```bash
python submit_jobs.py \
  --config config/local.yaml \
  --datasets LargeWhite_Pop1 \
  --traits BF \
  --reps 1 \
  --model bio_master_v11 \
  --out-sh run_jobs.sh
```

Generate a CPU-only smoke-test job:

```bash
./generate_cpu_test.sh config/local.yaml
```

## Result Files

For a run like `dataset=LargeWhite_Pop1`, `trait=BF`, `model=bio_master_v11`, `rep=rep_01`, outputs will be placed under:

```text
results/LargeWhite_Pop1_BF/bio_master_v11/rep_01/
```

Typical artifacts include:

- `global_pca_features.csv`
- `snps_for_emb.csv`
- `selected_snp_ids.txt`
- `delta_embeddings.npy`
- `gene_knowledge.npy`
- `fusion_geno.bed/.bim/.fam`
- `best_model.pt`
- `pred.csv`
- `stats.json`

## Comparing Results

You can summarize experiment outputs with:

```bash
python scripts/compare_v5_vs_baselines.py \
  --root . \
  --out reports/compare
```

## Notes For Maintenance

- Model file names are intentionally unchanged.
- Generated artifacts such as `__pycache__`, local editor settings, transient CSV reports, and host-specific job wrappers are not tracked.
- If you add a new dataset, only update the config file. The pipeline code should not need new machine-specific edits.

## Citation

If you use this repository in research, cite the associated manuscript or project report for Bio-Series / DF_GSF_v5.
