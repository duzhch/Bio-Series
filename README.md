# Bio-Series

Bio-Series is a deep genomic prediction framework built around a biologically informed wide-and-deep architecture for livestock traits. The codebase combines three signal sources:

- additive SNP effects from genotype matrices
- population structure correction from genome-wide PCA
- biological priors from sequence-context embeddings and Gene2Vec features

The current implementation originated from the `DF_GSF_v5` experiment line and has been cleaned up here as a shareable GitHub repository.

## Repository Layout

```text
.
├── DF_GSF_v5.py                 # Main training / pipeline entrypoint
├── submit_jobs.py               # Slurm job generator and optional waiter
├── generate_cpu_test.sh         # Example helper to generate a CPU-only test job
├── test_gpu.py                  # Lightweight runtime/device sanity check
├── config/                      # Public-safe configuration templates
├── scripts/                     # Result summarization utilities
├── src/
│   ├── data.py                  # PLINK extraction helpers
│   ├── features.py              # Delta embedding and Gene2Vec feature builders
│   ├── gwas.py                  # GWAS + clumping pipeline
│   └── models/                  # Model variants
└── thesis_materials/bio_series/ # Writing notes and manuscript support files
```

## Method Summary

The pipeline runs in five stages:

1. GWAS screening with PCA correction and LD clumping to select informative SNPs.
2. Delta embedding extraction from sequence windows using a PigBERT-style encoder.
3. Gene annotation and Gene2Vec lookup for functional priors.
4. PLINK-based genotype extraction for the selected marker set.
5. Wide-and-deep model training with additive, context, and transformer-based components.

The deep tower is designed to capture non-linear and epistatic patterns, while the wide and context branches keep additive and population-structure effects explicit.

## Setup

Install Python dependencies first:

```bash
pip install -r requirements.txt
```

This project also expects external command-line tools to be available:

- `plink`
- `gcta64`

For feature generation you need local resource files that are not included in this repository:

- a reference genome FASTA
- a matching GTF annotation file
- a pretrained PigBERT model directory
- a Gene2Vec model file
- PLINK genotype datasets
- phenotype tables

## Configuration

The files in `config/` are templates intended for publication and reuse. Before running the pipeline, replace placeholder paths with your own local paths.

Recommended starting point:

```bash
cp config/global_config.yaml config/my_config.yaml
```

Then edit:

- `exp_root`
- every path under `resources`
- dataset-specific `plink` and `pheno` paths
- Slurm partition names if you use cluster submission

## Running

Single run:

```bash
python DF_GSF_v5.py run-all \
  --config config/my_config.yaml \
  --dataset Example_Population \
  --trait TraitA \
  --rep rep_01 \
  --model bio_master_v11
```

Only test the GWAS stage:

```bash
python DF_GSF_v5.py run-gwas \
  --config config/my_config.yaml \
  --dataset Example_Population \
  --trait TraitA \
  --rep rep_01 \
  --model bio_master_v11
```

Generate Slurm scripts:

```bash
python submit_jobs.py \
  --config config/my_config.yaml \
  --datasets Example_Population \
  --traits TraitA \
  --reps 3 \
  --out-sh run_jobs.sh
```

Aggregate benchmark results:

```bash
python scripts/compare_v5_vs_baselines.py
```

## Notes

- The repository intentionally excludes local caches, generated result files, and machine-specific paths.
- `test_gpu.py` is a sanity check only. It will skip model initialization until valid resource paths are configured.
- Some manuscript-support files are kept under `thesis_materials/` because they document the biological motivation and evaluation framing of the project.

## Citation

If you use this repository in research, cite the associated manuscript or project report for Bio-Series / DF_GSF_v5.
