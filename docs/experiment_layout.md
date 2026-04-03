# Experiment Layout

This document defines the runtime layout for `DF_GSF_v5` runs against `Bio-series-data`.

## Runtime Root

Recommended runtime root:

```text
runs/df_gsf_v5_bio_series_20260402/
```

All generated artifacts should stay under that root instead of mixing with source files.

## Directories

- `data/splits/`
  - staged train and test split files in launcher-compatible layout
  - pattern: `<dataset>_<trait>/rep_<NN>/train.ids` and `test.ids`
- `manifests/`
  - JSON manifests describing staged datasets and generated submission batches
- `slurm_jobs/scripts_v5/`
  - generated `sbatch` scripts
- `slurm_jobs/logs_v5/`
  - Slurm `.out` and `.err` logs
- `results/<dataset>_<trait>/<model>__<ablation>/<rep>/`
  - per-run outputs such as `run_meta.json`, `stats.json`, `pred.csv`, `best_model.pt`
- `summaries/ablation/`
  - outputs from `scripts/compare_ablations.py`
- `summaries/matrix/`
  - flattened run tables and trait-level aggregates for the full experiment matrix

## Batch Execution Rules

- Generate job scripts on login nodes.
- Submit training through Slurm only.
- Every job script must:
  - run with `set -euo pipefail`
  - `source ~/.bashrc`
  - `conda activate NT`
- Default to one node per job.

## Naming Conventions

- Baseline anchor: `full`
- Biological ablations:
  - `no_delta`
  - `no_gene2vec`
  - `no_bio_prior`
  - `no_pca`
  - `pca_only_prior_off`

- Fold-to-replicate mapping:
  - `fold_1 -> rep_01`
  - `fold_10 -> rep_10`
