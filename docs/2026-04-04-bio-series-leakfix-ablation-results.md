# Bio-series Leak-Fix Baseline and Ablation Results

This document summarizes the repaired `DF_GSF_v5` Bio-series experiment run rooted at:

```text
/work/home/zyqlab/dzhichao/zyqgroup02_duanzhichao/exp_00/experiments/01_benchmarks/runs/df_gsf_v5_bio_series_20260403_leakfix
```

The run uses the leakage-fixed evaluation protocol in `bio_master_v11`:

- model selection uses an internal train/validation split from `train.ids`
- predictor-side PCA/context features are fit on the training subset only and then projected to validation and test
- final metrics are still reported on `test.ids`

## Code and Config Used

- Main repository:
  - `Bio-Series_repo/`
- Main branch during repair work:
  - `fix/ablation-eval-protocol`
- Cluster config for this run:
  - `config/bio_series_data.cluster.leakfix_20260403.yaml`

Main scripts used for this experiment:

- `submit_jobs.py`
  - generates Slurm job scripts and manifest submit wrappers
- `scripts/stage_bio_series_runtime.py`
  - stages Bio-series split files into launcher-compatible runtime layout
- `scripts/summarize_bio_series_matrix.py`
  - produces flat run tables and trait-level summaries
- `scripts/compare_ablations.py`
  - compares each ablation against the `full` anchor

## Where the Experiment Scripts Are

Submission manifests for the repaired run are stored under:

```text
runs/df_gsf_v5_bio_series_20260403_leakfix/manifests/
```

Baseline manifests:

- `baseline_duroc.sh`
- `baseline_lwp1.sh`
- `baseline_lwp2.sh`

Ablation manifests:

- `ablation_duroc.sh`
- `ablation_lwp1.sh`
- `ablation_lwp2.sh`

The corresponding submission logs are in the same directory:

- `baseline_duroc.submit.log`
- `baseline_lwp1.submit.log`
- `baseline_lwp2.submit.log`
- `ablation_duroc.submit.log`
- `ablation_lwp1.submit.log`
- `ablation_lwp2.submit.log`

Generated per-job Slurm scripts and runtime logs are under:

```text
runs/df_gsf_v5_bio_series_20260403_leakfix/slurm_jobs/scripts_v5/
runs/df_gsf_v5_bio_series_20260403_leakfix/slurm_jobs/logs_v5/
```

## Where the Results Are

Per-run outputs are stored under:

```text
runs/df_gsf_v5_bio_series_20260403_leakfix/results/<dataset>_<trait>/<model>__<ablation>/<rep>/
```

Examples:

- `results/American_Duroc_LMA/bio_master_v11__full/rep_01/`
- `results/American_Duroc_LMA/bio_master_v11__no_delta/rep_01/`
- `results/LargeWhite_Pop2_BF_BF/bio_master_v11__no_pca/rep_10/`

Each run directory contains artifacts such as:

- `run_meta.json`
- `stats.json`
- `pred.csv`
- `best_model.pt`
- GWAS, PCA, GRM, and intermediate feature files

## Main Summary Files

Baseline summaries:

- `reports/final_baseline_summary/bio_series_run_table.csv`
- `reports/final_baseline_summary/bio_series_trait_summary.csv`
- `reports/baseline_group_summary.csv`

Ablation summaries:

- `summaries/ablation/ablation_compare_rep.csv`
- `summaries/ablation/ablation_compare_summary.csv`
- `summaries/ablation/ablation_group_summary.csv`
- `summaries/ablation/ablation_overall_summary.csv`

## Experiment Coverage

This repaired run completed:

- `110` baseline `full` runs
- `550` ablation runs
- `660` total runs

Ablation modes:

- `full`
- `no_delta`
- `no_gene2vec`
- `no_bio_prior`
- `no_pca`
- `pca_only_prior_off`

All six modes completed `110` runs each. There were no missing `stats.json` files in the final runtime tree.

## Baseline Results

Overall baseline mean PCC:

- `full`: `0.292553`

Group-level baseline mean PCC:

- `American_Duroc`: `0.256546`
- `Canadian_Duroc`: `0.145562`
- `LargeWhite_Pop1`: `0.359483`
- `LargeWhite_Pop1_Repro`: `0.265748`
- `LargeWhite_Pop2`: `0.524548`
- `LargeWhite_Pop2_BF`: `0.373115`

These values come from:

- `reports/baseline_group_summary.csv`

## Ablation Results

Overall mean PCC by ablation:

- `full`: `0.292553`
- `no_bio_prior`: `0.294359`
- `no_delta`: `0.295336`
- `no_gene2vec`: `0.293598`
- `no_pca`: `0.296426`
- `pca_only_prior_off`: `0.292959`

Relative to `full`, the overall `full_minus_this` values are:

- `no_bio_prior`: `-0.001805`
- `no_delta`: `-0.002783`
- `no_gene2vec`: `-0.001044`
- `no_pca`: `-0.003873`
- `pca_only_prior_off`: `-0.000405`

At the overall level, `full` is not the best-performing mode. The strongest overall mean PCC is from `no_pca`, followed by `no_delta`.

## Group-Level Ablation Patterns

Group-level comparisons are in:

- `summaries/ablation/ablation_group_summary.csv`

Main patterns:

- `LargeWhite_Pop2_BF` is the clearest group where `full` helps.
  - `full` outperforms every ablation in this group.
  - The largest drops come from `no_delta` (`full_minus_this = 0.009503`) and `no_bio_prior` (`0.008764`).
- `LargeWhite_Pop2` is mixed.
  - `full` is better than `no_pca` and `pca_only_prior_off`.
  - `full` is worse than `no_delta`, `no_bio_prior`, and `no_gene2vec`.
- `LargeWhite_Pop1` does not support the full model on average.
  - Every tested ablation is at least slightly better than `full`.
  - The largest gain comes from `no_delta` (`full_minus_this = -0.009852`).
- `LargeWhite_Pop1_Repro` also does not support the full model on average.
  - `no_pca` gives the largest improvement over `full` (`-0.006342`).
- `American_Duroc` is mixed.
  - `full` is slightly better than `no_gene2vec`, `no_bio_prior`, and `pca_only_prior_off`.
  - `no_delta` and `no_pca` are slightly better than `full`.
- `Canadian_Duroc` does not support the PCA/context component.
  - `no_pca` performs better than `full` by `0.008954`.

## Interpretation

The repaired experiment does not support the broad claim that the full biologically informed model is uniformly better than its module ablations.

The more accurate conclusion is:

- module contribution is strongly dataset- and trait-dependent
- some groups, especially `LargeWhite_Pop2_BF`, do show evidence that the biological modules help
- other groups show the opposite pattern, where removing one or more modules improves average PCC
- after removing evaluation leakage, the global average no longer supports `full` as the strongest configuration

This means the ablation result should be discussed as a heterogeneous contribution pattern rather than a single global positive effect.
