# DF_GSF_v5 Project Structure

## Mainline

- `DF_GSF_v5.py`: pipeline entry point for GWAS, feature extraction, genotype extraction, and model training
- `submit_jobs.py`: Slurm job script generation and optional submission helper
- `src/`: runnable source code for data processing, feature engineering, GWAS, and models
- `scripts/compare_ablations.py`: ablation result summarization utility
- `tests/`: regression and wiring tests for launcher and ablation behavior
- `config/examples/public_template.yaml`: GitHub-safe configuration template

## Archived or supporting content

- historical model variants in `src/models/`
- helper shell scripts kept for operational history, including `jobs.sh`, `generate_cpu_test.sh`, and `submit_cpu_test.sh`
- `scripts/compare_v5_vs_baselines.py`: older comparison utility kept for reference
- `thesis_materials/`: manuscript and thesis support notes
- `docs/superpowers/`: design and implementation records produced during development
