# DF_GSF_v5

`DF_GSF_v5` is a genomic prediction pipeline centered on biologically informed wide-and-deep models for pig breeding traits. The current mainline combines GWAS-based SNP selection, sequence and gene-function priors, genotype extraction, deep model training, and ablation comparison.

## Mainline entry points

- `DF_GSF_v5.py`: single-run launcher
- `submit_jobs.py`: Slurm batch script generator
- `src/`: source code for GWAS, feature extraction, data handling, and models
- `scripts/compare_ablations.py`: summarize ablation outputs against the `full` anchor
- `tests/`: launcher and model wiring tests
- `config/examples/public_template.yaml`: GitHub-safe config template

For a concise map of runnable versus archival content, see [docs/project_structure.md](docs/project_structure.md).

## Repository layout

The repository keeps older model variants, helper shell scripts, and thesis materials for traceability, but the active execution path is the mainline listed above. Historical and supporting materials are preserved rather than deleted.

## Configuration

Start from the public template:

```bash
cp config/examples/public_template.yaml config/my_local.yaml
```

Fill in local absolute paths in `config/my_local.yaml`, or override machine-specific paths with environment variables.

Supported environment overrides:

- `BIO_SERIES_EXP_ROOT`
- `BIO_SERIES_PYTHON_BIN`
- `BIO_SERIES_PLINK_BIN`
- `BIO_SERIES_GCTA_BIN`
- `BIO_SERIES_REFERENCE_GENOME`
- `BIO_SERIES_GTF_FILE`
- `BIO_SERIES_PIGBERT_MODEL`
- `BIO_SERIES_GENE2VEC_MODEL`
- `BIO_SERIES_DATASET_<DATASET>_PLINK`
- `BIO_SERIES_DATASET_<DATASET>_PHENO`

For dataset-specific variables, `<DATASET>` is the dataset key converted to uppercase with non-alphanumeric characters replaced by underscores.

The files under `config/` such as `v11_config.yaml` are retained as local lab configs. They are not the public portability baseline.

## Single experiment

Run the full pipeline with explicit arguments:

```bash
python DF_GSF_v5.py run-all \
  --config config/my_local.yaml \
  --dataset LargeWhite_Pop1 \
  --trait BF \
  --rep rep_01 \
  --model bio_master_v11 \
  --ablation full
```

The launcher also supports `run-gwas` for isolated GWAS and PCA generation.

## Ablation modes

Current biological module ablations for `bio_master_v11`:

- `full`
- `no_delta`
- `no_gene2vec`
- `no_bio_prior`
- `no_pca`
- `pca_only_prior_off`

Results are written under:

```text
results/<dataset>_<trait>/<model>__<ablation>/<rep>/
```

Each run writes a `run_meta.json` describing the ablation semantics and resolved paths for the generated artifacts.

## Batch submission

Generate Slurm jobs for one or more traits and ablations:

```bash
python submit_jobs.py \
  --config config/my_local.yaml \
  --datasets LargeWhite_Pop1 \
  --traits BF \
  --model bio_master_v11 \
  --ablations full,no_delta,no_pca \
  --out-sh submit_all.sh
```

Optional flags:

- `--cpu-only`
- `--wait`
- `--wait-interval`
- `--max-wait-hours`

## Compare ablations

Summarize per-replicate and aggregated ablation results:

```bash
python scripts/compare_ablations.py \
  --root /path/to/experiment_root \
  --out /path/to/summary_dir \
  --model bio_master_v11
```

Outputs:

- `ablation_compare_rep.csv`
- `ablation_compare_summary.csv`

## Dependencies

The code expects a Python environment with at least:

- `PyYAML`
- `numpy`
- `pandas`
- `torch`
- `scikit-learn`
- `scipy`
- `pandas_plink`

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

Feature extraction and GWAS stages also depend on external tools and resources configured in YAML:

- `plink`
- `gcta`
- reference genome FASTA
- GTF annotation
- PigBERT model
- Gene2Vec model

## Historical and supporting content

The repository intentionally keeps:

- older `src/models/` variants
- shell helpers such as `jobs.sh`, `generate_cpu_test.sh`, and `submit_cpu_test.sh`
- `scripts/compare_v5_vs_baselines.py`
- `thesis_materials/`
- `docs/superpowers/`

These materials are preserved for reproducibility and documentation history, but they are not the recommended starting point for new runs.
