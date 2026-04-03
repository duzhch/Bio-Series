# DF_GSF_v5 Bio-Series Baseline And Ablation Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run `DF_GSF_v5` against the curated `Bio-series-data` datasets for the full-model baseline and biologically motivated module ablations, using Slurm batch jobs on compute nodes, with standardized storage for manifests, job scripts, logs, intermediate artifacts, and summaries.

**Architecture:** Keep the current `DF_GSF_v5` training and ablation logic intact. Bridge `Bio-series-data` into the launcher's expected runtime layout with a lightweight split-staging layer instead of rewriting the pipeline, add one cluster-specific config for the real dataset paths, and standardize Slurm generation so every job sources `~/.bashrc`, activates `conda` env `NT`, and writes into a dedicated runtime root.

**Tech Stack:** Python 3, YAML, pathlib, pandas, Slurm (`sbatch`, `squeue`, `sacct`, `pestat`), bash, symlinks, PyTorch, PLINK, GCTA

---

## File Map

### Existing files to modify

- `DF_GSF_v5/submit_jobs.py`
- `DF_GSF_v5/README.md`
- `DF_GSF_v5/config/config.yaml`

### New files to create

- `DF_GSF_v5/config/bio_series_data.cluster.yaml`
- `DF_GSF_v5/scripts/stage_bio_series_runtime.py`
- `DF_GSF_v5/scripts/summarize_bio_series_matrix.py`
- `DF_GSF_v5/tests/test_stage_bio_series_runtime.py`
- `DF_GSF_v5/docs/experiment_layout.md`

### Runtime directories to create outside versioned source

- `/work/home/zyqlab/dzhichao/zyqgroup02_duanzhichao/exp_00/experiments/01_benchmarks/runs/df_gsf_v5_bio_series_20260402/`
- `/work/home/zyqlab/dzhichao/zyqgroup02_duanzhichao/exp_00/experiments/01_benchmarks/runs/df_gsf_v5_bio_series_20260402/data/splits/`
- `/work/home/zyqlab/dzhichao/zyqgroup02_duanzhichao/exp_00/experiments/01_benchmarks/runs/df_gsf_v5_bio_series_20260402/manifests/`
- `/work/home/zyqlab/dzhichao/zyqgroup02_duanzhichao/exp_00/experiments/01_benchmarks/runs/df_gsf_v5_bio_series_20260402/slurm_jobs/scripts_v5/`
- `/work/home/zyqlab/dzhichao/zyqgroup02_duanzhichao/exp_00/experiments/01_benchmarks/runs/df_gsf_v5_bio_series_20260402/slurm_jobs/logs_v5/`
- `/work/home/zyqlab/dzhichao/zyqgroup02_duanzhichao/exp_00/experiments/01_benchmarks/runs/df_gsf_v5_bio_series_20260402/results/`
- `/work/home/zyqlab/dzhichao/zyqgroup02_duanzhichao/exp_00/experiments/01_benchmarks/runs/df_gsf_v5_bio_series_20260402/summaries/`

### Data contract confirmed from `Bio-series-data`

- Genotype inputs come from `01_processed_data/*/*_QCed.{bed,bim,fam}`
- Phenotype inputs come from `01_processed_data/*/aligned_*.csv`
- Cross-validation inputs come from `02_cv_datasets/<dataset>/<trait>/fold_<1..10>/train_ids.txt` and `test_ids.txt`
- CV generation uses `KFold(n_splits=10, shuffle=True, random_state=42)`
- Deep-learning runs should consume the provided `train_ids.txt` and `test_ids.txt` directly

### Dataset mapping to standardize

- `Amercian_duroci` genotype/pheno directory pairs with CV dataset key `Amercian_duroc`
- `Canadian_duroic` genotype/pheno directory pairs with CV dataset key `Canadian_duroc`
- `lwp1` maps to `LargeWhite_Pop1`
- `lwp2` maps to `LargeWhite_Pop2`

### Target experiment matrix

- `American_Duroc`: `LMA`, `LMD`
- `Canadian_Duroc`: `LMA`, `LMD`
- `LargeWhite_Pop1`: `AGE`, `BF`, `LW`, `NBA`, `TNB`
- `LargeWhite_Pop2`: `AGE`, `BF`
- 10 folds each
- Ablations: `full`, `no_delta`, `no_gene2vec`, `no_bio_prior`, `no_pca`, `pca_only_prior_off`

Total planned runs after smoke verification:

- Baseline only: `11 trait tasks * 10 folds = 110 runs`
- Baseline plus 5 ablations: `11 trait tasks * 10 folds * 6 modes = 660 runs`

## Task 1: Add a runtime staging bridge for `Bio-series-data`

**Files:**

- Create: `DF_GSF_v5/scripts/stage_bio_series_runtime.py`
- Create: `DF_GSF_v5/tests/test_stage_bio_series_runtime.py`
- Create: `DF_GSF_v5/config/bio_series_data.cluster.yaml`

- [ ] **Step 1: Write the failing test for split staging**

```python
from pathlib import Path

from scripts.stage_bio_series_runtime import stage_runtime


def test_stage_runtime_creates_launcher_compatible_split_tree(tmp_path):
    cv_root = tmp_path / "02_cv_datasets"
    fold_dir = cv_root / "Amercian_duroc" / "LMA" / "fold_1"
    fold_dir.mkdir(parents=True)
    (fold_dir / "train_ids.txt").write_text("1\t1\n2\t2\n")
    (fold_dir / "test_ids.txt").write_text("3\t3\n")

    runtime_root = tmp_path / "runtime"
    stage_runtime(cv_root=cv_root, runtime_root=runtime_root, dataset_map={
        "American_Duroc": {
            "cv_name": "Amercian_duroc",
            "traits": ["LMA"],
        }
    })

    train_ids = runtime_root / "data" / "splits" / "American_Duroc_LMA" / "rep_01" / "train.ids"
    test_ids = runtime_root / "data" / "splits" / "American_Duroc_LMA" / "rep_01" / "test.ids"

    assert train_ids.exists()
    assert test_ids.exists()
    assert train_ids.read_text() == "1\t1\n2\t2\n"
    assert test_ids.read_text() == "3\t3\n"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest DF_GSF_v5/tests/test_stage_bio_series_runtime.py -v`

Expected: FAIL because `scripts/stage_bio_series_runtime.py` does not exist yet.

- [ ] **Step 3: Implement minimal split staging**

```python
from pathlib import Path
import shutil


def stage_runtime(cv_root, runtime_root, dataset_map):
    cv_root = Path(cv_root)
    runtime_root = Path(runtime_root)

    for dataset_key, info in dataset_map.items():
        cv_name = info["cv_name"]
        for trait in info["traits"]:
            for fold in range(1, 11):
                src_dir = cv_root / cv_name / trait / f"fold_{fold}"
                dst_dir = runtime_root / "data" / "splits" / f"{dataset_key}_{trait}" / f"rep_{fold:02d}"
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_dir / "train_ids.txt", dst_dir / "train.ids")
                shutil.copy2(src_dir / "test_ids.txt", dst_dir / "test.ids")
```

- [ ] **Step 4: Add the cluster config with real dataset paths**

```yaml
exp_root: "/work/home/zyqlab/dzhichao/zyqgroup02_duanzhichao/exp_00/experiments/01_benchmarks/runs/df_gsf_v5_bio_series_20260402"

resources:
  reference_genome: "/work/home/zyqgroup02/duanzhichao/exp_00/data/references/Sus_scrofa.Sscrofa11.1.dna.toplevel.fa"
  gtf_file: "/work/home/zyqgroup02/duanzhichao/exp_00/data/annotations/Sus_scrofa.Sscrofa11.1.111.gtf.gz"
  pigbert_model: "/work/home/zyqgroup02/duanzhichao/exp_00/models/pig_bert2_v1"
  gene2vec_model: "/work/home/zyqgroup02/duanzhichao/data/pigGTEx/pig_gene2vec_v2/output/models_v2/pig_g2v_id8_sz300_neg5_sg1.vector"
  plink_bin: "plink"
  gcta_bin: "gcta64"
  python_bin: "python"

slurm:
  cpu_partition: "XiaoQingHe"
  gpu_partition: "JuMaHe"
  cpus_per_task: 8
  mem: "64G"
  gres: "gpu:1"
  time: "24:00:00"

experiment:
  replicates: 10
  top_n_snps: 3000
  batch_size: 64
  epochs: 150
  lr: 5.0e-4
  lambda_l1: 0.005

datasets:
  American_Duroc:
    plink: "/work/home/zyqlab/dzhichao/zyqgroup02_duanzhichao/exp_00/experiments/01_benchmarks/Bio-series-data/01_processed_data/Amercian_duroci/Amercian_duroic_QCed"
    pheno: "/work/home/zyqlab/dzhichao/zyqgroup02_duanzhichao/exp_00/experiments/01_benchmarks/Bio-series-data/01_processed_data/Amercian_duroci/aligned_phenotype.csv"
    id_col: "ID"
    traits: ["LMA", "LMD"]
    sep: ","

  Canadian_Duroc:
    plink: "/work/home/zyqlab/dzhichao/zyqgroup02_duanzhichao/exp_00/experiments/01_benchmarks/Bio-series-data/01_processed_data/Canadian_duroic/Canadian_duroic_QCed"
    pheno: "/work/home/zyqlab/dzhichao/zyqgroup02_duanzhichao/exp_00/experiments/01_benchmarks/Bio-series-data/01_processed_data/Canadian_duroic/aligned_phenotype.csv"
    id_col: "ID"
    traits: ["LMA", "LMD"]
    sep: ","

  LargeWhite_Pop1:
    plink: "/work/home/zyqlab/dzhichao/zyqgroup02_duanzhichao/exp_00/experiments/01_benchmarks/Bio-series-data/01_processed_data/lwp1/lwp1_QCed"
    pheno: "/work/home/zyqlab/dzhichao/zyqgroup02_duanzhichao/exp_00/experiments/01_benchmarks/Bio-series-data/01_processed_data/lwp1/aligned_Growth_phenotype.csv"
    id_col: "ID"
    traits: ["AGE", "BF"]
    sep: ","

  LargeWhite_Pop1_Repro:
    plink: "/work/home/zyqlab/dzhichao/zyqgroup02_duanzhichao/exp_00/experiments/01_benchmarks/Bio-series-data/01_processed_data/lwp1/lwp1_QCed"
    pheno: "/work/home/zyqlab/dzhichao/zyqgroup02_duanzhichao/exp_00/experiments/01_benchmarks/Bio-series-data/01_processed_data/lwp1/aligned_Reproductive_phenotype.csv"
    id_col: "ID"
    traits: ["LW", "NBA", "TNB"]
    sep: ","

  LargeWhite_Pop2:
    plink: "/work/home/zyqlab/dzhichao/zyqgroup02_duanzhichao/exp_00/experiments/01_benchmarks/Bio-series-data/01_processed_data/lwp2/lwp2_QCed"
    pheno: "/work/home/zyqlab/dzhichao/zyqgroup02_duanzhichao/exp_00/experiments/01_benchmarks/Bio-series-data/01_processed_data/lwp2/aligned_age.csv"
    id_col: "ID"
    traits: ["AGE"]
    sep: ","

  LargeWhite_Pop2_BF:
    plink: "/work/home/zyqlab/dzhichao/zyqgroup02_duanzhichao/exp_00/experiments/01_benchmarks/Bio-series-data/01_processed_data/lwp2/lwp2_QCed"
    pheno: "/work/home/zyqlab/dzhichao/zyqgroup02_duanzhichao/exp_00/experiments/01_benchmarks/Bio-series-data/01_processed_data/lwp2/aligned_BF.csv"
    id_col: "ID"
    traits: ["BF"]
    sep: ","
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `python -m pytest DF_GSF_v5/tests/test_stage_bio_series_runtime.py -v`

Expected: PASS

## Task 2: Make Slurm jobs cluster-correct and environment-stable

**Files:**

- Modify: `DF_GSF_v5/submit_jobs.py`
- Modify: `DF_GSF_v5/README.md`
- Create: `DF_GSF_v5/docs/experiment_layout.md`

- [ ] **Step 1: Add bootstrap lines to every generated job**

```python
body = [
    "set -euo pipefail",
    "source ~/.bashrc",
    "conda activate NT",
    f"cd {code_dir}",
    f"{cfg['resources']['python_bin']} DF_GSF_v5.py run-all "
    f"--config {config_path} "
    f"--dataset {ds} --trait {trait} --rep {rep} "
    f"--model {args.model} --ablation {ablation}",
]
```

- [ ] **Step 2: Preserve standardized job and log placement**

Use these directories only:

```text
<exp_root>/slurm_jobs/scripts_v5/
<exp_root>/slurm_jobs/logs_v5/
<exp_root>/results/
<exp_root>/summaries/
```

- [ ] **Step 3: Document the runtime layout**

```markdown
# Experiment Layout

- `data/splits/`: staged fold-wise train/test ids in launcher-compatible layout
- `slurm_jobs/scripts_v5/`: generated `sbatch` scripts
- `slurm_jobs/logs_v5/`: `.out` and `.err` files
- `results/<dataset>_<trait>/<model>__<ablation>/<rep>/`: run artifacts
- `summaries/ablation/`: aggregate CSV summaries from `compare_ablations.py`
```

- [ ] **Step 4: Verify generated scripts contain the required cluster bootstrap**

Run:

```bash
python submit_jobs.py \
  --config config/bio_series_data.cluster.yaml \
  --datasets American_Duroc \
  --traits LMA \
  --model bio_master_v11 \
  --ablations full \
  --reps 1 \
  --out-sh /tmp/check_submit.sh
```

Expected generated job body contains:

- `source ~/.bashrc`
- `conda activate NT`
- `python DF_GSF_v5.py run-all ...`

## Task 3: Stage the experiment matrix in controlled batches

**Files:**

- Create: `DF_GSF_v5/scripts/summarize_bio_series_matrix.py`
- Reuse: `DF_GSF_v5/scripts/compare_ablations.py`

- [ ] **Step 1: Create a smoke batch before large-scale submission**

Smoke set:

```text
American_Duroc/LMA/fold_1/full
Canadian_Duroc/LMD/fold_1/full
LargeWhite_Pop1/BF/fold_1/full
LargeWhite_Pop1_Repro/LW/fold_1/full
LargeWhite_Pop2/AGE/fold_1/full
LargeWhite_Pop2_BF/BF/fold_1/full
```

Run:

```bash
python scripts/stage_bio_series_runtime.py \
  --cv-root /work/home/zyqlab/dzhichao/zyqgroup02_duanzhichao/exp_00/experiments/01_benchmarks/Bio-series-data/02_cv_datasets \
  --runtime-root /work/home/zyqlab/dzhichao/zyqgroup02_duanzhichao/exp_00/experiments/01_benchmarks/runs/df_gsf_v5_bio_series_20260402
```

Then generate smoke jobs:

```bash
python submit_jobs.py \
  --config config/bio_series_data.cluster.yaml \
  --datasets American_Duroc,Canadian_Duroc,LargeWhite_Pop1,LargeWhite_Pop1_Repro,LargeWhite_Pop2,LargeWhite_Pop2_BF \
  --traits LMA,LMD,BF,LW,AGE \
  --model bio_master_v11 \
  --ablations full \
  --reps 1 \
  --out-sh /work/home/zyqlab/dzhichao/zyqgroup02_duanzhichao/exp_00/experiments/01_benchmarks/runs/df_gsf_v5_bio_series_20260402/manifests/smoke_full.sh
```

- [ ] **Step 2: Submit smoke jobs to GPU compute nodes only**

Run:

```bash
sbatch <generated_job_script>
```

Inspection commands:

```bash
squeue -u dzhichao
scontrol show job <jobid>
sacct -j <jobid>
```

Expected:

- No training runs launched on login nodes
- Jobs run in `JuMaHe` by default
- Each job requests `--nodes=1`

- [ ] **Step 3: Run the full baseline matrix after smoke passes**

Full baseline batches:

- Batch A: `American_Duroc`, `Canadian_Duroc`
- Batch B: `LargeWhite_Pop1`, `LargeWhite_Pop1_Repro`
- Batch C: `LargeWhite_Pop2`, `LargeWhite_Pop2_BF`

Run one batch at a time:

```bash
python submit_jobs.py \
  --config config/bio_series_data.cluster.yaml \
  --datasets American_Duroc,Canadian_Duroc \
  --traits LMA,LMD \
  --model bio_master_v11 \
  --ablations full \
  --reps 10 \
  --out-sh /work/home/zyqlab/dzhichao/zyqgroup02_duanzhichao/exp_00/experiments/01_benchmarks/runs/df_gsf_v5_bio_series_20260402/manifests/baseline_duroc.sh
```

- [ ] **Step 4: Run ablations only after the `full` anchor is complete for the same dataset group**

Run:

```bash
python submit_jobs.py \
  --config config/bio_series_data.cluster.yaml \
  --datasets American_Duroc,Canadian_Duroc \
  --traits LMA,LMD \
  --model bio_master_v11 \
  --ablations no_delta,no_gene2vec,no_bio_prior,no_pca,pca_only_prior_off \
  --reps 10 \
  --out-sh /work/home/zyqlab/dzhichao/zyqgroup02_duanzhichao/exp_00/experiments/01_benchmarks/runs/df_gsf_v5_bio_series_20260402/manifests/ablation_duroc.sh
```

Reasoning:

- `full` is the baseline anchor
- ablation deltas should be interpreted against the already-completed `full` runs
- this reduces wasted queue usage if a dataset-path issue appears early

## Task 4: Summarize and validate outputs

**Files:**

- Reuse: `DF_GSF_v5/scripts/compare_ablations.py`
- Create: `DF_GSF_v5/scripts/summarize_bio_series_matrix.py`

- [ ] **Step 1: Validate per-run output completeness**

For each run directory verify:

- `run_meta.json`
- `pred.csv`
- `stats.json`
- `best_model.pt`

- [ ] **Step 2: Aggregate ablation summaries by dataset group**

Run:

```bash
python scripts/compare_ablations.py \
  --root /work/home/zyqlab/dzhichao/zyqgroup02_duanzhichao/exp_00/experiments/01_benchmarks/runs/df_gsf_v5_bio_series_20260402 \
  --out /work/home/zyqlab/dzhichao/zyqgroup02_duanzhichao/exp_00/experiments/01_benchmarks/runs/df_gsf_v5_bio_series_20260402/summaries/ablation \
  --model bio_master_v11
```

- [ ] **Step 3: Build a matrix-level summary table**

`scripts/summarize_bio_series_matrix.py` should merge:

- dataset key
- trait
- fold
- ablation
- `pcc_test`
- `mse`
- `sparsity`

Output:

```text
summaries/matrix/bio_series_run_table.csv
summaries/matrix/bio_series_trait_summary.csv
```

- [ ] **Step 4: Record final interpretation constraints**

Required notes in final report:

- `Amercian_duroc` and `Canadian_duroc` source directories contain spelling inconsistencies and were normalized only at the config/runtime layer
- `LargeWhite_Pop1` and `LargeWhite_Pop2` use separate phenotype files for different trait groups, so identical genotype prefixes are reused across multiple dataset keys
- conclusions about module contribution must compare the same dataset, same trait, same fold, and same seed-controlled split
- if any ablation shows improvement over `full`, interpret that as a signal about regularization or prior mismatch rather than automatically as evidence the removed module is harmful

## Queue And Partition Guidance

- Use `sbatch`, not direct training on login nodes `103/104/105`
- Default to `--nodes=1`
- Prefer `JuMaHe` first for full pipeline runs because current snapshot showed idle GPU capacity
- Keep `XiaoQingHe` as CPU fallback for preprocessing-only debugging
- Use `pestat` before bulk submission and `squeue -u dzhichao` during monitoring

## Recommended Execution Order

1. Implement Task 1 and Task 2
2. Stage splits and generate smoke jobs
3. Submit smoke jobs and inspect logs
4. Run all `full` baseline jobs by dataset group
5. Run ablations by dataset group
6. Aggregate summaries and write the interpretation report

## Risks To Watch

- Current local `DF_GSF_v5` directory is not itself a `.git` worktree, so confirm the exact execution copy before modifying production files
- `submit_jobs.py` currently does not source `~/.bashrc` or activate `NT`, so jobs may fail until Task 2 is implemented
- `DF_GSF_v5.py` expects staged split files named `train.ids` and `test.ids`; the raw data uses `train_ids.txt` and `test_ids.txt`
- `LargeWhite_Pop1` and `LargeWhite_Pop2` traits are split across multiple phenotype files, so a single dataset key cannot represent all traits without either extra dataset aliases or pheno switching

Plan complete and saved to `docs/superpowers/plans/2026-04-02-df-gsf-v5-bio-series-baseline-ablation-execution.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints
