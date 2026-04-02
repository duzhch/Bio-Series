# DF_GSF_v5 Repository Standardization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Standardize `DF_GSF_v5` into a GitHub-ready mainline repository, preserve historical materials, replace hard-coded local paths with config-plus-environment overrides, and push the cleaned project to GitHub.

**Architecture:** Keep the existing runnable pipeline intact and layer repository cleanup around it. Use a small path-resolution helper to centralize config overrides, add public-safe example configs and ignore rules, refresh documentation to match the real CLI, and preserve older materials under clearer boundaries instead of deleting them.

**Tech Stack:** Python 3, YAML, pathlib, argparse, pytest, git, GitHub remote push

---

## File Map

### Primary files to modify

- `DF_GSF_v5/DF_GSF_v5.py`
- `DF_GSF_v5/submit_jobs.py`
- `DF_GSF_v5/README.md`
- `DF_GSF_v5/config/config.yaml`
- `DF_GSF_v5/config/global_config.yaml`
- `DF_GSF_v5/config/v11_config.yaml`
- `DF_GSF_v5/config/v12_config.yaml`
- `DF_GSF_v5/tests/test_launcher_ablation.py`

### Files to create

- `DF_GSF_v5/src/config_utils.py`
- `DF_GSF_v5/tests/test_config_utils.py`
- `DF_GSF_v5/config/examples/public_template.yaml`
- `DF_GSF_v5/.gitignore`
- `DF_GSF_v5/docs/project_structure.md`

### Files or directories to relocate or remove from versioned source

- `DF_GSF_v5/__pycache__/`
- `DF_GSF_v5/src/__pycache__/`
- `DF_GSF_v5/src/models/__pycache__/`
- `DF_GSF_v5/scripts/__pycache__/`
- `DF_GSF_v5/tests/__pycache__/`
- `DF_GSF_v5/tests/.tmp_bio_master_v11_verify/`
- `DF_GSF_v5/scripts/compare_rep.csv`
- `DF_GSF_v5/scripts/compare_summary.csv`
- `DF_GSF_v5/scripts/missing_report.csv`
- `DF_GSF_v5/https:/`

### Historical content to preserve and document

- `DF_GSF_v5/src/models/bio_master_v8.py`
- `DF_GSF_v5/src/models/bio_master_v9.py`
- `DF_GSF_v5/src/models/bio_master_v10.py`
- `DF_GSF_v5/src/models/bio_master_v13.py`
- `DF_GSF_v5/src/models/v5.py`
- `DF_GSF_v5/src/models/v9.py`
- `DF_GSF_v5/src/models/v10.py`
- `DF_GSF_v5/src/models/v11.py`
- `DF_GSF_v5/src/models/v12.py`
- `DF_GSF_v5/src/models/transformer_v1.py`
- `DF_GSF_v5/generate_cpu_test.sh`
- `DF_GSF_v5/jobs.sh`
- `DF_GSF_v5/submit_cpu_test.sh`
- `DF_GSF_v5/test_gpu.py`
- `DF_GSF_v5/thesis_materials/`
- `DF_GSF_v5/docs/superpowers/`

### Task 1: Add centralized path override support

**Files:**
- Create: `DF_GSF_v5/src/config_utils.py`
- Modify: `DF_GSF_v5/DF_GSF_v5.py`
- Modify: `DF_GSF_v5/submit_jobs.py`
- Test: `DF_GSF_v5/tests/test_config_utils.py`
- Test: `DF_GSF_v5/tests/test_launcher_ablation.py`

- [ ] **Step 1: Write the failing tests for config override behavior**

```python
from pathlib import Path

from src.config_utils import apply_env_overrides, get_resource_path


def test_apply_env_overrides_rewrites_machine_specific_fields(monkeypatch):
    cfg = {
        "exp_root": "/old/root",
        "resources": {
            "python_bin": "/old/python",
            "plink_bin": "plink",
            "gcta_bin": "gcta64",
            "reference_genome": "/old/ref.fa",
            "gtf_file": "/old/genes.gtf.gz",
            "pigbert_model": "/old/pigbert",
            "gene2vec_model": "/old/g2v.vec",
        },
        "datasets": {
            "pig": {
                "plink": "/old/pig/plink",
                "pheno": "/old/pig/pheno.csv",
            }
        },
    }

    monkeypatch.setenv("BIO_SERIES_EXP_ROOT", "/env/root")
    monkeypatch.setenv("BIO_SERIES_REFERENCE_GENOME", "/env/ref.fa")
    monkeypatch.setenv("BIO_SERIES_DATASET_PIG_PLINK", "/env/pig/plink")

    resolved = apply_env_overrides(cfg)

    assert resolved["exp_root"] == "/env/root"
    assert resolved["resources"]["reference_genome"] == "/env/ref.fa"
    assert resolved["datasets"]["pig"]["plink"] == "/env/pig/plink"
    assert resolved["datasets"]["pig"]["pheno"] == "/old/pig/pheno.csv"


def test_get_resource_path_falls_back_to_yaml_value():
    cfg = {"resources": {"plink_bin": "plink"}}
    assert get_resource_path(cfg, "plink_bin") == "plink"
```

```python
def test_load_cfg_applies_environment_overrides(monkeypatch, tmp_path):
    launcher = load_launcher_module(monkeypatch)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "exp_root: /yaml/root\n"
        "resources:\n"
        "  plink_bin: plink\n"
        "datasets:\n"
        "  pig:\n"
        "    plink: /yaml/plink\n"
        "    pheno: /yaml/pheno.csv\n"
    )

    monkeypatch.setenv("BIO_SERIES_EXP_ROOT", "/env/root")
    monkeypatch.setenv("BIO_SERIES_DATASET_PIG_PHENO", "/env/pheno.csv")

    cfg = launcher.load_cfg(str(cfg_path))

    assert cfg["exp_root"] == "/env/root"
    assert cfg["datasets"]["pig"]["pheno"] == "/env/pheno.csv"
    assert cfg["datasets"]["pig"]["plink"] == "/yaml/plink"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest DF_GSF_v5/tests/test_config_utils.py DF_GSF_v5/tests/test_launcher_ablation.py -k "config_utils or load_cfg_applies_environment_overrides" -v`

Expected: FAIL with import error or missing override behavior because `src/config_utils.py` does not exist yet and `load_cfg()` returns raw YAML.

- [ ] **Step 3: Write the minimal override helper and wire it into loaders**

```python
# DF_GSF_v5/src/config_utils.py
import copy
import os


RESOURCE_ENV_MAP = {
    "python_bin": "BIO_SERIES_PYTHON_BIN",
    "plink_bin": "BIO_SERIES_PLINK_BIN",
    "gcta_bin": "BIO_SERIES_GCTA_BIN",
    "reference_genome": "BIO_SERIES_REFERENCE_GENOME",
    "gtf_file": "BIO_SERIES_GTF_FILE",
    "pigbert_model": "BIO_SERIES_PIGBERT_MODEL",
    "gene2vec_model": "BIO_SERIES_GENE2VEC_MODEL",
}


def apply_env_overrides(cfg):
    resolved = copy.deepcopy(cfg)
    exp_root = os.getenv("BIO_SERIES_EXP_ROOT")
    if exp_root:
        resolved["exp_root"] = exp_root

    resources = resolved.setdefault("resources", {})
    for key, env_name in RESOURCE_ENV_MAP.items():
        value = os.getenv(env_name)
        if value:
            resources[key] = value

    datasets = resolved.get("datasets", {})
    for dataset_name, dataset_cfg in datasets.items():
        prefix = f"BIO_SERIES_DATASET_{dataset_name.upper()}_"
        plink = os.getenv(prefix + "PLINK")
        pheno = os.getenv(prefix + "PHENO")
        if plink:
            dataset_cfg["plink"] = plink
        if pheno:
            dataset_cfg["pheno"] = pheno

    return resolved


def get_resource_path(cfg, key):
    return cfg.get("resources", {}).get(key)
```

```python
# DF_GSF_v5/DF_GSF_v5.py
from src.config_utils import apply_env_overrides


def load_cfg(path: str):
    with open(path) as f:
        raw_cfg = yaml.safe_load(f)
    return apply_env_overrides(raw_cfg)
```

```python
# DF_GSF_v5/submit_jobs.py
from src.config_utils import apply_env_overrides


def load_cfg(path: str):
    with open(path) as f:
        raw_cfg = yaml.safe_load(f)
    return apply_env_overrides(raw_cfg)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest DF_GSF_v5/tests/test_config_utils.py DF_GSF_v5/tests/test_launcher_ablation.py -k "config_utils or load_cfg_applies_environment_overrides" -v`

Expected: PASS for the new override tests and no regression in the existing launcher path tests.

- [ ] **Step 5: Commit**

```bash
git add DF_GSF_v5/src/config_utils.py DF_GSF_v5/DF_GSF_v5.py DF_GSF_v5/submit_jobs.py DF_GSF_v5/tests/test_config_utils.py DF_GSF_v5/tests/test_launcher_ablation.py
git commit -m "refactor: centralize config path overrides"
```

### Task 2: Publish safe example configs and stop treating private absolute paths as public defaults

**Files:**
- Create: `DF_GSF_v5/config/examples/public_template.yaml`
- Modify: `DF_GSF_v5/config/config.yaml`
- Modify: `DF_GSF_v5/config/global_config.yaml`
- Modify: `DF_GSF_v5/config/v11_config.yaml`
- Modify: `DF_GSF_v5/config/v12_config.yaml`
- Test: `DF_GSF_v5/tests/test_config_utils.py`

- [ ] **Step 1: Write the failing tests for public template completeness**

```python
import yaml
from pathlib import Path


def test_public_template_includes_required_top_level_sections():
    cfg = yaml.safe_load(
        Path("DF_GSF_v5/config/examples/public_template.yaml").read_text()
    )
    assert "exp_root" in cfg
    assert "resources" in cfg
    assert "datasets" in cfg
    assert "experiment" in cfg
    assert "slurm" in cfg


def test_public_template_contains_no_lab_specific_absolute_home_paths():
    text = Path("DF_GSF_v5/config/examples/public_template.yaml").read_text()
    assert "/work/home/" not in text
    assert "zyqgroup" not in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest DF_GSF_v5/tests/test_config_utils.py -k "public_template" -v`

Expected: FAIL because the public template does not exist yet.

- [ ] **Step 3: Create a GitHub-safe template and mark local configs as machine-specific**

```yaml
# DF_GSF_v5/config/examples/public_template.yaml
exp_root: "/absolute/path/to/project_runtime_root"

resources:
  reference_genome: "/absolute/path/to/reference.fa"
  gtf_file: "/absolute/path/to/annotation.gtf.gz"
  pigbert_model: "/absolute/path/to/pigbert_model_dir"
  gene2vec_model: "/absolute/path/to/gene2vec.vector"
  plink_bin: "plink"
  gcta_bin: "gcta64"
  python_bin: "python"

experiment:
  replicates: 10
  top_n_snps: 3000
  lr: 3.0e-4
  batch_size: 64
  epochs: 150
  lambda_l1: 0.005

slurm:
  cpu_partition: "cpu"
  gpu_partition: "gpu"
  cpus_per_task: 8
  mem: "32G"
  gres: "gpu:1"
  time: "24:00:00"

datasets:
  example_population:
    plink: "/absolute/path/to/plink/prefix"
    pheno: "/absolute/path/to/phenotypes.tsv"
    traits: ["trait_1", "trait_2"]
```

```yaml
# DF_GSF_v5/config/v11_config.yaml
# Local lab config. Copy from config/examples/public_template.yaml when setting up a new machine.
```

Apply the same header pattern to:

- `DF_GSF_v5/config/config.yaml`
- `DF_GSF_v5/config/global_config.yaml`
- `DF_GSF_v5/config/v12_config.yaml`

Do not delete current local values from the machine-specific configs unless they block execution. Keep them as local-use configs, and make the public template the documented standard.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest DF_GSF_v5/tests/test_config_utils.py -k "public_template" -v`

Expected: PASS with the template present and free of private lab paths.

- [ ] **Step 5: Commit**

```bash
git add DF_GSF_v5/config/examples/public_template.yaml DF_GSF_v5/config/config.yaml DF_GSF_v5/config/global_config.yaml DF_GSF_v5/config/v11_config.yaml DF_GSF_v5/config/v12_config.yaml DF_GSF_v5/tests/test_config_utils.py
git commit -m "docs: add public config template"
```

### Task 3: Clean repository artifacts and formalize mainline versus archival content

**Files:**
- Create: `DF_GSF_v5/.gitignore`
- Create: `DF_GSF_v5/docs/project_structure.md`
- Modify: `DF_GSF_v5/README.md`

- [ ] **Step 1: Write the failing tests for repository hygiene markers**

```python
from pathlib import Path


def test_gitignore_covers_python_cache_and_results():
    text = Path("DF_GSF_v5/.gitignore").read_text()
    assert "__pycache__/" in text
    assert "*.pyc" in text
    assert "results/" in text
    assert "slurm_jobs/" in text
    assert "tests/.tmp_bio_master_v11_verify/" in text


def test_project_structure_doc_mentions_mainline_and_archival_content():
    text = Path("DF_GSF_v5/docs/project_structure.md").read_text()
    assert "Mainline" in text
    assert "Archived or supporting content" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest DF_GSF_v5/tests/test_config_utils.py -k "gitignore or project_structure" -v`

Expected: FAIL because neither file exists yet.

- [ ] **Step 3: Add ignore rules, remove generated junk, and document boundaries**

```gitignore
# DF_GSF_v5/.gitignore
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
results/
slurm_jobs/
tests/.tmp_bio_master_v11_verify/
scripts/*.csv
*.pt
```

```markdown
# DF_GSF_v5 Project Structure

## Mainline

- `DF_GSF_v5.py`: pipeline entry point
- `submit_jobs.py`: Slurm job generation
- `src/`: runnable source code
- `scripts/compare_ablations.py`: ablation result summarization
- `tests/`: regression and wiring tests
- `config/examples/public_template.yaml`: public-safe configuration template

## Archived or supporting content

- historical model variants in `src/models/`
- helper shell scripts kept for local operational history
- `thesis_materials/`: manuscript support notes
- `docs/superpowers/`: design and implementation records
```

Remove these from the working tree if present:

```bash
rm -rf DF_GSF_v5/__pycache__ DF_GSF_v5/src/__pycache__ DF_GSF_v5/src/models/__pycache__ DF_GSF_v5/scripts/__pycache__ DF_GSF_v5/tests/__pycache__ DF_GSF_v5/tests/.tmp_bio_master_v11_verify DF_GSF_v5/https:
rm -f DF_GSF_v5/scripts/compare_rep.csv DF_GSF_v5/scripts/compare_summary.csv DF_GSF_v5/scripts/missing_report.csv
```

Refresh the README section that explains which files are mainline and which are historical.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest DF_GSF_v5/tests/test_config_utils.py -k "gitignore or project_structure" -v`

Expected: PASS with the new metadata files present.

- [ ] **Step 5: Commit**

```bash
git add DF_GSF_v5/.gitignore DF_GSF_v5/docs/project_structure.md DF_GSF_v5/README.md
git add -u DF_GSF_v5
git commit -m "chore: clean repo artifacts and document structure"
```

### Task 4: Rewrite README around the actual CLI and ablation workflow

**Files:**
- Modify: `DF_GSF_v5/README.md`
- Modify: `DF_GSF_v5/docs/project_structure.md`

- [ ] **Step 1: Write the failing tests for README contract**

```python
from pathlib import Path


def test_readme_documents_real_run_all_command():
    text = Path("DF_GSF_v5/README.md").read_text()
    assert "run-all" in text
    assert "--config" in text
    assert "--dataset" in text
    assert "--trait" in text
    assert "--rep" in text
    assert "--model" in text


def test_readme_documents_ablation_modes():
    text = Path("DF_GSF_v5/README.md").read_text()
    assert "no_delta" in text
    assert "no_gene2vec" in text
    assert "no_bio_prior" in text
    assert "no_pca" in text
    assert "pca_only_prior_off" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest DF_GSF_v5/tests/test_config_utils.py -k "readme_documents" -v`

Expected: FAIL because the current README still shows outdated usage and does not fully document the current CLI contract.

- [ ] **Step 3: Rewrite the README to match the current repository**

```markdown
## Quick Start

1. Copy `config/examples/public_template.yaml` to a local config file.
2. Fill in dataset and resource paths, or provide them through environment variables.
3. Run a single experiment:

```bash
python DF_GSF_v5.py run-all \
  --config config/my_local.yaml \
  --dataset LargeWhite1 \
  --trait backfat \
  --rep rep_01 \
  --model bio_master_v11 \
  --ablation full
```

## Ablation Modes

- `full`
- `no_delta`
- `no_gene2vec`
- `no_bio_prior`
- `no_pca`
- `pca_only_prior_off`

## Batch Submission

```bash
python submit_jobs.py \
  --config config/my_local.yaml \
  --datasets LargeWhite1 \
  --traits backfat \
  --model bio_master_v11 \
  --ablations full,no_delta,no_pca \
  --out-sh submit_all.sh
```

## Compare Ablations

```bash
python scripts/compare_ablations.py \
  --root /path/to/experiment_root \
  --out /path/to/output_dir \
  --model bio_master_v11
```
```

Also add a short section listing:

- mainline files
- historical/supporting folders
- config template plus environment override variables

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest DF_GSF_v5/tests/test_config_utils.py -k "readme_documents" -v`

Expected: PASS with README synchronized to the real CLI.

- [ ] **Step 5: Commit**

```bash
git add DF_GSF_v5/README.md DF_GSF_v5/docs/project_structure.md DF_GSF_v5/tests/test_config_utils.py
git commit -m "docs: align readme with current pipeline"
```

### Task 5: Create a clean git history root and push the standardized mainline to GitHub

**Files:**
- Modify: `DF_GSF_v5/.gitignore`
- Modify: repository root metadata as needed during git initialization

- [ ] **Step 1: Write the failing verification checklist**

```text
Required before push:
- DF_GSF_v5 is the repository root
- no __pycache__ directories remain tracked
- no tests temp artifacts remain tracked
- public template exists
- README documents current CLI
- ablation code changes are included
```

- [ ] **Step 2: Run local repository status checks**

Run: `find DF_GSF_v5 -type d \\( -name __pycache__ -o -name .pytest_cache \\ )`

Expected: no output after cleanup.

Run: `find DF_GSF_v5/tests -maxdepth 2 -type d -name '.tmp_*'`

Expected: no output after cleanup.

Run: `git status --short`

Expected: only intended source, docs, config, and cleanup changes appear.

- [ ] **Step 3: Initialize or stage the clean repository root and commit**

If `DF_GSF_v5` is not already a git repository:

```bash
cd DF_GSF_v5
git init
git branch -M main
git add .
git commit -m "feat: add ablation workflow and standardize repository"
```

If using a clean upload workspace cloned from GitHub:

```bash
rsync -a --delete DF_GSF_v5/ /tmp/bio-series-upload/repo/
git -C /tmp/bio-series-upload/repo status --short
git -C /tmp/bio-series-upload/repo add .
git -C /tmp/bio-series-upload/repo commit -m "feat: add ablation workflow and standardize repository"
```

- [ ] **Step 4: Push to the GitHub remote**

Run:

```bash
GIT_TERMINAL_PROMPT=0 git -C /tmp/bio-series-upload/repo push https://x-access-token:${GITHUB_TOKEN}@github.com/duzhch/Bio-Series.git main:main
```

Expected: push completes successfully and the remote branch reflects the cleaned `DF_GSF_v5` mainline repository contents.

- [ ] **Step 5: Final verification and commit note**

Run: `git -C /tmp/bio-series-upload/repo rev-parse --short HEAD`

Expected: print the pushed commit SHA for reporting back to the user.

Document any skipped runtime checks explicitly because this compute node is not the acceptance environment.

```bash
git -C /tmp/bio-series-upload/repo log -1 --stat
```

Expected: show the final cleanup-and-feature commit summary for the user report.

## Self-Review

- Spec coverage check:
  - repository mainline clarity is covered by Tasks 3 and 4
  - path de-hardcoding is covered by Tasks 1 and 2
  - GitHub-safe public config is covered by Task 2
  - generated junk cleanup is covered by Task 3
  - git commit and GitHub push are covered by Task 5
  - preservation of historical materials is covered by Tasks 3 and 4
- Placeholder scan:
  - no `TODO`, `TBD`, or abstract “handle appropriately” steps remain
  - every code-producing task contains concrete file paths and example content
- Type and naming consistency:
  - `apply_env_overrides()` is the central config helper throughout
  - `public_template.yaml` is the single named public template throughout
  - environment variable prefix uses `BIO_SERIES_` consistently
