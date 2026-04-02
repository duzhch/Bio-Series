# DF_GSF_v5 Biological Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a runtime ablation mechanism to `DF_GSF_v5` so `bio_master_v11` can run biologically meaningful module ablations without a large rewrite.

**Architecture:** Keep the existing launcher, training flow, and model file. Add one new runtime control surface, thread it into `bio_master_v11`, and add focused tests plus a dedicated ablation summary script. Preserve current behavior when `--ablation` is omitted or set to `full`.

**Tech Stack:** Python 3, argparse, PyTorch, pandas, numpy, pytest

---

## File Map

**Create:**

- `DF_GSF_v5/tests/test_launcher_ablation.py`
- `DF_GSF_v5/tests/test_bio_master_v11_ablation.py`
- `DF_GSF_v5/scripts/compare_ablations.py`

**Modify:**

- `DF_GSF_v5/DF_GSF_v5.py`
- `DF_GSF_v5/src/models/bio_master_v11.py`
- `DF_GSF_v5/submit_jobs.py`

## Task 1: Add launcher-level ablation support

**Files:**

- Modify: `DF_GSF_v5/DF_GSF_v5.py`
- Test: `DF_GSF_v5/tests/test_launcher_ablation.py`

- [ ] **Step 1: Write the failing launcher tests**

```python
from pathlib import Path

import DF_GSF_v5 as launcher


def test_infer_paths_includes_ablation_folder():
    cfg = {"exp_root": "/tmp/exp-root"}
    out_dir, train_ids, test_ids = launcher.infer_paths(
        cfg,
        dataset="LargeWhite_Pop1",
        trait="BF",
        rep="rep_01",
        model_name="bio_master_v11",
        ablation="no_delta",
    )

    assert str(out_dir).endswith("results/LargeWhite_Pop1_BF/bio_master_v11__no_delta/rep_01")
    assert str(train_ids).endswith("data/splits/LargeWhite_Pop1_BF/rep_01/train.ids")
    assert str(test_ids).endswith("data/splits/LargeWhite_Pop1_BF/rep_01/test.ids")


def test_step_run_all_passes_ablation_to_trainer(tmp_path, monkeypatch):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        \"\"\"
exp_root: "{root}"
resources:
  reference_genome: "ref.fa"
  pigbert_model: "pigbert"
  gtf_file: "genes.gtf.gz"
  gene2vec_model: "gene2vec.vec"
  plink_bin: "plink"
  gcta_bin: "gcta64"
experiment:
  top_n_snps: 10
  lr: 0.001
  batch_size: 8
  epochs: 2
  lambda_l1: 0.01
datasets:
  Demo:
    plink: "demo_plink"
    pheno: "demo.csv"
\"\"\".format(root=tmp_path)
    )

    calls = {}

    monkeypatch.setattr(launcher, "run_gwas_pipeline", lambda **kwargs: {
        "snps_for_emb": str(tmp_path / "snps.csv"),
        "plink_snps": str(tmp_path / "plink_snps.txt"),
    })
    monkeypatch.setattr(launcher, "extract_delta", lambda *args, **kwargs: None)
    monkeypatch.setattr(launcher, "annotate_snps_with_gtf", lambda *args, **kwargs: None)
    monkeypatch.setattr(launcher, "extract_gene2vec", lambda *args, **kwargs: None)
    monkeypatch.setattr(launcher, "make_all_ids", lambda *args, **kwargs: None)
    monkeypatch.setattr(launcher, "plink_extract", lambda *args, **kwargs: None)

    def fake_train(**kwargs):
        calls.update(kwargs)
        return {"pcc": 0.1, "mse": 1.0}

    monkeypatch.setattr(launcher, "get_model_trainer", lambda model_name: fake_train)

    args = type("Args", (), {
        "config": str(cfg_path),
        "dataset": "Demo",
        "trait": "BF",
        "rep": "rep_01",
        "model": "bio_master_v11",
        "ablation": "no_delta",
    })()

    launcher.step_run_all(args)

    assert calls["ablation"] == "no_delta"
    assert calls["out_dir"].endswith("bio_master_v11__no_delta/rep_01")
```

- [ ] **Step 2: Run launcher tests to verify they fail**

Run:

```bash
pytest DF_GSF_v5/tests/test_launcher_ablation.py -q
```

Expected:

- FAIL because `infer_paths()` does not yet accept `ablation`
- FAIL because `step_run_all()` does not yet pass `ablation` to the trainer

- [ ] **Step 3: Implement minimal launcher changes**

```python
VALID_ABLATIONS = {
    "full",
    "no_delta",
    "no_gene2vec",
    "no_bio_prior",
    "no_pca",
    "pca_only_prior_off",
}


def normalize_ablation(ablation: str) -> str:
    ablation = (ablation or "full").strip()
    if ablation not in VALID_ABLATIONS:
        raise ValueError(f"Unsupported ablation: {ablation}")
    return ablation


def infer_paths(cfg, dataset, trait, rep, model_name, ablation="full"):
    exp_root = Path(cfg["exp_root"])
    ablation = normalize_ablation(ablation)
    folder_name = f"{model_name}__{ablation}"
    out_dir = exp_root / f"results/{dataset}_{trait}/{folder_name}/{rep}"
    split_dir = exp_root / f"data/splits/{dataset}_{trait}/{rep}"
    return out_dir, split_dir / "train.ids", split_dir / "test.ids"
```

```python
common.add_argument("--ablation", default="full")
```

```python
train_fn(
    plink_prefix=str(fusion_prefix),
    pheno_file=ds["pheno"],
    train_ids=str(train_ids),
    test_ids=str(test_ids),
    trait=args.trait,
    delta_path=str(delta_path),
    gene_path=str(gene_path),
    out_dir=str(out_dir),
    lr=cfg["experiment"]["lr"],
    batch_size=cfg["experiment"]["batch_size"],
    epochs=cfg["experiment"]["epochs"],
    lambda_l1=cfg["experiment"]["lambda_l1"],
    device="auto",
    ablation=normalize_ablation(args.ablation),
)
```

- [ ] **Step 4: Add run metadata output**

```python
import json
```

```python
meta = {
    "dataset": args.dataset,
    "trait": args.trait,
    "rep": args.rep,
    "model": args.model,
    "ablation": normalize_ablation(args.ablation),
    "result_dir": str(out_dir),
    "delta_path": str(delta_path),
    "gene_path": str(gene_path),
    "plink_prefix": str(fusion_prefix),
}
with open(out_dir / "run_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
```

- [ ] **Step 5: Run launcher tests to verify they pass**

Run:

```bash
pytest DF_GSF_v5/tests/test_launcher_ablation.py -q
```

Expected:

- PASS

## Task 2: Add ablation-aware model behavior in `bio_master_v11`

**Files:**

- Modify: `DF_GSF_v5/src/models/bio_master_v11.py`
- Test: `DF_GSF_v5/tests/test_bio_master_v11_ablation.py`

- [ ] **Step 1: Write the failing model tests**

```python
import numpy as np
import torch

from DF_GSF_v5.src.models.bio_master_v11 import BioMasterV10


def make_inputs():
    delta = np.random.randn(12, 8).astype(np.float32)
    gene = np.random.randn(12, 6).astype(np.float32)
    x = torch.randn(4, 12)
    p = torch.randn(4, 5)
    return delta, gene, x, p


def test_full_mode_forward_runs():
    delta, gene, x, p = make_inputs()
    model = BioMasterV10(delta, gene, num_snps=12, num_pcs=5, block_size=4, ablation="full")
    pred, priors, feat_deep, feat_ctx = model(x, p)
    assert pred.shape == (4, 1)
    assert priors.shape[0] == 12
    assert feat_ctx is not None


def test_no_pca_disables_context_features():
    delta, gene, x, p = make_inputs()
    model = BioMasterV10(delta, gene, num_snps=12, num_pcs=5, block_size=4, ablation="no_pca")
    pred, priors, feat_deep, feat_ctx = model(x, p)
    assert pred.shape == (4, 1)
    assert feat_ctx is None


def test_no_bio_prior_uses_single_channel_transformer():
    delta, gene, x, p = make_inputs()
    model = BioMasterV10(delta, gene, num_snps=12, num_pcs=5, block_size=4, ablation="no_bio_prior")
    pred, priors, feat_deep, feat_ctx = model(x, p)
    assert pred.shape == (4, 1)
    assert model.prior_mode == "single_channel"
    assert priors.shape[0] == 12


def test_pca_only_prior_off_uses_zeroed_prior_channel():
    delta, gene, x, p = make_inputs()
    model = BioMasterV10(delta, gene, num_snps=12, num_pcs=5, block_size=4, ablation="pca_only_prior_off")
    _, priors, _, feat_ctx = model(x, p)
    assert torch.allclose(priors, torch.zeros_like(priors))
    assert feat_ctx is not None
```

- [ ] **Step 2: Run model tests to verify they fail**

Run:

```bash
pytest DF_GSF_v5/tests/test_bio_master_v11_ablation.py -q
```

Expected:

- FAIL because `BioMasterV10` does not yet accept `ablation`

- [ ] **Step 3: Implement ablation config helpers**

```python
VALID_ABLATIONS = {
    "full",
    "no_delta",
    "no_gene2vec",
    "no_bio_prior",
    "no_pca",
    "pca_only_prior_off",
}


def normalize_ablation(ablation):
    ablation = (ablation or "full").strip()
    if ablation not in VALID_ABLATIONS:
        raise ValueError(f"Unsupported ablation: {ablation}")
    return ablation
```

```python
class AblationConfig:
    def __init__(self, ablation):
        ablation = normalize_ablation(ablation)
        self.name = ablation
        self.use_delta = ablation not in {"no_delta", "no_bio_prior", "pca_only_prior_off"}
        self.use_gene = ablation not in {"no_gene2vec", "no_bio_prior", "pca_only_prior_off"}
        self.use_context = ablation != "no_pca"
        if ablation == "no_bio_prior":
            self.prior_mode = "single_channel"
        elif ablation == "pca_only_prior_off":
            self.prior_mode = "zero_two_channel"
        else:
            self.prior_mode = "learned_two_channel"
```

- [ ] **Step 4: Thread ablation config into the model**

```python
class BioMasterV10(nn.Module):
    def __init__(self, delta_E, gene_E, num_snps, num_pcs, block_size=100, ablation="full"):
        super().__init__()
        self.ablation = AblationConfig(ablation)
        self.prior_mode = self.ablation.prior_mode
        self.use_context = self.ablation.use_context
        self.register_buffer("delta_E", torch.tensor(delta_E, dtype=torch.float32))
        self.register_buffer("gene_E", torch.tensor(gene_E, dtype=torch.float32))
```

```python
deep_in_channels = 1 if self.prior_mode == "single_channel" else 2
self.genomic_transformer = GenomicTransformer(
    block_size=block_size,
    in_channels=deep_in_channels,
    d_model=self.d_model,
    nhead=4,
    num_layers=2,
)
```

```python
if self.use_context:
    feat_ctx = self.context_feat_extractor(X_pcs)
    out_ctx = self.context_out(feat_ctx)
else:
    feat_ctx = None
    out_ctx = 0.0
```

- [ ] **Step 5: Implement prior construction without rewriting the rest of `forward()`**

```python
def build_priors(self, num_snps, device):
    if self.prior_mode == "zero_two_channel":
        return torch.zeros((num_snps, 1), device=device)

    if self.prior_mode == "single_channel":
        return torch.zeros((num_snps, 1), device=device)

    delta_input = self.delta_E
    gene_input = self.gene_E

    if not self.ablation.use_delta:
        delta_input = torch.zeros_like(delta_input)
    if not self.ablation.use_gene:
        gene_input = torch.zeros_like(gene_input)

    return self.prior_gen(delta_input, gene_input)
```

```python
priors = self.build_priors(N, X_snps.device)

if self.pad_len > 0:
    X_p = F.pad(X_snps, (0, self.pad_len))
    P_p = F.pad(priors.t(), (0, self.pad_len)).t()
else:
    X_p, P_p = X_snps, priors

g_blocks = X_p.view(B, self.n_blocks, self.block_size)

if self.prior_mode == "single_channel":
    x_folded = g_blocks.unsqueeze(2).view(-1, 1, self.block_size)
else:
    p_blocks = P_p.view(1, self.n_blocks, self.block_size).expand(B, -1, -1)
    x_folded = torch.stack([g_blocks, p_blocks], dim=2).view(-1, 2, self.block_size)
```

- [ ] **Step 6: Disable orthogonality loss automatically for `no_pca`**

```python
def forward(self, pred, target, deep_feat, context_feat):
    p = pred.squeeze()
    t = target.squeeze()
    loss_pcc = self.pcc_loss(p, t)
    loss_rank = self.listnet_loss(p, t)
    if deep_feat is not None and context_feat is not None:
        orth_loss = torch.mean(torch.abs(F.cosine_similarity(deep_feat, context_feat.detach(), dim=0)))
    else:
        orth_loss = torch.tensor(0.0, device=pred.device)
    return self.alpha * loss_pcc + self.beta * loss_rank + self.gamma * orth_loss
```

- [ ] **Step 7: Pass ablation through `train()` and enrich `run_meta.json` inputs**

```python
def train(..., device="auto", ablation="full"):
    ...
    ablation = normalize_ablation(ablation)
    model = BioMasterV10(
        delta,
        gene,
        num_snps=X_tr.shape[1],
        num_pcs=P_tr.shape[1],
        block_size=100,
        ablation=ablation,
    ).to(dev)
```

- [ ] **Step 8: Run model tests to verify they pass**

Run:

```bash
pytest DF_GSF_v5/tests/test_bio_master_v11_ablation.py -q
```

Expected:

- PASS

## Task 3: Add ablation-aware training metadata and smoke coverage

**Files:**

- Modify: `DF_GSF_v5/DF_GSF_v5.py`
- Modify: `DF_GSF_v5/src/models/bio_master_v11.py`
- Test: `DF_GSF_v5/tests/test_launcher_ablation.py`

- [ ] **Step 1: Extend launcher test to assert metadata fields**

```python
import json


def test_step_run_all_writes_run_meta(tmp_path, monkeypatch):
    ...
    launcher.step_run_all(args)

    meta_path = tmp_path / "results/Demo_BF/bio_master_v11__no_delta/rep_01/run_meta.json"
    payload = json.loads(meta_path.read_text())
    assert payload["ablation"] == "no_delta"
    assert payload["model"] == "bio_master_v11"
    assert payload["result_dir"].endswith("bio_master_v11__no_delta/rep_01")
```

- [ ] **Step 2: Run the launcher test to verify it fails**

Run:

```bash
pytest DF_GSF_v5/tests/test_launcher_ablation.py::test_step_run_all_writes_run_meta -q
```

Expected:

- FAIL if `run_meta.json` is not yet complete

- [ ] **Step 3: Add ablation-specific metadata updates**

```python
meta.update({
    "context_enabled": args.ablation != "no_pca",
    "bio_prior_mode": (
        "single_channel"
        if args.ablation == "no_bio_prior"
        else "zero_two_channel"
        if args.ablation == "pca_only_prior_off"
        else "learned_two_channel"
    ),
    "orthogonality_enabled": args.ablation != "no_pca",
})
```

- [ ] **Step 4: Run the metadata test to verify it passes**

Run:

```bash
pytest DF_GSF_v5/tests/test_launcher_ablation.py::test_step_run_all_writes_run_meta -q
```

Expected:

- PASS

## Task 4: Extend batch submission for ablation runs

**Files:**

- Modify: `DF_GSF_v5/submit_jobs.py`
- Test: `DF_GSF_v5/tests/test_launcher_ablation.py`

- [ ] **Step 1: Write the failing submit script test**

```python
import subprocess
import sys
from pathlib import Path


def test_submit_jobs_writes_model_and_ablation_commands(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        \"\"\"
exp_root: "{root}"
resources:
  python_bin: "/usr/bin/python3"
slurm:
  cpu_partition: "cpu"
  gpu_partition: "gpu"
  cpus_per_task: 2
  mem: "4G"
  gres: "gpu:1"
  time: "01:00:00"
experiment:
  replicates: 1
datasets:
  Demo:
    traits: ["BF"]
\"\"\".format(root=tmp_path)
    )

    out_sh = tmp_path / "submit.sh"

    subprocess.run(
        [
            sys.executable,
            "DF_GSF_v5/submit_jobs.py",
            "--config",
            str(cfg_path),
            "--datasets",
            "Demo",
            "--traits",
            "BF",
            "--model",
            "bio_master_v11",
            "--ablations",
            "full,no_delta",
            "--out-sh",
            str(out_sh),
        ],
        check=True,
        cwd=Path.cwd(),
    )

    text = out_sh.read_text()
    assert "--model bio_master_v11" in text
    assert "--ablation full" in text
    assert "--ablation no_delta" in text
```

- [ ] **Step 2: Run the submit script test to verify it fails**

Run:

```bash
pytest DF_GSF_v5/tests/test_launcher_ablation.py::test_submit_jobs_writes_model_and_ablation_commands -q
```

Expected:

- FAIL because `submit_jobs.py` does not yet accept `--model` and `--ablations`

- [ ] **Step 3: Implement minimal submit script changes**

```python
ap.add_argument("--model", default="bio_master_v11")
ap.add_argument("--ablations", default="full", help="Comma-separated ablation names")
```

```python
ablation_list = [x.strip() for x in args.ablations.split(",") if x.strip()]
```

```python
for ablation in ablation_list:
    job = f"{ds}_{trait}_{args.model}_{ablation}_{rep}"
    ...
    body = [
        f"cd {exp_root/'DF_GSF_v5'}",
        f"{cfg['resources']['python_bin']} DF_GSF_v5.py run-all "
        f"--config config/global_config.yaml "
        f"--dataset {ds} --trait {trait} --rep {rep} "
        f"--model {args.model} --ablation {ablation}"
    ]
```

- [ ] **Step 4: Run the submit script test to verify it passes**

Run:

```bash
pytest DF_GSF_v5/tests/test_launcher_ablation.py::test_submit_jobs_writes_model_and_ablation_commands -q
```

Expected:

- PASS

## Task 5: Add ablation result summarization

**Files:**

- Create: `DF_GSF_v5/scripts/compare_ablations.py`
- Test: `DF_GSF_v5/tests/test_launcher_ablation.py`

- [ ] **Step 1: Write the failing summary script test**

```python
import json
import subprocess
import sys


def test_compare_ablations_summarizes_against_full(tmp_path):
    base = tmp_path / "results" / "Demo_BF"
    for model_name, pcc in [
        ("bio_master_v11__full", 0.82),
        ("bio_master_v11__no_delta", 0.75),
    ]:
        rep_dir = base / model_name / "rep_01"
        rep_dir.mkdir(parents=True, exist_ok=True)
        (rep_dir / "stats.json").write_text(json.dumps({"pcc_test": pcc, "mse": 1.0}))

    out_dir = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            "DF_GSF_v5/scripts/compare_ablations.py",
            "--root",
            str(tmp_path),
            "--out",
            str(out_dir),
            "--model",
            "bio_master_v11",
        ],
        check=True,
        cwd=Path.cwd(),
    )

    rep_csv = (out_dir / "ablation_compare_rep.csv").read_text()
    assert "PCC_full" in rep_csv
    assert "PCC_no_delta" in rep_csv
    assert "Delta_full_minus_no_delta" in rep_csv
```

- [ ] **Step 2: Run the summary script test to verify it fails**

Run:

```bash
pytest DF_GSF_v5/tests/test_launcher_ablation.py::test_compare_ablations_summarizes_against_full -q
```

Expected:

- FAIL because `compare_ablations.py` does not yet exist

- [ ] **Step 3: Implement the summary script**

```python
ABLATIONS = [
    "full",
    "no_delta",
    "no_gene2vec",
    "no_bio_prior",
    "no_pca",
    "pca_only_prior_off",
]
```

```python
def stats_path(base, model, ablation, rep):
    return base / f"{model}__{ablation}" / rep / "stats.json"
```

```python
header = [
    "dataset_trait",
    "rep",
    "PCC_full",
    "PCC_no_delta",
    "PCC_no_gene2vec",
    "PCC_no_bio_prior",
    "PCC_no_pca",
    "PCC_pca_only_prior_off",
    "Delta_full_minus_no_delta",
    "Delta_full_minus_no_gene2vec",
    "Delta_full_minus_no_bio_prior",
    "Delta_full_minus_no_pca",
    "Delta_full_minus_pca_only_prior_off",
]
```

- [ ] **Step 4: Run the summary script test to verify it passes**

Run:

```bash
pytest DF_GSF_v5/tests/test_launcher_ablation.py::test_compare_ablations_summarizes_against_full -q
```

Expected:

- PASS

## Task 6: Run focused verification before broader experiments

**Files:**

- Modify: `DF_GSF_v5/DF_GSF_v5.py`
- Modify: `DF_GSF_v5/src/models/bio_master_v11.py`
- Modify: `DF_GSF_v5/submit_jobs.py`
- Create: `DF_GSF_v5/scripts/compare_ablations.py`
- Test: `DF_GSF_v5/tests/test_launcher_ablation.py`
- Test: `DF_GSF_v5/tests/test_bio_master_v11_ablation.py`

- [ ] **Step 1: Run all new focused tests**

Run:

```bash
pytest DF_GSF_v5/tests/test_launcher_ablation.py DF_GSF_v5/tests/test_bio_master_v11_ablation.py -q
```

Expected:

- PASS

- [ ] **Step 2: Run a short smoke training for `full`**

Run:

```bash
python DF_GSF_v5/DF_GSF_v5.py run-all \
  --config DF_GSF_v5/config/global_config.yaml \
  --dataset LargeWhite_Pop1 \
  --trait BF \
  --rep rep_01 \
  --model bio_master_v11 \
  --ablation full
```

Expected:

- result directory `bio_master_v11__full/rep_01`
- `stats.json`, `pred.csv`, `run_meta.json`

- [ ] **Step 3: Run a short smoke training for `no_bio_prior`**

Run:

```bash
python DF_GSF_v5/DF_GSF_v5.py run-all \
  --config DF_GSF_v5/config/global_config.yaml \
  --dataset LargeWhite_Pop1 \
  --trait BF \
  --rep rep_01 \
  --model bio_master_v11 \
  --ablation no_bio_prior
```

Expected:

- result directory `bio_master_v11__no_bio_prior/rep_01`
- `stats.json`, `pred.csv`, `run_meta.json`

- [ ] **Step 4: Run a short smoke training for `no_pca`**

Run:

```bash
python DF_GSF_v5/DF_GSF_v5.py run-all \
  --config DF_GSF_v5/config/global_config.yaml \
  --dataset LargeWhite_Pop1 \
  --trait BF \
  --rep rep_01 \
  --model bio_master_v11 \
  --ablation no_pca
```

Expected:

- result directory `bio_master_v11__no_pca/rep_01`
- `stats.json`, `pred.csv`, `run_meta.json`

- [ ] **Step 5: Run ablation comparison summary**

Run:

```bash
python DF_GSF_v5/scripts/compare_ablations.py \
  --root /work/home/zyqlab/dzhichao/zyqgroup02_duanzhichao/exp_00/experiments/01_benchmarks \
  --out DF_GSF_v5/scripts/ablation_reports \
  --model bio_master_v11
```

Expected:

- `DF_GSF_v5/scripts/ablation_reports/ablation_compare_rep.csv`
- `DF_GSF_v5/scripts/ablation_reports/ablation_compare_summary.csv`

## Self-Review

Spec coverage check:

- CLI ablation support is covered in Task 1.
- model semantics for all six modes are covered in Task 2.
- metadata and interpretability trace fields are covered in Task 3.
- batch submission support is covered in Task 4.
- result summarization is covered in Task 5.
- verification before broader use is covered in Task 6.

Placeholder scan:

- No `TODO`, `TBD`, or deferred implementation markers remain.
- Every task contains concrete file paths, commands, and code snippets.

Type consistency:

- The plan uses one ablation parameter name everywhere: `ablation`.
- The result folder naming convention is consistently `model__ablation`.
- The summary script output names are consistently `ablation_compare_rep.csv` and `ablation_compare_summary.csv`.
