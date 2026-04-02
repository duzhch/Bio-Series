import json
import importlib.util
import importlib.machinery
import sys
import types
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn


def load_bio_master_module(monkeypatch):
    def stub_module(name):
        mod = types.ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        return mod

    pandas_plink_mod = stub_module("pandas_plink")
    pandas_plink_mod.read_plink = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("read_plink should be stubbed in the test")
    )

    sklearn_mod = stub_module("sklearn")
    preprocessing_mod = stub_module("sklearn.preprocessing")
    metrics_mod = stub_module("sklearn.metrics")

    class IdentityScaler:
        def fit_transform(self, x):
            return x

        def transform(self, x):
            return x

    preprocessing_mod.StandardScaler = IdentityScaler
    metrics_mod.mean_squared_error = lambda y_true, y_pred: 0.0

    scipy_mod = stub_module("scipy")
    scipy_stats_mod = stub_module("scipy.stats")
    scipy_stats_mod.pearsonr = lambda *_args, **_kwargs: (0.0, 0.0)

    monkeypatch.setitem(sys.modules, "pandas_plink", pandas_plink_mod)
    monkeypatch.setitem(sys.modules, "sklearn", sklearn_mod)
    monkeypatch.setitem(sys.modules, "sklearn.preprocessing", preprocessing_mod)
    monkeypatch.setitem(sys.modules, "sklearn.metrics", metrics_mod)
    monkeypatch.setitem(sys.modules, "scipy", scipy_mod)
    monkeypatch.setitem(sys.modules, "scipy.stats", scipy_stats_mod)

    module_path = Path(__file__).resolve().parents[1] / "src" / "models" / "bio_master_v11.py"
    module_name = f"bio_master_v11_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_inputs():
    rng = np.random.default_rng(0)
    num_snps = 5
    num_pcs = 3
    delta = rng.standard_normal((num_snps, 4)).astype(np.float32)
    gene = rng.standard_normal((num_snps, 6)).astype(np.float32)
    x_snps = torch.tensor(rng.standard_normal((2, num_snps)).astype(np.float32))
    x_pcs = torch.tensor(rng.standard_normal((2, num_pcs)).astype(np.float32))
    return delta, gene, x_snps, x_pcs


def configure_bias_leakage_trap(prior_gen, branch):
    with torch.no_grad():
        prior_gen.delta_compress.weight.zero_()
        prior_gen.delta_compress.bias.zero_()
        prior_gen.gene_compress.weight.zero_()
        prior_gen.gene_compress.bias.zero_()
        prior_gen.fuse[0].weight.zero_()
        prior_gen.fuse[0].bias.zero_()

        if branch == "delta":
            prior_gen.delta_compress.bias.fill_(2.0)
            prior_gen.fuse[0].weight[:, :16].fill_(1.0)
        elif branch == "gene":
            prior_gen.gene_compress.bias.fill_(3.0)
            prior_gen.fuse[0].weight[:, 16:].fill_(1.0)
        else:
            raise AssertionError(f"Unexpected branch: {branch}")


def test_full_mode_forward_runs(monkeypatch):
    module = load_bio_master_module(monkeypatch)
    delta, gene, x_snps, x_pcs = make_inputs()

    model = module.BioMasterV10(
        delta,
        gene,
        num_snps=x_snps.shape[1],
        num_pcs=x_pcs.shape[1],
        block_size=4,
        ablation="full",
    )

    pred, priors, feat_deep, feat_ctx = model(x_snps, x_pcs)

    assert model.genomic_transformer.local_conv[0].in_channels == 2
    assert pred.shape == (2, 1)
    assert priors.shape == (x_snps.shape[1], 1)
    assert feat_deep.shape == (2, 64)
    assert feat_ctx.shape == (2, 64)


def test_no_pca_disables_context_features(monkeypatch):
    module = load_bio_master_module(monkeypatch)
    delta, gene, x_snps, x_pcs = make_inputs()

    model = module.BioMasterV10(
        delta,
        gene,
        num_snps=x_snps.shape[1],
        num_pcs=x_pcs.shape[1],
        block_size=4,
        ablation="no_pca",
    )

    pred, _priors, feat_deep, feat_ctx = model(x_snps, x_pcs)
    loss = module.HybridPCCLoss()(pred, torch.zeros_like(pred), feat_deep, feat_ctx)

    assert model.context_feat_extractor is None
    assert feat_ctx is None
    assert torch.isfinite(loss)


def test_no_bio_prior_uses_single_channel_mode(monkeypatch):
    module = load_bio_master_module(monkeypatch)
    delta, gene, x_snps, x_pcs = make_inputs()

    model = module.BioMasterV10(
        delta,
        gene,
        num_snps=x_snps.shape[1],
        num_pcs=x_pcs.shape[1],
        block_size=4,
        ablation="no_bio_prior",
    )

    _pred, _priors, _feat_deep, feat_ctx = model(x_snps, x_pcs)

    assert model.genomic_transformer.local_conv[0].in_channels == 1
    assert feat_ctx.shape == (2, 64)


def test_pca_only_prior_off_returns_zero_priors_and_keeps_context(monkeypatch):
    module = load_bio_master_module(monkeypatch)
    delta, gene, x_snps, x_pcs = make_inputs()

    model = module.BioMasterV10(
        delta,
        gene,
        num_snps=x_snps.shape[1],
        num_pcs=x_pcs.shape[1],
        block_size=4,
        ablation="pca_only_prior_off",
    )

    _pred, priors, _feat_deep, feat_ctx = model(x_snps, x_pcs)

    assert model.genomic_transformer.local_conv[0].in_channels == 2
    assert torch.count_nonzero(priors) == 0
    assert feat_ctx.shape == (2, 64)


def test_no_delta_zeros_branch_after_projection_to_prevent_bias_leak(monkeypatch):
    module = load_bio_master_module(monkeypatch)
    delta, gene, _x_snps, _x_pcs = make_inputs()

    full_model = module.BioMasterV10(delta, gene, num_snps=delta.shape[0], num_pcs=3, ablation="full")
    no_delta_model = module.BioMasterV10(
        delta,
        gene,
        num_snps=delta.shape[0],
        num_pcs=3,
        ablation="no_delta",
    )
    configure_bias_leakage_trap(full_model.prior_gen, branch="delta")
    configure_bias_leakage_trap(no_delta_model.prior_gen, branch="delta")

    full_priors = full_model._build_priors()
    no_delta_priors = no_delta_model._build_priors()

    assert torch.all(full_priors > 0.5)
    assert torch.allclose(no_delta_priors, torch.full_like(no_delta_priors, 0.5))


def test_no_gene2vec_zeros_branch_after_projection_to_prevent_bias_leak(monkeypatch):
    module = load_bio_master_module(monkeypatch)
    delta, gene, _x_snps, _x_pcs = make_inputs()

    full_model = module.BioMasterV10(delta, gene, num_snps=delta.shape[0], num_pcs=3, ablation="full")
    no_gene_model = module.BioMasterV10(
        delta,
        gene,
        num_snps=delta.shape[0],
        num_pcs=3,
        ablation="no_gene2vec",
    )
    configure_bias_leakage_trap(full_model.prior_gen, branch="gene")
    configure_bias_leakage_trap(no_gene_model.prior_gen, branch="gene")

    full_priors = full_model._build_priors()
    no_gene_priors = no_gene_model._build_priors()

    assert torch.all(full_priors > 0.5)
    assert torch.allclose(no_gene_priors, torch.full_like(no_gene_priors, 0.5))


def test_hybrid_loss_orthogonality_uses_per_sample_feature_alignment(monkeypatch):
    module = load_bio_master_module(monkeypatch)
    loss_fn = module.HybridPCCLoss(alpha=0.0, beta=0.0, gamma=1.0)

    pred = torch.zeros((2, 1), dtype=torch.float32)
    target = torch.zeros((2, 1), dtype=torch.float32)
    deep_feat = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    context_feat = torch.tensor([[2.0, 1.0], [4.0, 3.0]], dtype=torch.float32)

    loss = loss_fn(pred, target, deep_feat, context_feat)
    expected = torch.tensor((0.8 + 24.0 / 25.0) / 2.0, dtype=torch.float32)

    assert torch.isclose(loss, expected, atol=1e-6)


def test_train_threads_ablation_to_model(monkeypatch, tmp_path):
    module = load_bio_master_module(monkeypatch)
    captured = {}

    class DummyBed:
        def __init__(self, data):
            self._data = data

        def compute(self):
            return self._data

    class DummyModel(nn.Module):
        def __init__(self, delta_E, gene_E, num_snps, num_pcs, block_size=100, ablation="full"):
            super().__init__()
            captured["ablation"] = ablation
            self.bias = nn.Parameter(torch.zeros(1))
            self.wide_linear = nn.Linear(num_snps, 1)

        def forward(self, x_snps, x_pcs):
            batch_size, num_snps = x_snps.shape
            pred = self.wide_linear(x_snps) + self.bias
            priors = torch.zeros(num_snps, 1, device=x_snps.device)
            feat_deep = torch.zeros(batch_size, 64, device=x_snps.device)
            feat_ctx = torch.zeros(batch_size, 64, device=x_snps.device)
            return pred, priors, feat_deep, feat_ctx

    def fake_read_csv(path, sep=",", names=None):
        path = str(path)
        if path.endswith("train.ids"):
            return pd.DataFrame([["F1", "1"]], columns=names)
        if path.endswith("test.ids"):
            return pd.DataFrame([["F2", "2"]], columns=names)
        raise AssertionError(f"Unexpected read_csv path: {path}")

    geno = np.array(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.4, 0.3, 0.2, 0.1],
        ],
        dtype=np.float32,
    )
    delta = np.ones((5, 4), dtype=np.float32)
    gene = np.ones((5, 300), dtype=np.float32)

    monkeypatch.setattr(module, "BioMasterV10", DummyModel)
    monkeypatch.setattr(module, "read_plink", lambda *_args, **_kwargs: (None, pd.DataFrame({"iid": ["1", "2"]}), DummyBed(geno.T)))
    monkeypatch.setattr(module, "_load_pheno_map", lambda *_args, **_kwargs: {"1": 1.0, "2": 2.0})
    monkeypatch.setattr(module.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(module, "load_pca", lambda _path, ids: np.zeros((len(ids), 3), dtype=np.float32))
    monkeypatch.setattr(module.np, "load", lambda path: delta if "delta" in str(path) else gene)

    result = module.train(
        plink_prefix="plink",
        pheno_file="pheno.csv",
        train_ids="train.ids",
        test_ids="test.ids",
        trait="trait",
        delta_path="delta.npy",
        gene_path="gene.npy",
        out_dir=str(tmp_path),
        epochs=0,
        ablation="no_delta",
    )

    assert captured["ablation"] == "no_delta"
    assert result == {"pcc": 0.0, "mse": 0.0}


def test_train_stats_sparsity_counts_nonzero_priors_per_snp(monkeypatch, tmp_path):
    module = load_bio_master_module(monkeypatch)

    class DummyBed:
        def __init__(self, data):
            self._data = data

        def compute(self):
            return self._data

    class DummyModel(nn.Module):
        def __init__(self, delta_E, gene_E, num_snps, num_pcs, block_size=100, ablation="full"):
            super().__init__()
            self.bias = nn.Parameter(torch.zeros(1))
            self.wide_linear = nn.Linear(num_snps, 1)

        def forward(self, x_snps, x_pcs):
            batch_size, num_snps = x_snps.shape
            pred = self.wide_linear(x_snps) + self.bias
            priors = torch.tensor(
                [[0.0], [0.0020], [0.5000], [0.0], [0.0005]],
                dtype=torch.float32,
                device=x_snps.device,
            )
            feat_deep = torch.zeros(batch_size, 64, device=x_snps.device)
            feat_ctx = torch.zeros(batch_size, 64, device=x_snps.device)
            return pred, priors[:num_snps], feat_deep, feat_ctx

    def fake_read_csv(path, sep=",", names=None):
        path = str(path)
        if path.endswith("train.ids"):
            return pd.DataFrame([["F1", "1"]], columns=names)
        if path.endswith("test.ids"):
            return pd.DataFrame([["F2", "2"]], columns=names)
        raise AssertionError(f"Unexpected read_csv path: {path}")

    geno = np.array(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.4, 0.3, 0.2, 0.1],
        ],
        dtype=np.float32,
    )
    delta = np.ones((5, 4), dtype=np.float32)
    gene = np.ones((5, 300), dtype=np.float32)

    monkeypatch.setattr(module, "BioMasterV10", DummyModel)
    monkeypatch.setattr(
        module,
        "read_plink",
        lambda *_args, **_kwargs: (None, pd.DataFrame({"iid": ["1", "2"]}), DummyBed(geno.T)),
    )
    monkeypatch.setattr(module, "_load_pheno_map", lambda *_args, **_kwargs: {"1": 1.0, "2": 2.0})
    monkeypatch.setattr(module.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(module, "load_pca", lambda _path, ids: np.zeros((len(ids), 3), dtype=np.float32))
    monkeypatch.setattr(module.np, "load", lambda path: delta if "delta" in str(path) else gene)

    module.train(
        plink_prefix="plink",
        pheno_file="pheno.csv",
        train_ids="train.ids",
        test_ids="test.ids",
        trait="trait",
        delta_path="delta.npy",
        gene_path="gene.npy",
        out_dir=str(tmp_path),
        epochs=0,
        ablation="full",
    )

    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["sparsity"] == 2
