import csv
import importlib.util
import json
import subprocess
import sys
import types
import uuid
from argparse import Namespace
from pathlib import Path

import pytest


def load_launcher_module(monkeypatch):
    src_pkg = types.ModuleType("src")
    src_pkg.__path__ = [str(Path(__file__).resolve().parents[1] / "src")]

    gwas_mod = types.ModuleType("src.gwas")
    gwas_mod.run_gwas_pipeline = lambda *_args, **_kwargs: {
        "snps_for_emb": "stub_snps.tsv",
        "plink_snps": "stub_plink.snps",
    }

    features_mod = types.ModuleType("src.features")
    features_mod.extract_delta = lambda *_args, **_kwargs: None
    features_mod.extract_gene2vec = lambda *_args, **_kwargs: None
    features_mod.annotate_snps_with_gtf = lambda *_args, **_kwargs: None

    data_mod = types.ModuleType("src.data")
    data_mod.make_all_ids = lambda *_args, **_kwargs: None
    data_mod.plink_extract = lambda *_args, **_kwargs: None

    monkeypatch.setitem(sys.modules, "src", src_pkg)
    monkeypatch.setitem(sys.modules, "src.gwas", gwas_mod)
    monkeypatch.setitem(sys.modules, "src.features", features_mod)
    monkeypatch.setitem(sys.modules, "src.data", data_mod)

    launcher_path = Path(__file__).resolve().parents[1] / "DF_GSF_v5.py"
    module_name = f"df_gsf_v5_launcher_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, launcher_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    original_sys_path = list(sys.path)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_sys_path
    return module


def load_submit_jobs_module():
    submit_jobs_path = Path(__file__).resolve().parents[1] / "submit_jobs.py"
    module_name = f"df_gsf_v5_submit_jobs_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, submit_jobs_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_launcher_import_isolated_from_real_src_dependencies(monkeypatch):
    before_path = list(sys.path)
    launcher = load_launcher_module(monkeypatch)
    assert hasattr(launcher, "step_run_all")
    assert sys.path == before_path


def test_infer_paths_includes_normalized_ablation_segment(monkeypatch):
    launcher = load_launcher_module(monkeypatch)
    cfg = {"exp_root": "/tmp/exp-root"}

    out_dir, train_ids, test_ids = launcher.infer_paths(
        cfg=cfg,
        dataset="pig",
        trait="backfat",
        rep="rep1",
        model_name="bio_master_v8",
        ablation="No_Delta",
    )

    assert str(out_dir).endswith("results/pig_backfat/bio_master_v8__no_delta/rep1")
    assert str(train_ids).endswith("data/splits/pig_backfat/rep1/train.ids")
    assert str(test_ids).endswith("data/splits/pig_backfat/rep1/test.ids")


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


def test_step_run_all_passes_normalized_ablation_and_writes_run_meta(tmp_path, monkeypatch):
    launcher = load_launcher_module(monkeypatch)
    cfg = {
        "exp_root": str(tmp_path),
        "datasets": {"pig": {"plink": "PLINK_PREFIX", "pheno": "PHENO_FILE"}},
        "resources": {
            "plink_bin": "plink",
            "gcta_bin": "gcta",
            "reference_genome": "ref.fa",
            "pigbert_model": "pigbert.pt",
            "gtf_file": "genes.gtf",
            "gene2vec_model": "gene2vec.bin",
        },
        "experiment": {
            "top_n_snps": 10,
            "lr": 1e-3,
            "batch_size": 16,
            "epochs": 1,
            "lambda_l1": 0.0,
        },
    }
    trainer_calls = {}

    def fake_train(
        plink_prefix,
        pheno_file,
        train_ids,
        test_ids,
        trait,
        delta_path,
        gene_path,
        out_dir,
        lr,
        batch_size,
        epochs,
        lambda_l1,
        device,
        ablation,
    ):
        trainer_calls.update(
            {
                "plink_prefix": plink_prefix,
                "pheno_file": pheno_file,
                "train_ids": train_ids,
                "test_ids": test_ids,
                "trait": trait,
                "delta_path": delta_path,
                "gene_path": gene_path,
                "out_dir": out_dir,
                "lr": lr,
                "batch_size": batch_size,
                "epochs": epochs,
                "lambda_l1": lambda_l1,
                "device": device,
                "ablation": ablation,
            }
        )

    monkeypatch.setattr(launcher, "load_cfg", lambda _path: cfg)
    monkeypatch.setattr(
        launcher,
        "run_gwas_pipeline",
        lambda **_kwargs: {"snps_for_emb": "snps.tsv", "plink_snps": "snps.list"},
    )
    monkeypatch.setattr(launcher, "extract_delta", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher, "annotate_snps_with_gtf", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher, "extract_gene2vec", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher, "make_all_ids", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher, "plink_extract", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher, "get_model_trainer", lambda _model: fake_train)

    args = Namespace(
        config="dummy.yaml",
        dataset="pig",
        trait="backfat",
        rep="rep2",
        model="my_model",
        ablation="No_Gene2Vec",
    )
    launcher.step_run_all(args)

    assert trainer_calls["ablation"] == "no_gene2vec"

    out_dir, _, _ = launcher.infer_paths(
        cfg=cfg,
        dataset="pig",
        trait="backfat",
        rep="rep2",
        model_name="my_model",
        ablation="no_gene2vec",
    )
    run_meta_path = out_dir / "run_meta.json"
    assert run_meta_path.exists()

    run_meta = json.loads(run_meta_path.read_text())
    assert run_meta["dataset"] == "pig"
    assert run_meta["trait"] == "backfat"
    assert run_meta["rep"] == "rep2"
    assert run_meta["model"] == "my_model"
    assert run_meta["ablation"] == "no_gene2vec"
    assert run_meta["result_dir"] == str(out_dir)
    assert run_meta["delta_path"] == str(out_dir / "delta_embeddings.npy")
    assert run_meta["gene_path"] == str(out_dir / "gene_knowledge.npy")
    assert run_meta["plink_prefix"] == str(out_dir / "fusion_geno")
    assert run_meta["context_enabled"] is True
    assert run_meta["bio_prior_mode"] == "learned_two_channel"
    assert run_meta["orthogonality_enabled"] is True


@pytest.mark.parametrize(
    ("ablation", "context_enabled", "bio_prior_mode", "orthogonality_enabled"),
    [
        ("full", True, "learned_two_channel", True),
        ("no_delta", True, "learned_two_channel", True),
        ("no_gene2vec", True, "learned_two_channel", True),
        ("no_bio_prior", True, "single_channel", True),
        ("no_pca", False, "learned_two_channel", False),
        ("pca_only_prior_off", True, "zero_two_channel", True),
    ],
)
def test_get_ablation_semantics_matches_expected_contract(
    monkeypatch, ablation, context_enabled, bio_prior_mode, orthogonality_enabled
):
    launcher = load_launcher_module(monkeypatch)

    semantics = launcher.get_ablation_semantics(ablation)

    assert semantics == {
        "context_enabled": context_enabled,
        "bio_prior_mode": bio_prior_mode,
        "orthogonality_enabled": orthogonality_enabled,
    }


def test_normalize_ablation_validates_supported_values(monkeypatch):
    launcher = load_launcher_module(monkeypatch)

    assert launcher.normalize_ablation("PCA_ONLY_PRIOR_OFF") == "pca_only_prior_off"
    assert launcher.normalize_ablation("No-Delta") == "no_delta"
    assert launcher.normalize_ablation(" No Delta ") == "no_delta"
    with pytest.raises(ValueError):
        launcher.normalize_ablation("unsupported_mode")


def test_step_run_gwas_creates_out_dir_and_uses_keyword_call(tmp_path, monkeypatch):
    launcher = load_launcher_module(monkeypatch)
    cfg = {
        "exp_root": str(tmp_path),
        "datasets": {"pig": {"plink": "PLINK_PREFIX", "pheno": "PHENO_FILE"}},
        "resources": {"plink_bin": "plink", "gcta_bin": "gcta"},
        "experiment": {"top_n_snps": 321},
    }
    call_kwargs = {}

    def fake_gwas_pipeline(**kwargs):
        call_kwargs.update(kwargs)
        return {"snps_for_emb": "snps.tsv", "plink_snps": "snps.list"}

    monkeypatch.setattr(launcher, "load_cfg", lambda _path: cfg)
    monkeypatch.setattr(launcher, "run_gwas_pipeline", fake_gwas_pipeline)

    args = Namespace(
        config="dummy.yaml",
        dataset="pig",
        trait="backfat",
        rep="rep7",
        model="my_model",
        ablation="no_pca",
    )
    launcher.step_run_gwas(args)

    out_dir, train_ids, _ = launcher.infer_paths(
        cfg=cfg,
        dataset="pig",
        trait="backfat",
        rep="rep7",
        model_name="my_model",
        ablation="no_pca",
    )
    assert out_dir.exists()
    assert call_kwargs["plink_prefix"] == "PLINK_PREFIX"
    assert call_kwargs["pheno_file"] == "PHENO_FILE"
    assert call_kwargs["train_ids"] == str(train_ids)
    assert call_kwargs["trait"] == "backfat"
    assert call_kwargs["out_dir"] == str(out_dir)
    assert call_kwargs["plink_bin"] == "plink"
    assert call_kwargs["gcta_bin"] == "gcta"
    assert call_kwargs["top_n"] == 321


def test_step_run_all_omits_ablation_for_legacy_trainer_when_full(tmp_path, monkeypatch):
    launcher = load_launcher_module(monkeypatch)
    cfg = {
        "exp_root": str(tmp_path),
        "datasets": {"pig": {"plink": "PLINK_PREFIX", "pheno": "PHENO_FILE"}},
        "resources": {
            "plink_bin": "plink",
            "gcta_bin": "gcta",
            "reference_genome": "ref.fa",
            "pigbert_model": "pigbert.pt",
            "gtf_file": "genes.gtf",
            "gene2vec_model": "gene2vec.bin",
        },
        "experiment": {
            "top_n_snps": 10,
            "lr": 1e-3,
            "batch_size": 16,
            "epochs": 1,
            "lambda_l1": 0.0,
        },
    }
    trainer_calls = {}

    def legacy_train(
        plink_prefix,
        pheno_file,
        train_ids,
        test_ids,
        trait,
        delta_path,
        gene_path,
        out_dir,
        lr,
        batch_size,
        epochs,
        lambda_l1,
        device,
    ):
        trainer_calls.update(
            {
                "plink_prefix": plink_prefix,
                "pheno_file": pheno_file,
                "out_dir": out_dir,
                "device": device,
            }
        )

    monkeypatch.setattr(launcher, "load_cfg", lambda _path: cfg)
    monkeypatch.setattr(
        launcher,
        "run_gwas_pipeline",
        lambda **_kwargs: {"snps_for_emb": "snps.tsv", "plink_snps": "snps.list"},
    )
    monkeypatch.setattr(launcher, "extract_delta", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher, "annotate_snps_with_gtf", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher, "extract_gene2vec", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher, "make_all_ids", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher, "plink_extract", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher, "get_model_trainer", lambda _model: legacy_train)

    args = Namespace(
        config="dummy.yaml",
        dataset="pig",
        trait="backfat",
        rep="rep_legacy",
        model="legacy_model",
        ablation="full",
    )
    launcher.step_run_all(args)

    assert "out_dir" in trainer_calls


def test_step_run_all_rejects_non_full_ablation_for_legacy_trainer(tmp_path, monkeypatch, capsys):
    launcher = load_launcher_module(monkeypatch)
    cfg = {
        "exp_root": str(tmp_path),
        "datasets": {"pig": {"plink": "PLINK_PREFIX", "pheno": "PHENO_FILE"}},
        "resources": {
            "plink_bin": "plink",
            "gcta_bin": "gcta",
            "reference_genome": "ref.fa",
            "pigbert_model": "pigbert.pt",
            "gtf_file": "genes.gtf",
            "gene2vec_model": "gene2vec.bin",
        },
        "experiment": {
            "top_n_snps": 10,
            "lr": 1e-3,
            "batch_size": 16,
            "epochs": 1,
            "lambda_l1": 0.0,
        },
    }

    def legacy_train(
        plink_prefix,
        pheno_file,
        train_ids,
        test_ids,
        trait,
        delta_path,
        gene_path,
        out_dir,
        lr,
        batch_size,
        epochs,
        lambda_l1,
        device,
    ):
        return None

    monkeypatch.setattr(launcher, "load_cfg", lambda _path: cfg)
    monkeypatch.setattr(
        launcher,
        "run_gwas_pipeline",
        lambda **_kwargs: {"snps_for_emb": "snps.tsv", "plink_snps": "snps.list"},
    )
    monkeypatch.setattr(launcher, "extract_delta", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher, "annotate_snps_with_gtf", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher, "extract_gene2vec", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher, "make_all_ids", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher, "plink_extract", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher, "get_model_trainer", lambda _model: legacy_train)
    monkeypatch.setattr(launcher.traceback, "print_exc", lambda: None)

    args = Namespace(
        config="dummy.yaml",
        dataset="pig",
        trait="backfat",
        rep="rep_legacy_2",
        model="legacy_model",
        ablation="no_delta",
    )
    with pytest.raises(SystemExit):
        launcher.step_run_all(args)

    out = capsys.readouterr().out
    assert "does not support ablation mode" in out


def test_step_run_all_rejects_non_full_ablation_for_kwargs_only_trainer(
    tmp_path, monkeypatch, capsys
):
    launcher = load_launcher_module(monkeypatch)
    cfg = {
        "exp_root": str(tmp_path),
        "datasets": {"pig": {"plink": "PLINK_PREFIX", "pheno": "PHENO_FILE"}},
        "resources": {
            "plink_bin": "plink",
            "gcta_bin": "gcta",
            "reference_genome": "ref.fa",
            "pigbert_model": "pigbert.pt",
            "gtf_file": "genes.gtf",
            "gene2vec_model": "gene2vec.bin",
        },
        "experiment": {
            "top_n_snps": 10,
            "lr": 1e-3,
            "batch_size": 16,
            "epochs": 1,
            "lambda_l1": 0.0,
        },
    }

    def kwargs_only_train(
        plink_prefix,
        pheno_file,
        train_ids,
        test_ids,
        trait,
        delta_path,
        gene_path,
        out_dir,
        lr,
        batch_size,
        epochs,
        lambda_l1,
        device,
        **kwargs,
    ):
        return None

    monkeypatch.setattr(launcher, "load_cfg", lambda _path: cfg)
    monkeypatch.setattr(
        launcher,
        "run_gwas_pipeline",
        lambda **_kwargs: {"snps_for_emb": "snps.tsv", "plink_snps": "snps.list"},
    )
    monkeypatch.setattr(launcher, "extract_delta", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher, "annotate_snps_with_gtf", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher, "extract_gene2vec", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher, "make_all_ids", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher, "plink_extract", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher, "get_model_trainer", lambda _model: kwargs_only_train)
    monkeypatch.setattr(launcher.traceback, "print_exc", lambda: None)

    args = Namespace(
        config="dummy.yaml",
        dataset="pig",
        trait="backfat",
        rep="rep_kwargs_only",
        model="kwargs_model",
        ablation="no_delta",
    )
    with pytest.raises(SystemExit):
        launcher.step_run_all(args)

    out = capsys.readouterr().out
    assert "does not support ablation mode" in out


def test_submit_jobs_defaults_emit_explicit_model_and_ablation_flags(tmp_path, monkeypatch):
    submit_jobs = load_submit_jobs_module()
    exp_root = tmp_path / "exp"
    config_path = tmp_path / "global_config.yaml"
    out_sh = tmp_path / "submit_all.sh"
    cfg = {
        "exp_root": str(exp_root),
        "experiment": {"replicates": 1},
        "datasets": {"pig": {"traits": ["backfat"]}},
        "slurm": {
            "cpu_partition": "cpu",
            "gpu_partition": "gpu",
            "cpus_per_task": 4,
            "mem": "8G",
            "gres": "gpu:1",
            "time": "00:30:00",
        },
        "resources": {"python_bin": "python"},
    }
    config_path.write_text(json.dumps(cfg))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "submit_jobs.py",
            "--config",
            str(config_path),
            "--datasets",
            "pig",
            "--traits",
            "backfat",
            "--out-sh",
            str(out_sh),
        ],
    )

    submit_jobs.main()

    submit_text = out_sh.read_text()
    script_paths = list((exp_root / "slurm_jobs" / "scripts_v5").glob("*.sh"))
    assert len(script_paths) == 1
    script_text = script_paths[0].read_text()
    expected_code_dir = Path(submit_jobs.__file__).resolve().parent
    expected_config = config_path.resolve()
    assert f"cd {expected_code_dir}" in script_text
    assert f"--config {expected_config}" in script_text
    assert "--model bio_master_v11" in script_text
    assert "--ablation full" in script_text
    assert "--model bio_master_v11 --ablation full" in script_text
    assert "bio_master_v11_full" in script_paths[0].name
    assert f"--config {expected_config}" in submit_text
    assert "--model bio_master_v11 --ablation full" in submit_text
    assert f"sbatch {script_paths[0]}" in submit_text


def test_submit_jobs_generates_one_script_per_rep_and_ablation_with_explicit_flags(
    tmp_path, monkeypatch
):
    submit_jobs = load_submit_jobs_module()
    exp_root = tmp_path / "exp_multi"
    config_path = tmp_path / "global_config_multi.yaml"
    out_sh = tmp_path / "submit_multi.sh"
    cfg = {
        "exp_root": str(exp_root),
        "experiment": {"replicates": 2},
        "datasets": {"pig": {"traits": ["backfat"]}},
        "slurm": {
            "cpu_partition": "cpu",
            "gpu_partition": "gpu",
            "cpus_per_task": 4,
            "mem": "8G",
            "gres": "gpu:1",
            "time": "00:30:00",
        },
        "resources": {"python_bin": "python"},
    }
    config_path.write_text(json.dumps(cfg))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "submit_jobs.py",
            "--config",
            str(config_path),
            "--datasets",
            "pig",
            "--traits",
            "backfat",
            "--model",
            "bio_master_v13",
            "--ablations",
            "full,No-Delta",
            "--out-sh",
            str(out_sh),
        ],
    )

    submit_jobs.main()

    script_paths = sorted((exp_root / "slurm_jobs" / "scripts_v5").glob("*.sh"))
    assert len(script_paths) == 4

    submit_lines = [line for line in out_sh.read_text().splitlines() if line.startswith("sbatch ")]
    assert len(submit_lines) == 4
    submit_text = out_sh.read_text()
    assert submit_text.count("--model bio_master_v13 --ablation full") == 2
    assert submit_text.count("--model bio_master_v13 --ablation no_delta") == 2
    assert all("No-Delta" not in line for line in submit_text.splitlines())

    script_texts = [script.read_text() for script in script_paths]
    assert all("--model bio_master_v13" in text for text in script_texts)
    assert sum("--ablation full" in text for text in script_texts) == 2
    assert sum("--ablation no_delta" in text for text in script_texts) == 2
    assert all("--config " + str(config_path.resolve()) in text for text in script_texts)
    assert any("bio_master_v13_no_delta" in script.name for script in script_paths)


def test_submit_jobs_rejects_unsupported_normalized_ablation(tmp_path, monkeypatch, capsys):
    submit_jobs = load_submit_jobs_module()
    exp_root = tmp_path / "exp_invalid"
    config_path = tmp_path / "global_config_invalid.yaml"
    out_sh = tmp_path / "submit_invalid.sh"
    cfg = {
        "exp_root": str(exp_root),
        "experiment": {"replicates": 1},
        "datasets": {"pig": {"traits": ["backfat"]}},
        "slurm": {
            "cpu_partition": "cpu",
            "gpu_partition": "gpu",
            "cpus_per_task": 4,
            "mem": "8G",
            "gres": "gpu:1",
            "time": "00:30:00",
        },
        "resources": {"python_bin": "python"},
    }
    config_path.write_text(json.dumps(cfg))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "submit_jobs.py",
            "--config",
            str(config_path),
            "--datasets",
            "pig",
            "--traits",
            "backfat",
            "--ablations",
            "No-Delta,unknown mode",
            "--out-sh",
            str(out_sh),
        ],
    )

    with pytest.raises(SystemExit):
        submit_jobs.main()

    err = capsys.readouterr().err.lower()
    assert "unsupported ablation" in err
    assert "unknown_mode" in err


def test_compare_ablations_summarizes_against_full(tmp_path):
    base = tmp_path / "results" / "Demo_BF"
    for model_name, pcc, mse in [
        ("bio_master_v11__full", 0.82, 1.0),
        ("bio_master_v11__drop_aux", 0.75, 1.2),
    ]:
        rep_dir = base / model_name / "rep_01"
        rep_dir.mkdir(parents=True, exist_ok=True)
        (rep_dir / "stats.json").write_text(json.dumps({"pcc_test": pcc, "mse": mse}))

    out_dir = tmp_path / "out"
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "compare_ablations.py"
    subprocess.run(
        [
            sys.executable,
            str(script_path),
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

    rep_path = out_dir / "ablation_compare_rep.csv"
    summary_path = out_dir / "ablation_compare_summary.csv"

    rep_rows = list(csv.DictReader(rep_path.open(newline="")))
    assert rep_rows
    rep_fieldnames = rep_rows[0].keys()
    assert "PCC_full" in rep_rows[0]
    assert "PCC_drop_aux" in rep_rows[0]
    assert "Delta_full_minus_drop_aux" in rep_rows[0]
    assert "MSE_full" in rep_rows[0]
    assert "MSE_drop_aux" in rep_rows[0]
    assert "PCC_no_gene2vec" not in rep_fieldnames
    assert list(rep_fieldnames)[:4] == [
        "dataset_trait",
        "rep",
        "PCC_full",
        "PCC_drop_aux",
    ]
    assert rep_rows[0]["dataset_trait"] == "Demo_BF"
    assert rep_rows[0]["rep"] == "rep_01"
    assert rep_rows[0]["PCC_full"] == "0.82"
    assert rep_rows[0]["PCC_drop_aux"] == "0.75"
    assert float(rep_rows[0]["Delta_full_minus_drop_aux"]) == pytest.approx(0.07)
    assert rep_rows[0]["MSE_drop_aux"] == "1.2"

    summary_rows = list(csv.DictReader(summary_path.open(newline="")))
    assert summary_rows
    summary_fieldnames = summary_rows[0].keys()
    assert "PCC_full_mean" in summary_rows[0]
    assert "mean_Delta_full_minus_drop_aux" in summary_rows[0]
    assert "MSE_drop_aux_mean" in summary_rows[0]
    assert "PCC_no_gene2vec_mean" not in summary_fieldnames
    assert list(summary_fieldnames)[:6] == [
        "dataset_trait",
        "N_full_reps",
        "PCC_full_mean",
        "PCC_full_std",
        "PCC_drop_aux_mean",
        "PCC_drop_aux_std",
    ]
    assert summary_rows[0]["dataset_trait"] == "Demo_BF"
    assert summary_rows[0]["N_full_reps"] == "1"
    assert summary_rows[0]["PCC_full_mean"] == "0.8200"
    assert summary_rows[0]["mean_Delta_full_minus_drop_aux"] == "0.0700"
    assert summary_rows[0]["MSE_drop_aux_mean"] == "1.2000"


def test_compare_ablations_mixed_layout_keeps_dataset_dirs(tmp_path):
    results_root = tmp_path / "results"

    stray_rep_dir = results_root / "bio_master_v11__full" / "rep_99"
    stray_rep_dir.mkdir(parents=True, exist_ok=True)
    (stray_rep_dir / "stats.json").write_text(json.dumps({"pcc_test": 0.11, "mse": 9.9}))

    dataset_base = results_root / "Demo_BF"
    for model_name, pcc, mse in [
        ("bio_master_v11__full", 0.82, 1.0),
        ("bio_master_v11__drop_aux", 0.75, 1.2),
    ]:
        rep_dir = dataset_base / model_name / "rep_01"
        rep_dir.mkdir(parents=True, exist_ok=True)
        (rep_dir / "stats.json").write_text(json.dumps({"pcc_test": pcc, "mse": mse}))

    out_dir = tmp_path / "out_mixed"
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "compare_ablations.py"
    subprocess.run(
        [
            sys.executable,
            str(script_path),
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

    rep_rows = list(csv.DictReader((out_dir / "ablation_compare_rep.csv").open(newline="")))
    assert len(rep_rows) == 1
    assert rep_rows[0]["dataset_trait"] == "Demo_BF"
    assert rep_rows[0]["rep"] == "rep_01"
    assert rep_rows[0]["PCC_full"] == "0.82"


def test_compare_ablations_treats_non_numeric_stats_as_missing(tmp_path):
    base = tmp_path / "results" / "Demo_BF"
    for model_name, stats in [
        ("bio_master_v11__full", {"pcc_test": 0.82, "mse": 1.0}),
        ("bio_master_v11__drop_aux", {"pcc_test": "NA", "mse": ""}),
    ]:
        rep_dir = base / model_name / "rep_01"
        rep_dir.mkdir(parents=True, exist_ok=True)
        (rep_dir / "stats.json").write_text(json.dumps(stats))

    out_dir = tmp_path / "out_non_numeric"
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "compare_ablations.py"
    subprocess.run(
        [
            sys.executable,
            str(script_path),
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

    rep_rows = list(csv.DictReader((out_dir / "ablation_compare_rep.csv").open(newline="")))
    assert len(rep_rows) == 1
    assert rep_rows[0]["PCC_full"] == "0.82"
    assert rep_rows[0]["PCC_drop_aux"] == ""
    assert rep_rows[0]["Delta_full_minus_drop_aux"] == ""
    assert rep_rows[0]["MSE_drop_aux"] == ""

    summary_rows = list(csv.DictReader((out_dir / "ablation_compare_summary.csv").open(newline="")))
    assert len(summary_rows) == 1
    assert summary_rows[0]["PCC_drop_aux_mean"] == ""
    assert summary_rows[0]["MSE_drop_aux_mean"] == ""
