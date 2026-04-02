from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import yaml


def load_config_utils_module():
    module_path = Path(__file__).resolve().parents[1] / "src" / "config_utils.py"
    spec = spec_from_file_location("df_gsf_v5_config_utils", module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_apply_env_overrides_rewrites_machine_specific_fields(monkeypatch):
    config_utils = load_config_utils_module()
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

    resolved = config_utils.apply_env_overrides(cfg)

    assert resolved["exp_root"] == "/env/root"
    assert resolved["resources"]["reference_genome"] == "/env/ref.fa"
    assert resolved["datasets"]["pig"]["plink"] == "/env/pig/plink"
    assert resolved["datasets"]["pig"]["pheno"] == "/old/pig/pheno.csv"


def test_get_resource_path_falls_back_to_yaml_value():
    config_utils = load_config_utils_module()
    cfg = {"resources": {"plink_bin": "plink"}}
    assert config_utils.get_resource_path(cfg, "plink_bin") == "plink"


def test_public_template_includes_required_top_level_sections():
    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "config" / "examples" / "public_template.yaml").read_text()
    )
    assert "exp_root" in cfg
    assert "resources" in cfg
    assert "datasets" in cfg
    assert "experiment" in cfg
    assert "slurm" in cfg


def test_public_template_contains_no_lab_specific_absolute_home_paths():
    text = (
        Path(__file__).resolve().parents[1] / "config" / "examples" / "public_template.yaml"
    ).read_text()
    assert "/work/home/" not in text
    assert "zyqgroup" not in text


def test_gitignore_covers_python_cache_and_results():
    text = (Path(__file__).resolve().parents[1] / ".gitignore").read_text()
    assert "__pycache__/" in text
    assert "*.pyc" in text
    assert "results/" in text
    assert "slurm_jobs/" in text
    assert "tests/.tmp_bio_master_v11_verify/" in text


def test_project_structure_doc_mentions_mainline_and_archival_content():
    text = (Path(__file__).resolve().parents[1] / "docs" / "project_structure.md").read_text()
    assert "Mainline" in text
    assert "Archived or supporting content" in text


def test_readme_documents_real_run_all_command():
    text = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    assert "run-all" in text
    assert "--config" in text
    assert "--dataset" in text
    assert "--trait" in text
    assert "--rep" in text
    assert "--model" in text


def test_readme_documents_ablation_modes():
    text = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    assert "no_delta" in text
    assert "no_gene2vec" in text
    assert "no_bio_prior" in text
    assert "no_pca" in text
    assert "pca_only_prior_off" in text
