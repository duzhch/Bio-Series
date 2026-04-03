from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd


def load_stage_runtime_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "stage_bio_series_runtime.py"
    spec = spec_from_file_location("df_gsf_v5_stage_runtime", module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_stage_runtime_creates_launcher_compatible_split_tree(tmp_path):
    stage_runtime = load_stage_runtime_module()

    cv_root = tmp_path / "02_cv_datasets"
    fold_dir = cv_root / "Amercian_duroc" / "LMA" / "fold_1"
    fold_dir.mkdir(parents=True)
    (fold_dir / "train_ids.txt").write_text("1\t1\n2\t2\n")
    (fold_dir / "test_ids.txt").write_text("3\t3\n")

    processed_root = tmp_path / "01_processed_data"
    fam_path = processed_root / "Amercian_duroci" / "Amercian_duroic_QCed.fam"
    fam_path.parent.mkdir(parents=True, exist_ok=True)
    fam_path.write_text(
        "0\t2_1\t0\t0\t0\t-9\n"
        "0\t5_2\t0\t0\t0\t-9\n"
        "0\t9_3\t0\t0\t0\t-9\n"
    )

    runtime_root = tmp_path / "runtime"
    stage_runtime.stage_runtime(
        cv_root=cv_root,
        runtime_root=runtime_root,
        dataset_map={
            "American_Duroc": {
                "cv_name": "Amercian_duroc",
                "traits": ["LMA"],
                "processed_name": "Amercian_duroci",
                "plink_stem": "Amercian_duroic_QCed",
            }
        },
        processed_root=processed_root,
        folds=[1],
    )

    train_ids = runtime_root / "data" / "splits" / "American_Duroc_LMA" / "rep_01" / "train.ids"
    test_ids = runtime_root / "data" / "splits" / "American_Duroc_LMA" / "rep_01" / "test.ids"

    assert train_ids.exists()
    assert test_ids.exists()
    assert train_ids.read_text() == "0\t2_1\n0\t5_2\n"
    assert test_ids.read_text() == "0\t9_3\n"


def test_stage_runtime_writes_manifest(tmp_path):
    stage_runtime = load_stage_runtime_module()

    cv_root = tmp_path / "02_cv_datasets"
    fold_dir = cv_root / "lwp2" / "AGE" / "fold_2"
    fold_dir.mkdir(parents=True)
    (fold_dir / "train_ids.txt").write_text("10\t10\n")
    (fold_dir / "test_ids.txt").write_text("20\t20\n")

    processed_root = tmp_path / "01_processed_data"
    fam_path = processed_root / "lwp2" / "lwp2_QCed.fam"
    fam_path.parent.mkdir(parents=True, exist_ok=True)
    fam_path.write_text(
        "0\t10_10\t0\t0\t0\t-9\n"
        "0\t20_20\t0\t0\t0\t-9\n"
    )

    runtime_root = tmp_path / "runtime"
    manifest = stage_runtime.stage_runtime(
        cv_root=cv_root,
        runtime_root=runtime_root,
        dataset_map={
            "LargeWhite_Pop2": {
                "cv_name": "lwp2",
                "traits": ["AGE"],
                "processed_name": "lwp2",
                "plink_stem": "lwp2_QCed",
            }
        },
        processed_root=processed_root,
        folds=[2],
    )

    assert manifest["runtime_root"] == str(runtime_root)
    assert manifest["datasets"][0]["dataset_key"] == "LargeWhite_Pop2"
    assert manifest["datasets"][0]["traits"] == ["AGE"]
    manifest_path = runtime_root / "manifests" / "bio_series_runtime_manifest.json"
    assert manifest_path.exists()


def test_id_mapping_maps_clean_ids_and_phenotypes_to_plink_ids(tmp_path):
    module_path = Path(__file__).resolve().parents[1] / "src" / "id_mapping.py"
    spec = spec_from_file_location("df_gsf_v5_id_mapping", module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    plink_prefix = tmp_path / "demo_prefix"
    fam_path = plink_prefix.with_suffix(".fam")
    fam_path.write_text(
        "0\t100_1\t0\t0\t0\t-9\n"
        "0\t200_2\t0\t0\t0\t-9\n"
        "0\t300_3\t0\t0\t0\t-9\n"
    )

    keep_df, strategy = module.map_clean_ids_to_plink_ids(
        plink_prefix=str(plink_prefix),
        clean_ids=["1", "3"],
    )
    assert strategy == "suffix_after_underscore"
    assert keep_df.to_dict("records") == [
        {"FID": "0", "IID": "100_1", "clean_id": "1"},
        {"FID": "0", "IID": "300_3", "clean_id": "3"},
    ]

    pheno_df = pd.DataFrame({"ID": ["1", "3"], "LMA": [39.1, 35.5]})
    mapped = module.map_pheno_ids_to_plink_ids(
        pheno_df=pheno_df,
        plink_prefix=str(plink_prefix),
        id_col="ID",
    )
    assert list(mapped.columns)[:3] == ["FID", "IID", "ID"]
    assert mapped["FID"].tolist() == ["0", "0"]
    assert mapped["IID"].tolist() == ["100_1", "300_3"]
