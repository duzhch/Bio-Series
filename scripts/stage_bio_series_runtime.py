#!/usr/bin/env python3
import argparse
import json
import shutil
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CURRENT_DIR))

from src.id_mapping import write_mapped_ids_file


DEFAULT_DATASET_MAP = {
    "American_Duroc": {
        "cv_name": "Amercian_duroc",
        "traits": ["LMA", "LMD"],
        "processed_name": "Amercian_duroci",
        "plink_stem": "Amercian_duroic_QCed",
    },
    "Canadian_Duroc": {
        "cv_name": "Canadian_duroc",
        "traits": ["LMA", "LMD"],
        "processed_name": "Canadian_duroic",
        "plink_stem": "Canadian_duroic_QCed",
    },
    "LargeWhite_Pop1": {
        "cv_name": "lwp1",
        "traits": ["AGE", "BF"],
        "processed_name": "lwp1",
        "plink_stem": "lwp1_QCed",
    },
    "LargeWhite_Pop1_Repro": {
        "cv_name": "lwp1",
        "traits": ["LW", "NBA", "TNB"],
        "processed_name": "lwp1",
        "plink_stem": "lwp1_QCed",
    },
    "LargeWhite_Pop2": {
        "cv_name": "lwp2",
        "traits": ["AGE"],
        "processed_name": "lwp2",
        "plink_stem": "lwp2_QCed",
    },
    "LargeWhite_Pop2_BF": {
        "cv_name": "lwp2",
        "traits": ["BF"],
        "processed_name": "lwp2",
        "plink_stem": "lwp2_QCed",
    },
}


def parse_csv_list(raw_value):
    if not raw_value:
        return []
    return [item.strip() for item in str(raw_value).split(",") if item.strip()]


def ensure_runtime_dirs(runtime_root):
    runtime_root = Path(runtime_root)
    for rel in [
        "data/splits",
        "manifests",
        "slurm_jobs/scripts_v5",
        "slurm_jobs/logs_v5",
        "results",
        "summaries",
    ]:
        (runtime_root / rel).mkdir(parents=True, exist_ok=True)


def stage_runtime(cv_root, runtime_root, dataset_map=None, folds=None, processed_root=None):
    cv_root = Path(cv_root)
    runtime_root = Path(runtime_root)
    dataset_map = dataset_map or DEFAULT_DATASET_MAP
    processed_root = Path(processed_root) if processed_root else cv_root.parent / "01_processed_data"
    folds = sorted(set(folds or range(1, 11)))

    ensure_runtime_dirs(runtime_root)

    manifest = {
        "cv_root": str(cv_root),
        "runtime_root": str(runtime_root),
        "folds": folds,
        "datasets": [],
    }

    for dataset_key, info in dataset_map.items():
        cv_name = info["cv_name"]
        traits = list(info["traits"])
        processed_name = info.get("processed_name", cv_name)
        plink_stem = info.get("plink_stem")
        if not plink_stem:
            fam_candidates = sorted((processed_root / processed_name).glob("*_QCed.fam"))
            if len(fam_candidates) != 1:
                raise ValueError(
                    f"Could not infer unique *_QCed.fam under {(processed_root / processed_name)}. "
                    f"Please set 'plink_stem' explicitly in dataset_map."
                )
            plink_stem = fam_candidates[0].stem
        plink_prefix = processed_root / processed_name / plink_stem
        dataset_entry = {
            "dataset_key": dataset_key,
            "cv_name": cv_name,
            "traits": traits,
            "plink_prefix": str(plink_prefix),
            "runs": [],
        }

        for trait in traits:
            for fold in folds:
                src_dir = cv_root / cv_name / trait / f"fold_{fold}"
                train_src = src_dir / "train_ids.txt"
                test_src = src_dir / "test_ids.txt"
                if not train_src.exists() or not test_src.exists():
                    raise FileNotFoundError(
                        f"Missing split files under {src_dir}. "
                        f"Expected {train_src.name} and {test_src.name}."
                    )

                rep_name = f"rep_{fold:02d}"
                dst_dir = runtime_root / "data" / "splits" / f"{dataset_key}_{trait}" / rep_name
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(train_src, dst_dir / "train.clean.ids")
                shutil.copy2(test_src, dst_dir / "test.clean.ids")
                write_mapped_ids_file(train_src, str(plink_prefix), dst_dir / "train.ids")
                write_mapped_ids_file(test_src, str(plink_prefix), dst_dir / "test.ids")
                dataset_entry["runs"].append(
                    {
                        "trait": trait,
                        "fold": fold,
                        "rep": rep_name,
                        "source_dir": str(src_dir),
                        "target_dir": str(dst_dir),
                    }
                )

        manifest["datasets"].append(dataset_entry)

    manifest_path = runtime_root / "manifests" / "bio_series_runtime_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Stage Bio-series CV split files into the DF_GSF_v5 runtime layout."
    )
    parser.add_argument("--cv-root", required=True, help="Path to Bio-series-data/02_cv_datasets")
    parser.add_argument("--runtime-root", required=True, help="Runtime root used as exp_root")
    parser.add_argument(
        "--processed-root",
        default="",
        help="Optional path to Bio-series-data/01_processed_data. Defaults to sibling of cv-root.",
    )
    parser.add_argument(
        "--datasets",
        default="",
        help="Optional comma-separated dataset keys to stage. Default stages all supported datasets.",
    )
    parser.add_argument(
        "--folds",
        default="",
        help="Optional comma-separated fold numbers such as '1,2,3'. Default stages folds 1..10.",
    )
    args = parser.parse_args()

    selected_dataset_keys = parse_csv_list(args.datasets)
    selected_folds = [int(value) for value in parse_csv_list(args.folds)] if args.folds else None

    dataset_map = DEFAULT_DATASET_MAP
    if selected_dataset_keys:
        unknown = sorted(set(selected_dataset_keys) - set(DEFAULT_DATASET_MAP))
        if unknown:
            raise ValueError(
                f"Unsupported dataset keys: {unknown}. "
                f"Supported keys: {sorted(DEFAULT_DATASET_MAP)}"
            )
        dataset_map = {key: DEFAULT_DATASET_MAP[key] for key in selected_dataset_keys}

    manifest = stage_runtime(
        cv_root=args.cv_root,
        runtime_root=args.runtime_root,
        dataset_map=dataset_map,
        folds=selected_folds,
        processed_root=args.processed_root or None,
    )
    print(
        json.dumps(
            {
                "runtime_root": manifest["runtime_root"],
                "folds": manifest["folds"],
                "dataset_count": len(manifest["datasets"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
