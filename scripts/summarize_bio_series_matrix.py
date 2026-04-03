#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd


def collect_run_rows(root, model):
    root = Path(root)
    results_root = root / "results"
    rows = []

    if not results_root.exists():
        return rows

    for task_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
        if "_" not in task_dir.name:
            continue
        dataset, trait = task_dir.name.rsplit("_", 1)

        for model_dir in sorted(path for path in task_dir.iterdir() if path.is_dir()):
            if "__" not in model_dir.name:
                continue
            model_name, ablation = model_dir.name.split("__", 1)
            if model_name != model:
                continue

            for rep_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
                stats_path = rep_dir / "stats.json"
                meta_path = rep_dir / "run_meta.json"
                if not stats_path.exists():
                    continue

                stats = json.loads(stats_path.read_text())
                meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
                rows.append(
                    {
                        "dataset": dataset,
                        "trait": trait,
                        "rep": rep_dir.name,
                        "ablation": ablation,
                        "model": model_name,
                        "pcc_test": stats.get("pcc_test"),
                        "mse": stats.get("mse"),
                        "sparsity": stats.get("sparsity"),
                        "result_dir": str(rep_dir),
                        "plink_prefix": meta.get("plink_prefix"),
                    }
                )

    return rows


def build_outputs(root, out, model):
    rows = collect_run_rows(root=root, model=model)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    run_table = pd.DataFrame(rows)
    run_table_path = out / "bio_series_run_table.csv"
    if run_table.empty:
        run_table.to_csv(run_table_path, index=False)
        summary_path = out / "bio_series_trait_summary.csv"
        run_table.to_csv(summary_path, index=False)
        return run_table_path, summary_path

    run_table = run_table.sort_values(["dataset", "trait", "ablation", "rep"]).reset_index(drop=True)
    run_table.to_csv(run_table_path, index=False)

    summary = (
        run_table.groupby(["dataset", "trait", "ablation"], as_index=False)
        .agg(
            runs=("rep", "count"),
            pcc_mean=("pcc_test", "mean"),
            pcc_std=("pcc_test", "std"),
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            sparsity_mean=("sparsity", "mean"),
        )
        .sort_values(["dataset", "trait", "ablation"])
        .reset_index(drop=True)
    )
    summary_path = out / "bio_series_trait_summary.csv"
    summary.to_csv(summary_path, index=False)
    return run_table_path, summary_path


def main():
    parser = argparse.ArgumentParser(description="Summarize DF_GSF_v5 Bio-series run outputs.")
    parser.add_argument("--root", required=True, help="Runtime root that contains results/")
    parser.add_argument("--out", required=True, help="Output directory for summary CSV files")
    parser.add_argument("--model", default="bio_master_v11", help="Model name to summarize")
    args = parser.parse_args()

    run_table_path, summary_path = build_outputs(args.root, args.out, args.model)
    print(f"Wrote {run_table_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
