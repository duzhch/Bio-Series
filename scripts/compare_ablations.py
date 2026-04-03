#!/usr/bin/env python3
import argparse
import csv
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path


def first_present(mapping, keys):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def coerce_metric(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value or value.lower() in {"na", "n/a", "nan", "none", "null"}:
            return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def read_stats(path):
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {"pcc": None, "mse": None}

    pcc = first_present(payload, ["pcc_test", "pcc", "PCC_TEST", "PCC"])
    mse = first_present(payload, ["mse", "mse_test", "MSE", "MSE_TEST"])
    return {
        "pcc": coerce_metric(pcc),
        "mse": coerce_metric(mse),
    }


def safe_stats(values):
    valid = [value for value in values if value not in ("", None)]
    if not valid:
        return None, None
    if len(valid) == 1:
        return valid[0], 0.0
    return sum(valid) / len(valid), st.pstdev(valid)


def format_summary_value(value):
    return f"{value:.4f}" if value is not None else ""


def stats_path(base_dir, model, ablation, rep):
    return base_dir / f"{model}__{ablation}" / rep / "stats.json"


def split_model_ablation(dirname, model):
    prefix = f"{model}__"
    if not dirname.startswith(prefix):
        return None
    ablation = dirname[len(prefix) :]
    return ablation or None


def discover_dataset_dirs(root, model):
    root = Path(root)
    search_root = root / "results" if (root / "results").exists() else root

    try:
        children = sorted(path for path in search_root.iterdir() if path.is_dir())
    except FileNotFoundError:
        return []

    dataset_dirs = []
    has_direct_model_dirs = False
    for child in children:
        if split_model_ablation(child.name, model):
            has_direct_model_dirs = True
            continue
        try:
            has_model_dir = any(
                grandchild.is_dir() and split_model_ablation(grandchild.name, model)
                for grandchild in child.iterdir()
            )
        except OSError:
            has_model_dir = False
        if has_model_dir:
            dataset_dirs.append((child.name, child))
    if dataset_dirs:
        return dataset_dirs
    if has_direct_model_dirs:
        return [(search_root.name, search_root)]
    return dataset_dirs


def discover_ablations(base_dir, model):
    ablations = set()
    try:
        children = sorted(path for path in base_dir.iterdir() if path.is_dir())
    except OSError:
        return []

    for child in children:
        ablation = split_model_ablation(child.name, model)
        if ablation:
            ablations.add(ablation)
    return ordered_ablations(ablations)


def ordered_ablations(ablations):
    names = sorted(name for name in ablations if name != "full")
    return (["full"] if "full" in ablations else []) + names


def build_rep_header(ablations):
    return (
        ["dataset_trait", "rep"]
        + [f"PCC_{ablation}" for ablation in ablations]
        + [f"Delta_full_minus_{ablation}" for ablation in ablations if ablation != "full"]
        + [f"MSE_{ablation}" for ablation in ablations]
    )


def build_summary_header(ablations):
    return (
        ["dataset_trait", "N_full_reps"]
        + [f"N_{ablation}_reps" for ablation in ablations if ablation != "full"]
        + [
            f"N_paired_full_vs_{ablation}_reps"
            for ablation in ablations
            if ablation != "full"
        ]
        + [
            column
            for ablation in ablations
            for column in (f"PCC_{ablation}_mean", f"PCC_{ablation}_std")
        ]
        + [f"mean_Delta_full_minus_{ablation}" for ablation in ablations if ablation != "full"]
        + [
            column
            for ablation in ablations
            for column in (f"MSE_{ablation}_mean", f"MSE_{ablation}_std")
        ]
    )


def build_rep_row(dataset_trait, rep, ablations, pccs, mses):
    row = [dataset_trait, rep]
    row.extend(pccs.get(ablation, "") if pccs.get(ablation) is not None else "" for ablation in ablations)
    for ablation in ablations:
        if ablation == "full":
            continue
        value = pccs.get(ablation)
        row.append(pccs["full"] - value if value is not None else "")
    row.extend(mses.get(ablation, "") if mses.get(ablation) is not None else "" for ablation in ablations)
    return row


def collect_rep_rows(dataset_dirs, model, ablations):
    rows = []
    for dataset_trait, base_dir in dataset_dirs:
        if "full" not in discover_ablations(base_dir, model):
            continue

        full_dir = base_dir / f"{model}__full"
        for rep_dir in sorted(path for path in full_dir.glob("rep*") if path.is_dir()):
            rep = rep_dir.name
            full_stats = read_stats(stats_path(base_dir, model, "full", rep))
            if full_stats["pcc"] is None:
                continue

            pccs = {"full": full_stats["pcc"]}
            mses = {"full": full_stats["mse"]}
            for ablation in ablations:
                if ablation == "full":
                    continue
                stats = read_stats(stats_path(base_dir, model, ablation, rep))
                pccs[ablation] = stats["pcc"]
                mses[ablation] = stats["mse"]
            rows.append(build_rep_row(dataset_trait, rep, ablations, pccs, mses))
    return rows


def build_summary_rows(rep_rows, rep_header, ablations):
    grouped = defaultdict(list)
    for row in rep_rows:
        grouped[row[0]].append(row)

    header_index = {name: idx for idx, name in enumerate(rep_header)}
    summary_rows = []
    for dataset_trait in sorted(grouped):
        rows = grouped[dataset_trait]
        summary_row = [dataset_trait, len(rows)]

        for ablation in ablations:
            if ablation == "full":
                continue
            n_valid = sum(
                1
                for row in rows
                if row[header_index[f"PCC_{ablation}"]] not in ("", None)
            )
            summary_row.append(n_valid)

        for ablation in ablations:
            if ablation == "full":
                continue
            n_paired = sum(
                1
                for row in rows
                if row[header_index[f"Delta_full_minus_{ablation}"]] not in ("", None)
            )
            summary_row.append(n_paired)

        for ablation in ablations:
            mean, std = safe_stats([row[header_index[f"PCC_{ablation}"]] for row in rows])
            summary_row.extend([format_summary_value(mean), format_summary_value(std)])

        for ablation in ablations:
            if ablation == "full":
                continue
            mean, _std = safe_stats([row[header_index[f"Delta_full_minus_{ablation}"]] for row in rows])
            summary_row.append(format_summary_value(mean))

        for ablation in ablations:
            mean, std = safe_stats([row[header_index[f"MSE_{ablation}"]] for row in rows])
            summary_row.extend([format_summary_value(mean), format_summary_value(std)])

        summary_rows.append(summary_row)

    return summary_rows


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare model ablations against the full anchor.")
    parser.add_argument("--root", required=True, help="Experiment root or results root")
    parser.add_argument("--out", required=True, help="Directory for CSV outputs")
    parser.add_argument("--model", required=True, help="Model folder prefix, e.g. bio_master_v11")
    args = parser.parse_args(argv)

    dataset_dirs = discover_dataset_dirs(Path(args.root), args.model)
    discovered_ablations = set()
    for _dataset_trait, base_dir in dataset_dirs:
        discovered_ablations.update(discover_ablations(base_dir, args.model))

    ablations = ordered_ablations(discovered_ablations)
    rep_header = build_rep_header(ablations)
    summary_header = build_summary_header(ablations)
    rep_rows = collect_rep_rows(dataset_dirs, args.model, ablations)
    summary_rows = build_summary_rows(rep_rows, rep_header, ablations)

    out_dir = Path(args.out)
    write_csv(out_dir / "ablation_compare_rep.csv", rep_header, rep_rows)
    write_csv(out_dir / "ablation_compare_summary.csv", summary_header, summary_rows)


if __name__ == "__main__":
    main()
