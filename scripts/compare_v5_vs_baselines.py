#!/usr/bin/env python3
import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

# Format: internal_key -> (csv_label, folder_name, json_filename)
MODELS_CONFIG = {
    'v5': ('v5', 'DF_GSF_v5', 'DF_GSF_v5_stats.json'),
    'bio10': ('bio10', 'bio_master_v10', 'stats.json'),
    'bio9': ('bio9', 'bio_master_v9', 'stats.json'),
    'svr': ('SVR', 'SVR', 'SVR_stats.json'),
    'gblup': ('GBLUP', 'GBLUP*', '*stats.json'),
}

ANCHOR_MODEL = 'v5'


def read_json_stats(path: Path):
    try:
        obj = json.loads(path.read_text())
    except Exception:
        return None, None
    pcc = obj.get('pcc_test') or obj.get('pcc') or obj.get('PCC_TEST') or obj.get('PCC')
    mse = obj.get('mse') or obj.get('MSE')
    return pcc, mse


def safe_stats(values):
    valid = [v for v in values if v is not None]
    if not valid:
        return None, None
    mean = sum(valid) / len(valid)
    std = st.pstdev(valid) if len(valid) > 1 else 0.0
    return mean, std


def discover(root: Path, datasets=None, traits=None):
    root = Path(root)
    ds_traits = []
    search_dir = root / 'results' if (root / 'results').exists() else root

    if (search_dir / 'DF_GSF_v5').exists():
        return [(root.name, root)]

    for directory in search_dir.glob('*_*'):
        if not directory.is_dir():
            continue
        name = directory.name
        parts = name.split('_', 1)
        if len(parts) < 2:
            continue
        ds, trait = parts[0], parts[1]
        if datasets and ds not in datasets:
            continue
        if traits and trait not in traits:
            continue
        ds_traits.append((name, directory))
    return sorted(ds_traits)


def collect_paths(base: Path):
    paths_map = defaultdict(list)
    for key, cfg in MODELS_CONFIG.items():
        folder_pattern = cfg[1]
        file_pattern = cfg[2]
        if '*' in folder_pattern:
            found_files = []
            for sub in base.glob(folder_pattern):
                if sub.is_dir():
                    found_files.extend(sub.glob(f'rep_*/{file_pattern}'))
            paths_map[key] = found_files
        else:
            paths_map[key] = list(base.glob(f'{folder_pattern}/rep_*/{file_pattern}'))
    return paths_map


def get_rep_id(path: Path):
    return path.parent.name


def main():
    ap = argparse.ArgumentParser(description='Compare v5 vs bio10/9/SVR/GBLUP')
    ap.add_argument('--root', default='.', help='Experiment root')
    ap.add_argument('--out', default='reports/compare', help='Output directory')
    ap.add_argument('--datasets', default=None)
    ap.add_argument('--traits', default=None)
    args = ap.parse_args()

    datasets = set(x.strip() for x in args.datasets.split(',')) if args.datasets else None
    traits = set(x.strip() for x in args.traits.split(',')) if args.traits else None

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_keys = ['v5', 'bio10', 'bio9', 'svr', 'gblup']
    header = ['dataset_trait', 'rep']
    for key in model_keys:
        header.append(f'PCC_{MODELS_CONFIG[key][0]}')
    for key in model_keys:
        if key == ANCHOR_MODEL:
            continue
        header.append(f'Delta_{MODELS_CONFIG[ANCHOR_MODEL][0]}_minus_{MODELS_CONFIG[key][0]}')
    for key in model_keys:
        header.append(f'MSE_{MODELS_CONFIG[key][0]}')

    rep_rows = []
    missing_report = []

    target_dirs = discover(root, datasets, traits)
    print(f"Found {len(target_dirs)} directories to process.")

    for name, base in target_dirs:
        paths_map = collect_paths(base)
        rep_maps = {key: {get_rep_id(path): path for path in paths} for key, paths in paths_map.items()}
        anchor_reps = sorted(rep_maps[ANCHOR_MODEL].keys())

        if not anchor_reps:
            missing_report.append([name, 'No v5 reps found', 'Skipping'])
            continue

        for rep in anchor_reps:
            pcc_base, mse_base = read_json_stats(rep_maps[ANCHOR_MODEL][rep])
            if pcc_base is None:
                continue

            row = [name, rep]
            current_pccs = {ANCHOR_MODEL: pcc_base}

            for key in model_keys:
                if key == ANCHOR_MODEL:
                    row.append(pcc_base)
                else:
                    path = rep_maps[key].get(rep)
                    pcc, mse = read_json_stats(path) if path else (None, None)
                    current_pccs[key] = pcc
                    row.append(pcc)

            for key in model_keys:
                if key == ANCHOR_MODEL:
                    continue
                value = current_pccs[key]
                row.append(pcc_base - value if value is not None else None)

            for key in model_keys:
                if key == ANCHOR_MODEL:
                    row.append(mse_base)
                else:
                    path = rep_maps[key].get(rep)
                    _, mse = read_json_stats(path) if path else (None, None)
                    row.append(mse)

            rep_rows.append(row)

    csv_file = out_dir / 'compare_rep.csv'
    with csv_file.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rep_rows)

    sum_file = out_dir / 'compare_summary.csv'
    sum_header = ['dataset_trait', 'N_v5_reps']
    for key in model_keys:
        label = MODELS_CONFIG[key][0]
        sum_header.extend([f'PCC_{label}_mean', f'PCC_{label}_std'])
    for key in model_keys:
        if key == ANCHOR_MODEL:
            continue
        label = MODELS_CONFIG[key][0]
        sum_header.append(f'mean_Delta_{MODELS_CONFIG[ANCHOR_MODEL][0]}_minus_{label}')
    for key in model_keys:
        label = MODELS_CONFIG[key][0]
        sum_header.extend([f'MSE_{label}_mean', f'MSE_{label}_std'])

    by_trait = defaultdict(list)
    for row in rep_rows:
        by_trait[row[0]].append(row)

    with sum_file.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(sum_header)

        for dataset_trait, rows in sorted(by_trait.items()):
            out_row = [dataset_trait, len(rows)]

            def get_col_stats(col_name):
                try:
                    idx = header.index(col_name)
                except ValueError:
                    return None, None
                values = [row[idx] for row in rows]
                return safe_stats(values)

            for key in model_keys:
                mean, std = get_col_stats(f'PCC_{MODELS_CONFIG[key][0]}')
                out_row.extend([mean, std])

            for key in model_keys:
                if key == ANCHOR_MODEL:
                    continue
                mean, _ = get_col_stats(f'Delta_{MODELS_CONFIG[ANCHOR_MODEL][0]}_minus_{MODELS_CONFIG[key][0]}')
                out_row.append(mean)

            for key in model_keys:
                mean, std = get_col_stats(f'MSE_{MODELS_CONFIG[key][0]}')
                out_row.extend([mean, std])

            writer.writerow(out_row)

    if missing_report:
        missing_file = out_dir / 'missing_report.csv'
        with missing_file.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['dataset_trait', 'status', 'note'])
            writer.writerows(missing_report)

    print(f"Wrote report files to {out_dir}")


if __name__ == '__main__':
    main()
