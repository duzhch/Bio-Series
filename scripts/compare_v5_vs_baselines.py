#!/usr/bin/env python3
import argparse
from pathlib import Path
import json
import csv
import statistics as st
from collections import defaultdict
import math

# === 配置区域 ===
# 格式: '内部键名': ('CSV列名', '文件夹名', 'JSON文件名')
MODELS_CONFIG = {
    'v5':    ('v5',    'DF_GSF_v5',      'DF_GSF_v5_stats.json'),
    'bio10': ('bio10', 'bio_master_v10', 'stats.json'),
    'bio9':  ('bio9',  'bio_master_v9',  'stats.json'),
    'svr':   ('SVR',   'SVR',            'SVR_stats.json'),
    'gblup': ('GBLUP', 'GBLUP*',         '*stats.json') 
}

# 基准模型：以它的 Rep 为主，计算 Delta
ANCHOR_MODEL = 'v5' 

def read_json_stats(p: Path):
    """读取 JSON 返回 (pcc, mse)。失败返回 (None, None)"""
    try:
        obj = json.loads(p.read_text())
    except Exception:
        return None, None
    pcc = obj.get('pcc_test') or obj.get('pcc') or obj.get('PCC_TEST') or obj.get('PCC')
    mse = obj.get('mse') or obj.get('MSE')
    return pcc, mse

def safe_stats(values):
    """计算均值和标准差，忽略 None"""
    valid = [v for v in values if v is not None]
    if not valid:
        return None, None
    m = sum(valid) / len(valid)
    s = st.pstdev(valid) if len(valid) > 1 else 0.0
    return m, s

def discover(root: Path, datasets=None, traits=None):
    root = Path(root)
    ds_traits = []
    # 优先在 root/results 下找，如果没有则在 root 下找
    search_dir = root / 'results' if (root / 'results').exists() else root
    
    # 如果用户不小心指定到了具体数据集层级（补救措施）
    if (search_dir / 'DF_GSF_v5').exists(): 
        # 说明 root 本身就是一个 dataset_trait 目录
        return [(root.name, root)]

    for d in search_dir.glob('*_*'):
        if not d.is_dir(): continue
        name = d.name
        if '_' not in name: continue
        parts = name.split('_', 1)
        if len(parts) < 2: continue
        ds, trait = parts[0], parts[1]
        if datasets and ds not in datasets: continue
        if traits and trait not in traits: continue
        ds_traits.append((name, d))
    return sorted(ds_traits)

def collect_paths(base: Path):
    paths_map = defaultdict(list)
    # 按照配置收集路径
    for key, cfg in MODELS_CONFIG.items():
        folder_pattern = cfg[1]
        file_pattern = cfg[2]
        # 处理 GBLUP 可能的多种子文件夹情况
        if '*' in folder_pattern:
            # 简单处理：搜索 base 下所有匹配 folder_pattern 的目录
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
    ap = argparse.ArgumentParser(description='Robust Compare: v5 vs bio10/9/SVR/GBLUP')
    ap.add_argument('--root', required=True, help='Experiment root')
    ap.add_argument('--out', required=True, help='Output dir')
    ap.add_argument('--datasets', default=None)
    ap.add_argument('--traits', default=None)
    args = ap.parse_args()

    datasets = set([x.strip() for x in args.datasets.split(',')]) if args.datasets else None
    traits = set([x.strip() for x in args.traits.split(',')]) if args.traits else None

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 定义列顺序
    model_keys = ['v5', 'bio10', 'bio9', 'svr', 'gblup']
    
    # 生成 CSV Header
    header = ['dataset_trait', 'rep']
    # PCC Cols
    for k in model_keys: header.append(f'PCC_{MODELS_CONFIG[k][0]}')
    # Delta Cols (Baseline - Other)
    for k in model_keys:
        if k == ANCHOR_MODEL: continue
        header.append(f'Delta_{MODELS_CONFIG[ANCHOR_MODEL][0]}_minus_{MODELS_CONFIG[k][0]}')
    # MSE Cols
    for k in model_keys: header.append(f'MSE_{MODELS_CONFIG[k][0]}')

    rep_rows = []
    missing_report = []

    target_dirs = discover(root, datasets, traits)
    print(f"Found {len(target_dirs)} directories to process.")

    for name, base in target_dirs:
        paths_map = collect_paths(base)
        
        # 构建 Rep 映射
        rep_maps = {k: {get_rep_id(p): p for p in v} for k, v in paths_map.items()}
        
        # === 核心修改：Left Join ===
        # 以 v5 (ANCHOR_MODEL) 的 rep 为主。如果 v5 没有，就忽略（因为它是基准）
        anchor_reps = sorted(rep_maps[ANCHOR_MODEL].keys())
        
        if not anchor_reps:
            missing_report.append([name, 'No v5 reps found', 'Skipping'])
            continue

        for rep in anchor_reps:
            # 读取基准数据
            pcc_base, mse_base = read_json_stats(rep_maps[ANCHOR_MODEL][rep])
            if pcc_base is None: continue # 基准读取失败则跳过

            row = [name, rep]
            
            # 临时存储以便计算 Delta
            current_pccs = {ANCHOR_MODEL: pcc_base} 
            
            # 1. 填充所有 PCC
            for k in model_keys:
                if k == ANCHOR_MODEL:
                    row.append(pcc_base)
                else:
                    path = rep_maps[k].get(rep)
                    p, m = read_json_stats(path) if path else (None, None)
                    current_pccs[k] = p
                    row.append(p)
            
            # 2. 填充所有 Delta
            for k in model_keys:
                if k == ANCHOR_MODEL: continue
                val = current_pccs[k]
                if val is not None:
                    row.append(pcc_base - val)
                else:
                    row.append(None) # Delta 也是空

            # 3. 填充所有 MSE (需要重新读取一次或优化，这里简单起见重新获取或从缓存取)
            # 为了代码简单，这里重新获取非基准的 MSE
            for k in model_keys:
                if k == ANCHOR_MODEL:
                    row.append(mse_base)
                else:
                    path = rep_maps[k].get(rep)
                    _, m = read_json_stats(path) if path else (None, None)
                    row.append(m)

            rep_rows.append(row)

    # 写入 Rep CSV
    csv_file = out_dir / 'compare_rep.csv'
    with csv_file.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rep_rows)

    # 写入 Summary CSV
    # 计算统计量时忽略 None
    sum_file = out_dir / 'compare_summary.csv'
    
    # 动态生成 Summary Header
    sum_header = ['dataset_trait', 'N_v5_reps']
    for k in model_keys: 
        n = MODELS_CONFIG[k][0]
        sum_header.extend([f'PCC_{n}_mean', f'PCC_{n}_std'])
    for k in model_keys:
        if k == ANCHOR_MODEL: continue
        n = MODELS_CONFIG[k][0]
        sum_header.append(f'mean_Delta_{MODELS_CONFIG[ANCHOR_MODEL][0]}_minus_{n}')
    for k in model_keys:
        n = MODELS_CONFIG[k][0]
        sum_header.extend([f'MSE_{n}_mean', f'MSE_{n}_std'])

    # 分组聚合
    by_trait = defaultdict(list)
    for r in rep_rows:
        by_trait[r[0]].append(r)

    with sum_file.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(sum_header)
        
        for dt, rows in sorted(by_trait.items()):
            out_row = [dt, len(rows)]
            
            # 辅助函数：按列索引提取并计算
            def get_col_stats(col_name):
                try:
                    idx = header.index(col_name)
                    vals = [r[idx] for r in rows]
                    return safe_stats(vals)
                except ValueError: return None, None

            # PCC Stats
            for k in model_keys:
                m, s = get_col_stats(f'PCC_{MODELS_CONFIG[k][0]}')
                out_row.extend([f"{m:.4f}" if m else "", f"{s:.4f}" if s else ""])
            
            # Delta Means
            for k in model_keys:
                if k == ANCHOR_MODEL: continue
                m, _ = get_col_stats(f'Delta_{MODELS_CONFIG[ANCHOR_MODEL][0]}_minus_{MODELS_CONFIG[k][0]}')
                out_row.append(f"{m:.4f}" if m else "")
            
            # MSE Stats
            for k in model_keys:
                m, s = get_col_stats(f'MSE_{MODELS_CONFIG[k][0]}')
                out_row.extend([f"{m:.4f}" if m else "", f"{s:.4f}" if s else ""])

            w.writerow(out_row)

    print(f"Done. Processed {len(rep_rows)} rows.")
    if missing_report:
        print(f"Note: Some datasets were skipped (check logic if unexpected):")
        for m in missing_report: print(m)

if __name__ == '__main__':
    main()