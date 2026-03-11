#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path
import pandas as pd


def make_all_ids(train_ids: str, test_ids: str, out_path: str) -> str:
    t1 = Path(train_ids).read_text().strip()
    t2 = Path(test_ids).read_text().strip()
    Path(out_path).write_text((t1 + '\n' + t2 + '\n').strip() + '\n')
    return out_path


def plink_extract(plink_bin: str, plink_prefix: str, keep_ids: str, extract_snps: str, out_prefix: str):
    cmd = (
        f"{plink_bin} --bfile {plink_prefix} --keep {keep_ids} --extract {extract_snps} "
        f"--make-bed --out {out_prefix} --chr-set 24 --allow-extra-chr --double-id --silent"
    )
    subprocess.run(cmd, shell=True, check=True)
    return out_prefix


def detect_id_col(pheno_file: str) -> str:
    df = pd.read_csv(pheno_file, sep='\t' if pheno_file.endswith('.tsv') else ',')
    for c in df.columns:
        if c.lower() in ['id', 'iid', 'sample_id', 'sample']:
            return c
    return df.columns[0]

