#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


FAM_COLUMNS = ["FID", "IID", "FatherID", "MotherID", "Sex", "Pheno_in_fam"]

ID_STRATEGIES = {
    "identity": lambda series: series.astype(str),
    "prefix_before_underscore": lambda series: series.astype(str).str.split("_").str[0],
    "suffix_after_underscore": lambda series: series.astype(str).str.split("_").str[-1],
}


def load_fam(plink_prefix: str) -> pd.DataFrame:
    fam_path = Path(f"{plink_prefix}.fam")
    if not fam_path.exists():
        raise FileNotFoundError(f"PLINK FAM file not found: {fam_path}")
    fam = pd.read_csv(fam_path, sep=r"\s+", header=None, names=FAM_COLUMNS, dtype=str)
    fam["FID"] = fam["FID"].astype(str)
    fam["IID"] = fam["IID"].astype(str)
    return fam


def _normalize_clean_ids(clean_ids) -> list[str]:
    normalized = []
    for value in clean_ids:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            normalized.append(text)
    return normalized


def map_clean_ids_to_plink_ids(plink_prefix: str, clean_ids) -> tuple[pd.DataFrame, str]:
    fam = load_fam(plink_prefix)
    requested_ids = _normalize_clean_ids(clean_ids)
    requested_unique = list(dict.fromkeys(requested_ids))
    if not requested_unique:
        return pd.DataFrame(columns=["FID", "IID", "clean_id"]), "identity"

    best_name = None
    best_mapping = None
    best_score = None
    requested_set = set(requested_unique)

    for strategy_name, strategy_fn in ID_STRATEGIES.items():
        mapping = fam[["FID", "IID"]].copy()
        mapping["clean_id"] = strategy_fn(fam["IID"]).astype(str)
        matched = mapping[mapping["clean_id"].isin(requested_set)].copy()
        missing_count = len(requested_set - set(matched["clean_id"].tolist()))
        duplicate_count = int(matched["clean_id"].duplicated(keep=False).sum())
        unique_match_count = int(matched["clean_id"].nunique())
        score = (missing_count, duplicate_count, -unique_match_count)
        if best_score is None or score < best_score:
            best_name = strategy_name
            best_mapping = mapping
            best_score = score

    if best_mapping is None:
        raise RuntimeError("Failed to build PLINK ID mapping.")

    matched = best_mapping[best_mapping["clean_id"].isin(requested_set)].copy()
    duplicates = matched[matched["clean_id"].duplicated(keep=False)]["clean_id"].unique().tolist()
    if duplicates:
        raise ValueError(
            f"Ambiguous PLINK ID mapping for clean IDs {duplicates} under strategy '{best_name}'."
        )

    ordered = pd.DataFrame({"clean_id": requested_unique}).merge(
        matched[["FID", "IID", "clean_id"]],
        on="clean_id",
        how="left",
    )
    missing = ordered[ordered["IID"].isna()]["clean_id"].tolist()
    if missing:
        raise ValueError(
            f"Could not map clean IDs {missing} to PLINK FAM IDs using strategy '{best_name}'."
        )

    ordered["FID"] = ordered["FID"].astype(str)
    ordered["IID"] = ordered["IID"].astype(str)
    return ordered[["FID", "IID", "clean_id"]], best_name


def write_mapped_ids_file(src_ids_path: str, plink_prefix: str, out_path: str) -> tuple[str, str]:
    src_df = pd.read_csv(src_ids_path, sep=r"\s+", header=None, names=["FID", "IID"], dtype=str)
    mapped_df, strategy = map_clean_ids_to_plink_ids(plink_prefix, src_df["IID"].tolist())
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    mapped_df[["FID", "IID"]].to_csv(out_path, sep="\t", index=False, header=False)
    return str(out_path), strategy


def map_pheno_ids_to_plink_ids(pheno_df: pd.DataFrame, plink_prefix: str, id_col: str) -> pd.DataFrame:
    out_df = pheno_df.copy()
    out_df[id_col] = out_df[id_col].astype(str).str.strip()
    mapped_ids, _ = map_clean_ids_to_plink_ids(plink_prefix, out_df[id_col].tolist())
    out_df.insert(0, "IID", mapped_ids["IID"].tolist())
    out_df.insert(0, "FID", mapped_ids["FID"].tolist())
    return out_df
