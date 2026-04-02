#!/usr/bin/env python3
import copy
import os
import re


RESOURCE_ENV_MAP = {
    "python_bin": "BIO_SERIES_PYTHON_BIN",
    "plink_bin": "BIO_SERIES_PLINK_BIN",
    "gcta_bin": "BIO_SERIES_GCTA_BIN",
    "reference_genome": "BIO_SERIES_REFERENCE_GENOME",
    "gtf_file": "BIO_SERIES_GTF_FILE",
    "pigbert_model": "BIO_SERIES_PIGBERT_MODEL",
    "gene2vec_model": "BIO_SERIES_GENE2VEC_MODEL",
}


def _dataset_env_key(dataset_name):
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", str(dataset_name).strip().upper())
    return re.sub(r"_+", "_", normalized).strip("_")


def apply_env_overrides(cfg):
    resolved = copy.deepcopy(cfg or {})

    exp_root = os.getenv("BIO_SERIES_EXP_ROOT")
    if exp_root:
        resolved["exp_root"] = exp_root

    resources = resolved.setdefault("resources", {})
    for key, env_name in RESOURCE_ENV_MAP.items():
        value = os.getenv(env_name)
        if value:
            resources[key] = value

    datasets = resolved.get("datasets", {})
    for dataset_name, dataset_cfg in datasets.items():
        env_key = _dataset_env_key(dataset_name)
        plink = os.getenv(f"BIO_SERIES_DATASET_{env_key}_PLINK")
        pheno = os.getenv(f"BIO_SERIES_DATASET_{env_key}_PHENO")
        if plink:
            dataset_cfg["plink"] = plink
        if pheno:
            dataset_cfg["pheno"] = pheno

    return resolved


def get_resource_path(cfg, key):
    return cfg.get("resources", {}).get(key)
