#!/usr/bin/env python3
import argparse
import importlib
import os
import sys
import traceback
from pathlib import Path

import yaml

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from src.data import make_all_ids, plink_extract
from src.features import annotate_snps_with_gtf, extract_delta, extract_gene2vec
from src.gwas import run_gwas_pipeline

PATHLIKE_RESOURCE_KEYS = {
    'reference_genome',
    'gtf_file',
    'pigbert_model',
    'gene2vec_model',
}


def get_model_trainer(model_name):
    try:
        module_path = f"src.models.{model_name}"
        mod = importlib.import_module(module_path)
        return getattr(mod, 'train')
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        raise


def _resolve_path(base_dir: Path, value: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def load_cfg(path: str):
    cfg_path = Path(path).resolve()
    repo_root = cfg_path.parent.parent

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    cfg['exp_root'] = _resolve_path(repo_root, cfg.get('exp_root', '.'))

    resources = cfg.get('resources', {})
    for key in PATHLIKE_RESOURCE_KEYS:
        if key in resources:
            resources[key] = _resolve_path(repo_root, resources[key])

    python_bin = resources.get('python_bin')
    if isinstance(python_bin, str) and any(ch in python_bin for ch in ('/', '~', '.')):
        resources['python_bin'] = _resolve_path(repo_root, python_bin)

    for dataset_cfg in cfg.get('datasets', {}).values():
        for key in ('plink', 'pheno'):
            if key in dataset_cfg:
                dataset_cfg[key] = _resolve_path(repo_root, dataset_cfg[key])

    return cfg


def _ensure_exists(path_str: str, label: str):
    if not Path(path_str).exists():
        raise FileNotFoundError(f"Missing {label}: {path_str}")


def _ensure_plink_prefix(prefix: str, label: str):
    missing = [f"{prefix}{suffix}" for suffix in ('.bed', '.bim', '.fam') if not Path(f"{prefix}{suffix}").exists()]
    if missing:
        raise FileNotFoundError(f"Missing {label} PLINK files: {missing}")


def _validate_common_inputs(ds, train_ids: str, test_ids: str | None = None):
    _ensure_plink_prefix(ds['plink'], 'dataset')
    _ensure_exists(ds['pheno'], 'phenotype file')
    _ensure_exists(train_ids, 'train split file')
    if test_ids is not None:
        _ensure_exists(test_ids, 'test split file')


def _validate_feature_inputs(cfg):
    resources = cfg['resources']
    _ensure_exists(resources['reference_genome'], 'reference genome')
    _ensure_exists(resources['gtf_file'], 'GTF annotation')
    _ensure_exists(resources['pigbert_model'], 'PigBERT model directory')
    _ensure_exists(resources['gene2vec_model'], 'Gene2Vec model file')


def infer_paths(cfg, dataset, trait, rep, model_name):
    exp_root = Path(cfg['exp_root'])
    folder_name = "DF_GSF_v5" if model_name == "bio_master_v8" else model_name
    out_dir = exp_root / f"results/{dataset}_{trait}/{folder_name}/{rep}"
    split_dir = exp_root / f"data/splits/{dataset}_{trait}/{rep}"
    return out_dir, split_dir / 'train.ids', split_dir / 'test.ids'


def step_run_all(args):
    try:
        print(f"Starting pipeline | dataset={args.dataset} trait={args.trait} model={args.model}")
        cfg = load_cfg(args.config)
        ds = cfg['datasets'][args.dataset]
        out_dir, train_ids, test_ids = infer_paths(cfg, args.dataset, args.trait, args.rep, args.model)
        out_dir.mkdir(parents=True, exist_ok=True)

        _validate_common_inputs(ds, str(train_ids), str(test_ids))

        annot_out = out_dir / 'snps_annotated.tsv'
        delta_path = out_dir / 'delta_embeddings.npy'
        gene_path = out_dir / 'gene_knowledge.npy'
        fusion_prefix = out_dir / 'fusion_geno'

        print("Step 1/5: GWAS + PCA + LD clumping")
        gwas_results = run_gwas_pipeline(
            plink_prefix=ds['plink'],
            pheno_file=ds['pheno'],
            train_ids=str(train_ids),
            trait=args.trait,
            out_dir=str(out_dir),
            plink_bin=cfg['resources']['plink_bin'],
            gcta_bin=cfg['resources']['gcta_bin'],
            top_n=cfg['experiment']['top_n_snps'],
        )

        print("Step 2/5: Delta embeddings")
        _validate_feature_inputs(cfg)
        extract_delta(
            gwas_results['snps_for_emb'],
            cfg['resources']['reference_genome'],
            cfg['resources']['pigbert_model'],
            str(delta_path),
            device='auto',
        )

        print("Step 3/5: GTF annotation + Gene2Vec")
        annotate_snps_with_gtf(gwas_results['snps_for_emb'], cfg['resources']['gtf_file'], str(annot_out))
        extract_gene2vec(str(annot_out), cfg['resources']['gene2vec_model'], str(gene_path))

        print("Step 4/5: PLINK extraction")
        all_ids = out_dir / 'all.ids'
        make_all_ids(str(train_ids), str(test_ids), str(all_ids))
        plink_extract(
            cfg['resources']['plink_bin'],
            ds['plink'],
            str(all_ids),
            gwas_results['plink_snps'],
            str(fusion_prefix),
        )

        print(f"Step 5/5: Model training ({args.model})")
        train_fn = get_model_trainer(args.model)
        train_fn(
            plink_prefix=str(fusion_prefix),
            pheno_file=ds['pheno'],
            train_ids=str(train_ids),
            test_ids=str(test_ids),
            trait=args.trait,
            delta_path=str(delta_path),
            gene_path=str(gene_path),
            out_dir=str(out_dir),
            lr=cfg['experiment']['lr'],
            batch_size=cfg['experiment']['batch_size'],
            epochs=cfg['experiment']['epochs'],
            lambda_l1=cfg['experiment']['lambda_l1'],
            device='auto',
        )
        print(f"Pipeline completed successfully for {args.model}")

    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


def step_run_gwas(args):
    cfg = load_cfg(args.config)
    ds = cfg['datasets'][args.dataset]
    out_dir, train_ids, _ = infer_paths(cfg, args.dataset, args.trait, args.rep, args.model)
    _validate_common_inputs(ds, str(train_ids))
    run_gwas_pipeline(
        ds['plink'],
        ds['pheno'],
        str(train_ids),
        args.trait,
        str(out_dir),
        cfg['resources']['plink_bin'],
        cfg['resources']['gcta_bin'],
    )


def main():
    ap = argparse.ArgumentParser(description='DF-GSF v5 modular launcher')
    sub = ap.add_subparsers(dest='cmd', required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--config', required=True)
    common.add_argument('--dataset', required=True)
    common.add_argument('--trait', required=True)
    common.add_argument('--rep', required=True)
    common.add_argument('--model', default='bio_master_v8')

    p1 = sub.add_parser('run-all', parents=[common])
    p1.set_defaults(func=step_run_all)

    p2 = sub.add_parser('run-gwas', parents=[common])
    p2.set_defaults(func=step_run_gwas)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
