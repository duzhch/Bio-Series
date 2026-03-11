#!/usr/bin/env python3
import argparse
import yaml
import sys
import traceback
import importlib
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# [IMPORTS]
from src.gwas import run_gwas_pipeline
from src.features import extract_delta, extract_gene2vec, annotate_snps_with_gtf
from src.data import make_all_ids, plink_extract

def get_model_trainer(model_name):
    try:
        module_path = f"src.models.{model_name}"
        mod = importlib.import_module(module_path)
        return getattr(mod, 'train')
    except Exception as e:
        print(f"❌ Error loading model {model_name}: {e}")
        raise

def load_cfg(path: str):
    with open(path) as f: return yaml.safe_load(f)

def infer_paths(cfg, dataset, trait, rep, model_name):
    exp_root = Path(cfg['exp_root'])
    # Compatibility: If using default bio_master_v8, keep old folder name to overwrite results
    folder_name = "DF_GSF_v5" if model_name == "bio_master_v8" else model_name
    out_dir = exp_root / f"results/{dataset}_{trait}/{folder_name}/{rep}"
    split_dir = exp_root / f"data/splits/{dataset}_{trait}/{rep}"
    return out_dir, split_dir / 'train.ids', split_dir / 'test.ids'

def step_run_all(args):
    try:
        print(f"🚀 Starting Pipeline | Dataset: {args.dataset} | Trait: {args.trait} | Model: {args.model}")
        cfg = load_cfg(args.config)
        ds = cfg['datasets'][args.dataset]
        out_dir, train_ids, test_ids = infer_paths(cfg, args.dataset, args.trait, args.rep, args.model)
        out_dir.mkdir(parents=True, exist_ok=True)

        annot_out = out_dir / 'snps_annotated.tsv'
        delta_path = out_dir / 'delta_embeddings.npy'
        gene_path = out_dir / 'gene_knowledge.npy'
        fusion_prefix = out_dir / 'fusion_geno'

        # --- Step 1: Optimized GWAS Pipeline (PCA + Clumping) ---
        print(">>> Step 1: Running GWAS Pipeline (PCA & LD Clumping)...")
        # run_gwas_pipeline handles everything and returns paths to key files
        gwas_results = run_gwas_pipeline(
            plink_prefix=ds['plink'], 
            pheno_file=ds['pheno'], 
            train_ids=str(train_ids),
            trait=args.trait, 
            out_dir=str(out_dir), 
            plink_bin=cfg['resources']['plink_bin'], 
            gcta_bin=cfg['resources']['gcta_bin'],
            top_n=cfg['experiment']['top_n_snps']
        )
        # gwas_results = {'snps_for_emb': ..., 'plink_snps': ...}

        # --- Step 2: Feature Engineering (Delta) ---
        print(">>> Step 2: Generating Delta Embeddings (PigBERT)...")
        extract_delta(gwas_results['snps_for_emb'], cfg['resources']['reference_genome'], 
                      cfg['resources']['pigbert_model'], str(delta_path), device='auto')

        # --- Step 3: Feature Engineering (Gene2Vec) ---
        print(">>> Step 3: Generating Gene Annotations...")
        annotate_snps_with_gtf(gwas_results['snps_for_emb'], cfg['resources']['gtf_file'], str(annot_out))
        extract_gene2vec(str(annot_out), cfg['resources']['gene2vec_model'], str(gene_path))

        # --- Step 4: Physical Genotype Extraction ---
        print(">>> Step 4: Extracting Genotype Matrix...")
        all_ids = out_dir / 'all.ids'
        make_all_ids(str(train_ids), str(test_ids), str(all_ids))
        
        # [CRITICAL UPDATE] Use the list of INDEPENDENT SNPs from Clumping, not just simple Top N
        plink_extract(
            cfg['resources']['plink_bin'], 
            ds['plink'], 
            str(all_ids), 
            gwas_results['plink_snps'], 
            str(fusion_prefix)
        )

        # --- Step 5: Wide & Deep Training ---
        print(f">>> Step 5: Training Model ({args.model})...")
        train_fn = get_model_trainer(args.model)
        
        # Note: The 'train' function in bio_master_v8 automatically looks for 'global_pca_features.csv' 
        # inside 'out_dir', which was generated in Step 1.
        train_fn(
            plink_prefix=str(fusion_prefix), pheno_file=ds['pheno'], 
            train_ids=str(train_ids), test_ids=str(test_ids), trait=args.trait,
            delta_path=str(delta_path), gene_path=str(gene_path), out_dir=str(out_dir),
            lr=cfg['experiment']['lr'], batch_size=cfg['experiment']['batch_size'], 
            epochs=cfg['experiment']['epochs'], lambda_l1=cfg['experiment']['lambda_l1'], 
            device='auto'
        )
        print(f"\n✅ Pipeline completed successfully for {args.model}")

    except Exception as e:
        print(f"❌ CRITICAL ERROR: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

# Debug/Helper steps
def step_run_gwas(args):
    cfg = load_cfg(args.config)
    ds = cfg['datasets'][args.dataset]
    out_dir, train_ids, _ = infer_paths(cfg, args.dataset, args.trait, args.rep, args.model)
    # Allows testing GWAS/PCA independently
    run_gwas_pipeline(ds['plink'], ds['pheno'], str(train_ids), args.trait, str(out_dir), 
                      cfg['resources']['plink_bin'], cfg['resources']['gcta_bin'])

def main():
    ap = argparse.ArgumentParser(description='DF-GSF v5 Modular Launcher (Wide & Deep Ready)')
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