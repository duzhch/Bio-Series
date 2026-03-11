#!/usr/bin/env python3
import os
import sys
import gzip
import time
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import pysam
import sentencepiece as spm
from transformers import AutoModelForMaskedLM
try:
    from intervaltree import IntervalTree
except ImportError:
    print("Installing missing dependency: intervaltree")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "intervaltree"])
    from intervaltree import IntervalTree

# ==========================================
# Device Utils
# ==========================================
def get_best_device(device_hint='auto'):
    if device_hint == 'cpu': return 'cpu'
    return 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# 1. Delta Engine (PigBERT) - 保持不变
# ==========================================
class _DeltaEngine:
    def __init__(self, ref_fa: str, model_dir: str, device: str = 'auto'):
        print(f"Initializing DeltaEngine...")
        self.device = get_best_device(device)
        if not os.path.exists(ref_fa + '.fai'): pysam.faidx(ref_fa)
        self.fasta = pysam.FastaFile(ref_fa)
        spm_paths = list(Path(model_dir).rglob('*.model'))
        if not spm_paths: raise FileNotFoundError(f"SPM model not found in {model_dir}")
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(spm_paths[0]))
        try:
            self.model = AutoModelForMaskedLM.from_pretrained(model_dir).to(self.device).eval()
        except:
            self.device = 'cpu'
            self.model = AutoModelForMaskedLM.from_pretrained(model_dir).to('cpu').eval()

    def _paired(self, snps_df: pd.DataFrame, window: int = 512):
        half = window // 2
        ref, alt = [], []
        for _, r in snps_df.iterrows():
            chrom = str(r['Chr'])
            pos = int(r['bp'])
            a1 = str(r['A1'])
            start = max(0, pos - half - 1)
            end = pos + half
            try:
                ctx = self.fasta.fetch(chrom, start, end).upper()
                if len(ctx) < window: ctx = ctx.ljust(window, 'N')
                seq = list(ctx)
                seq[half] = a1
                ref.append(ctx)
                alt.append(''.join(seq))
            except:
                ref.append('N' * window); alt.append('N' * window)
        return ref, alt

    def _encode_ids(self, seqs, max_len=512):
        ids_list = []
        cls = self.sp.piece_to_id('[CLS]')
        sep = self.sp.piece_to_id('[SEP]')
        pad = self.sp.piece_to_id('[PAD]')
        for s in seqs:
            ids = [cls] + self.sp.encode(s, out_type=int) + [sep]
            if len(ids) > max_len: ids = ids[:max_len]
            else: ids = ids + [pad] * (max_len - len(ids))
            ids_list.append(ids)
        return torch.tensor(ids_list).to(self.device)

    def compute(self, snps_csv: str, out_file: str):
        print(f"Computing delta embeddings for {snps_csv}")
        df = pd.read_csv(snps_csv)
        ref, alt = self._paired(df)
        bs = 16 if self.device != 'cpu' else 4
        outs = []
        for i in range(0, len(ref), bs):
            r = self._encode_ids(ref[i:i+bs])
            a = self._encode_ids(alt[i:i+bs])
            with torch.no_grad():
                er = self.model(r, output_hidden_states=True).hidden_states[-1][:, 0, :]
                ea = self.model(a, output_hidden_states=True).hidden_states[-1][:, 0, :]
                outs.append((ea - er).cpu().numpy())
        emb = np.vstack(outs)
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        np.save(out_file, emb)
        return out_file

def extract_delta(snps_csv: str, ref_genome: str, pigbert_model_dir: str, out_file: str, device: str = 'auto'):
    eng = _DeltaEngine(ref_genome, pigbert_model_dir, device)
    return eng.compute(snps_csv, out_file)

# ==========================================
# 2. GTF Annotation Engine (New & Optimized)
# ==========================================
def _load_gtf_optimized(gtf_path):
    """
    Parses GTF to IntervalTrees. 
    Optimization: Only reads 'gene' rows, skips rigorous parsing of non-essential attributes.
    """
    print(f"Loading GTF structure from {gtf_path}...")
    trees = {} 
    opener = gzip.open if gtf_path.endswith('.gz') else open
    
    # Pre-compile split logic for speed
    count = 0
    with opener(gtf_path, 'rt') as f:
        for line in f:
            if line.startswith('#'): continue
            # Fast filter: only process if it contains 'gene' feature
            # 3rd column is usually index 2. Let's do a quick split check.
            # Using maxsplit to avoid parsing the whole attributes string immediately
            parts = line.split('\t', 8) 
            if len(parts) < 9: continue
            
            if parts[2] != 'gene': continue
            
            chrom = parts[0]
            start = int(parts[3])
            end = int(parts[4])
            attr_str = parts[8]
            
            # Fast Attribute Parsing
            gid, gname = None, None
            # Heuristic parsing (faster than generic parser)
            if 'gene_id "' in attr_str:
                gid = attr_str.split('gene_id "')[1].split('"')[0]
            if 'gene_name "' in attr_str:
                gname = attr_str.split('gene_name "')[1].split('"')[0]
            
            if not gid: continue
            
            if chrom not in trees: trees[chrom] = IntervalTree()
            trees[chrom][start:end] = {'id': gid, 'name': gname or gid}
            count += 1
            
    print(f"  Loaded {count} genes across {len(trees)} chromosomes.")
    return trees

def annotate_snps_with_gtf(snps_csv: str, gtf_path: str, out_tsv: str):
    """
    Dynamically maps SNPs to Genes using GTF.
    Replaces the old 'filter_annotation_by_coord'.
    """
    print(f"Annotating SNPs using GTF...")
    print(f"  SNPs: {snps_csv}")
    print(f"  GTF:  {gtf_path}")
    
    trees = _load_gtf_optimized(gtf_path)
    df = pd.read_csv(snps_csv)
    
    results = []
    hits = 0
    
    for _, row in df.iterrows():
        chrom = str(row['Chr'])
        pos = int(row['bp'])
        snp_id = row['SNP']
        
        gid, gname = "Intergenic", "."
        
        if chrom in trees:
            overlaps = trees[chrom].at(pos)
            if overlaps:
                # Take the first overlap (or shortest? keeping it simple for speed)
                match = list(overlaps)[0].data
                gid = match['id']
                gname = match['name']
                hits += 1
        
        results.append({
            'SNP': snp_id, 'Chr': chrom, 'bp': pos,
            'GeneID': gid, 'GeneName': gname
        })
        
    res_df = pd.DataFrame(results)
    Path(out_tsv).parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_tsv, sep='\t', index=False)
    print(f"  Annotation done. {hits}/{len(res_df)} SNPs in genes. Saved to {out_tsv}")
    return out_tsv

# ==========================================
# 3. Gene2Vec Extraction (Robust)
# ==========================================
def _load_word2vec_format(path):
    # Try gensim first
    try:
        from gensim.models import KeyedVectors
        binary = path.endswith('.bin') or path.endswith('.model')
        if binary:
            try: return KeyedVectors.load(path)
            except: pass
        return KeyedVectors.load_word2vec_format(path, binary=binary)
    except:
        # Fallback manual parser
        print("  Gensim not available/failed, using manual parser...")
        class MockWV:
            def __init__(self, vocab, vecs, dim):
                self.key_to_index = vocab
                self.vectors = vecs
                self.vector_size = dim
        
        with open(path, 'r') as f:
            head = f.readline().strip().split()
            try: dim = int(head[1])
            except: f.seek(0); dim = None
            vocab, vecs = {}, []
            idx = 0
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2: continue
                if dim is None: dim = len(parts)-1
                vocab[parts[0]] = idx
                vecs.append([float(x) for x in parts[1:]])
                idx += 1
        return MockWV(vocab, np.array(vecs, dtype=np.float32), dim)

def extract_gene2vec(annot_tsv: str, model_path: str, out_file: str):
    print(f"Extracting Gene2Vec from {model_path}...")
    df = pd.read_csv(annot_tsv, sep='\t')
    gids = df['GeneID'].astype(str).tolist()
    
    # Load model
    wv = _load_word2vec_format(model_path)
    if hasattr(wv, 'key_to_index'): # Gensim style
        vocab = wv.key_to_index
        matrix = wv.vectors
        dim = wv.vector_size
    else:
        raise ValueError("Could not load Gene2Vec model format")
    
    idxs = []
    miss = 0
    for g in gids:
        # 1. Exact match
        i = vocab.get(g)
        # 2. Version strip (ENS001.1 -> ENS001)
        if i is None and '.' in g:
            i = vocab.get(g.split('.')[0])
        
        if i is None: miss += 1
        idxs.append(i)
        
    print(f"  Gene Match Rate: {1 - miss/len(gids):.2%}")
    
    out = np.zeros((len(gids), dim), dtype=np.float32)
    for j, i in enumerate(idxs):
        if i is not None:
            out[j] = matrix[i]
            
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_file, out)
    return out_file