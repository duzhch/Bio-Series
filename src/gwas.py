import os
import subprocess
import pandas as pd
import numpy as np
import logging
from pathlib import Path

from src.id_mapping import map_pheno_ids_to_plink_ids

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GWASSelector:
    def __init__(self, plink_prefix, out_dir, plink_bin, gcta_bin, thread_num=4):
        self.plink_prefix = plink_prefix
        self.out_dir = Path(out_dir)
        self.plink_bin = plink_bin
        self.gcta_bin = gcta_bin
        self.thread_num = str(thread_num)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def run_cmd(self, cmd):
        logger.info(f"Running: {cmd}")
        try:
            # [FIX] 捕获 stdout 和 stderr，以便调试
            result = subprocess.run(
                cmd, 
                shell=True, 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as e:
            # [FIX] 当 GCTA 报错时，它的错误信息通常在 stdout 而不是 stderr
            # 这里同时打印两者，确保不错过任何线索
            error_msg = (
                f"\n=== COMMAND FAILED ===\n"
                f"CMD: {cmd}\n"
                f"EXIT CODE: {e.returncode}\n"
                f"--- STDOUT (GCTA Log) ---\n{e.stdout.decode(errors='replace')}\n"
                f"--- STDERR (System Log) ---\n{e.stderr.decode(errors='replace')}\n"
                f"========================\n"
            )
            logger.error(error_msg)
            raise RuntimeError(f"Command execution failed. Check logs above for details.") from e

    def generate_global_pca(self, num_pcs=10):
        """
        [Wide Part Input] 计算全基因组 PCA。
        """
        pca_out = self.out_dir / "global_pca"
        final_pca_path = self.out_dir / "global_pca_features.csv"

        if final_pca_path.exists():
            logger.info("Global PCA features already exist. Skipping calculation.")
            return final_pca_path

        logger.info("Calculating Genome-wide PCA...")
        # PLINK 使用 --threads (注意复数)
        cmd = (
            f"{self.plink_bin} --bfile {self.plink_prefix} "
            f"--pca {num_pcs} "
            f"--out {pca_out} "
            f"--threads {self.thread_num}"
        )
        self.run_cmd(cmd)
        
        try:
            df = pd.read_csv(f"{pca_out}.eigenvec", sep='\s+', header=None)
            cols = ['FID', 'IID'] + [f'PC{i+1}' for i in range(num_pcs)]
            if df.shape[1] >= len(cols):
                df = df.iloc[:, :len(cols)]
            df.columns = cols
            df.to_csv(final_pca_path, index=False)
            logger.info(f"Global PCA features saved to {final_pca_path}")
        except Exception as e:
            logger.error(f"Failed to process PCA file: {e}")
            raise
            
        return final_pca_path

    def _prepare_clean_pheno(self, raw_pheno_file, trait):
        """
        [New Helper] 读取原始表型文件(CSV/TSV)，提取特定性状，
        清洗为 GCTA 需要的格式 (FID IID Value, No Header, Tab-separated)。
        """
        logger.info(f"Preparing clean phenotype file for trait: {trait}...")
        
        # 1. 自动检测分隔符
        raw_path = str(raw_pheno_file)
        sep = ',' if raw_path.endswith('.csv') else '\s+'
        
        try:
            df = pd.read_csv(raw_path, sep=sep, engine='python')
        except Exception as e:
            logger.error(f"Failed to read phenotype file {raw_path}: {e}")
            raise

        # 2. 标准化列名查找 (不区分大小写)
        cols_lower = {c.lower(): c for c in df.columns}
        
        # 查找 IID
        iid_col = None
        for candidate in ['sample_id', 'iid', 'id', 'sample']:
            if candidate in cols_lower:
                iid_col = cols_lower[candidate]
                break
        if iid_col is None:
            iid_col = df.columns[0] # 假设第一列是 ID
            
        # 查找 FID (如果不存在则复制 IID)
        fid_col = cols_lower.get('fid')
        
        # 查找 Trait
        if trait not in df.columns:
             # 尝试不区分大小写查找
            trait_lower = trait.lower()
            if trait_lower in cols_lower:
                trait = cols_lower[trait_lower]
            else:
                raise ValueError(f"Trait '{trait}' not found in phenotype file. Available columns: {list(df.columns)}")

        # 3. 构建清洗后的 DataFrame
        mapped_df = map_pheno_ids_to_plink_ids(
            pheno_df=df,
            plink_prefix=self.plink_prefix,
            id_col=iid_col,
        )
        out_df = pd.DataFrame()
        out_df['FID'] = mapped_df['FID']
        out_df['IID'] = mapped_df['IID']
        out_df['Trait'] = mapped_df[trait]
        
        # 4. 处理缺失值 (GCTA 默认识别 -9 或 NA，这里统一转为字符串 "NA")
        out_df['Trait'] = out_df['Trait'].replace([np.inf, -np.inf], np.nan)
        out_df['Trait'] = out_df['Trait'].fillna("NA")
        
        # 5. 保存临时文件
        clean_path = self.out_dir / f"gcta_clean_{trait}.pheno"
        # 关键：header=False, sep='\t'
        out_df.to_csv(clean_path, sep='\t', index=False, header=False)
        logger.info(f"Clean phenotype file saved to: {clean_path}")
        
        return clean_path

    def run_mlma_and_clump(self, pheno_file, train_ids_file, trait, top_n=3000):
        """
        [Deep Part Input] 运行 GCTA GWAS 并执行 PLINK Clumping。
        """
        # [FIX] 第一步：先清洗表型文件
        clean_pheno = self._prepare_clean_pheno(pheno_file, trait)
        
        grm_prefix = self.out_dir / "grm"
        
        # 1. 计算 GRM
        # [FIX] 将 --thread 改为 --thread-num (GCTA 标准参数)
        if not os.path.exists(f"{grm_prefix}.grm.bin"):
            logger.info("Calculating GRM for GCTA...")
            self.run_cmd(f"{self.gcta_bin} --bfile {self.plink_prefix} --make-grm --out {grm_prefix} --thread-num {self.thread_num}")

        # 2. 运行 MLMA
        mlma_out = self.out_dir / "gwas_result"
        if not os.path.exists(f"{mlma_out}.mlma"):
            logger.info("Running GCTA MLMA...")
            # [FIX] 将 --thread 改为 --thread-num
            # [FIX] 使用 clean_pheno 而不是原始 pheno_file
            cmd_mlma = (
                f"{self.gcta_bin} --mlma --bfile {self.plink_prefix} "
                f"--grm {grm_prefix} --pheno {clean_pheno} "
                f"--keep {train_ids_file} "
                f"--out {mlma_out} --thread-num {self.thread_num}"
            )
            self.run_cmd(cmd_mlma)

        # 3. 准备 Clumping 输入
        gwas_df = pd.read_csv(f"{mlma_out}.mlma", sep='\s+')
        clump_in_file = self.out_dir / "gwas_pvalues.txt"
        
        # [FIX] PLINK 默认寻找 "P" (大写)，GCTA 输出是 "p" (小写)。
        # 显式重命名并指定列名，防止 PLINK 找不到列。
        if 'p' in gwas_df.columns:
            gwas_df.rename(columns={'p': 'P'}, inplace=True)
            
        gwas_df[['SNP', 'P']].to_csv(clump_in_file, sep='\t', index=False)

        # 4. 运行 PLINK Clumping
        # PLINK 使用 --threads (复数) 或 --threads-num 不敏感，但通常用 --threads
        clump_out = self.out_dir / "clumped_res"
        logger.info("Running LD Clumping...")
        # [FIX] 显式指定 --clump-snp-field 和 --clump-field，确保万无一失
        cmd_clump = (
            f"{self.plink_bin} --bfile {self.plink_prefix} "
            f"--clump {clump_in_file} "
            f"--clump-p1 0.05 --clump-r2 0.2 --clump-kb 250 "
            f"--clump-snp-field SNP "
            f"--clump-field P "
            f"--out {clump_out} "
            f"--threads {self.thread_num}"
        )
        self.run_cmd(cmd_clump)

        # 5. 解析结果
        clumped_file = f"{clump_out}.clumped"
        selected_snps = []
        
        if os.path.exists(clumped_file) and os.path.getsize(clumped_file) > 0:
            df_clump = pd.read_csv(clumped_file, delim_whitespace=True)
            if len(df_clump) > top_n:
                df_clump = df_clump.sort_values('P')
                selected_snps = df_clump['SNP'].head(top_n).values
            else:
                selected_snps = df_clump['SNP'].values
            logger.info(f"Clumping selected {len(selected_snps)} independent SNPs.")
        else:
            logger.warning("Clumping failed. Fallback to raw P-value sorting.")
            selected_snps = gwas_df.sort_values('P').head(top_n)['SNP'].values

        final_df = gwas_df[gwas_df['SNP'].isin(selected_snps)].copy()
        out_csv = self.out_dir / "snps_for_emb.csv"
        final_df.to_csv(out_csv, index=False)
        
        out_list = self.out_dir / "selected_snp_ids.txt"
        final_df[['SNP']].to_csv(out_list, index=False, header=False)

        return out_csv, out_list

def run_gwas_pipeline(plink_prefix, pheno_file, train_ids, trait, out_dir, plink_bin, gcta_bin, top_n=3000):
    selector = GWASSelector(plink_prefix, out_dir, plink_bin, gcta_bin)
    selector.generate_global_pca(num_pcs=10)
    # [FIX] 传递 trait 参数，以便清洗表型文件
    emb_csv, snp_list = selector.run_mlma_and_clump(pheno_file, train_ids, trait, top_n=top_n)
    return {
        'snps_for_emb': str(emb_csv),
        'plink_snps': str(snp_list)
    }
