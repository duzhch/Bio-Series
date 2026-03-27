#!/usr/bin/env python3
import argparse
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

import yaml

PATHLIKE_RESOURCE_KEYS = {
    'reference_genome',
    'gtf_file',
    'pigbert_model',
    'gene2vec_model',
}


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


def submit_and_wait(script_path, wait_interval=30):
    try:
        result = subprocess.run(['sbatch', str(script_path)], capture_output=True, text=True, check=True)
        match = re.search(r'Submitted batch job (\d+)', result.stdout)
        if match:
            return int(match.group(1))
        print(f"Warning: Could not extract job ID from: {result.stdout}")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job {script_path}: {e}")
        return None


def check_job_status(job_id):
    try:
        result = subprocess.run(['squeue', '-j', str(job_id)], capture_output=True, text=True)
        return str(job_id) in result.stdout
    except subprocess.CalledProcessError:
        return False


def wait_for_jobs(job_ids, wait_interval=30, max_wait_hours=48):
    if not job_ids:
        return

    print(f"Waiting for {len(job_ids)} jobs to complete...")
    start_time = time.time()
    max_wait_seconds = max_wait_hours * 3600

    while job_ids:
        time.sleep(wait_interval)
        elapsed = time.time() - start_time
        if elapsed > max_wait_seconds:
            print(f"Timeout after {max_wait_hours} hours. Remaining jobs: {job_ids}")
            break

        completed_jobs = []
        for job_id in job_ids:
            if not check_job_status(job_id):
                completed_jobs.append(job_id)
                print(f"Job {job_id} completed")

        for job_id in completed_jobs:
            job_ids.remove(job_id)

        if job_ids:
            print(f"Still waiting for {len(job_ids)} jobs... (elapsed: {elapsed/3600:.1f}h)")


def main():
    ap = argparse.ArgumentParser(description='Generate sbatch wrappers for DF-GSF v5')
    ap.add_argument('--config', required=True)
    ap.add_argument('--datasets', required=True, help='Comma-separated dataset keys')
    ap.add_argument('--traits', required=True, help='Comma-separated traits')
    ap.add_argument('--reps', type=int, default=None, help='Number of reps (default=from config)')
    ap.add_argument('--model', default='bio_master_v8', help='Model module name under src.models')
    ap.add_argument('--cpu-only', action='store_true', help='Submit to CPU partition only')
    ap.add_argument('--out-sh', required=True)
    ap.add_argument('--wait', action='store_true', help='Wait for all jobs to complete')
    ap.add_argument('--wait-interval', type=int, default=30, help='Job status check interval in seconds')
    ap.add_argument('--max-wait-hours', type=int, default=48, help='Maximum wait time in hours')
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    exp_root = Path(cfg['exp_root'])
    cfg_path = Path(args.config).resolve()
    nrep = args.reps or cfg['experiment']['replicates']
    ds_list = [x.strip() for x in args.datasets.split(',') if x.strip()]
    tr_list = [x.strip() for x in args.traits.split(',') if x.strip()]

    job_dir = exp_root / 'slurm_jobs/scripts_v5'
    log_dir = exp_root / 'slurm_jobs/logs_v5'
    job_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    submit_lines = ['#!/bin/bash', 'set -euo pipefail']
    job_scripts = []

    for ds in ds_list:
        if ds not in cfg['datasets']:
            raise KeyError(f"Unknown dataset key: {ds}")
        dscfg = cfg['datasets'][ds]
        for trait in tr_list:
            if trait not in dscfg['traits']:
                print(f"Skipping trait {trait} for dataset {ds}: not listed in config")
                continue

            for r in range(1, nrep + 1):
                rep = f'rep_{r:02d}'
                job = f'{ds}_{trait}_{args.model}_{rep}'
                out = log_dir / f'{job}.out'
                err = log_dir / f'{job}.err'
                script = job_dir / f'{job}.sh'

                header = [
                    '#!/bin/bash',
                    'set -euo pipefail',
                    f"#SBATCH -J {job}",
                    f"#SBATCH -p {cfg['slurm']['cpu_partition'] if args.cpu_only else cfg['slurm']['gpu_partition']}",
                    '#SBATCH --nodes=1',
                    '#SBATCH --ntasks=1',
                    f"#SBATCH --cpus-per-task={cfg['slurm']['cpus_per_task']}",
                    f"#SBATCH --mem={cfg['slurm']['mem']}",
                    *( [] if args.cpu_only else [f"#SBATCH --gres={cfg['slurm']['gres']}"] ),
                    f"#SBATCH -o {out}",
                    f"#SBATCH -e {err}",
                    f"#SBATCH --time={cfg['slurm']['time']}",
                ]
                body = [
                    f"cd {shlex.quote(str(exp_root))}",
                    f"{shlex.quote(str(cfg['resources']['python_bin']))} DF_GSF_v5.py run-all "
                    f"--config {shlex.quote(str(cfg_path))} "
                    f"--dataset {shlex.quote(ds)} --trait {shlex.quote(trait)} --rep {shlex.quote(rep)} "
                    f"--model {shlex.quote(args.model)}"
                ]
                script.write_text("\n".join(header + body) + "\n")
                script.chmod(0o755)
                job_scripts.append(script)
                submit_lines.append(f"sbatch {shlex.quote(str(script))}")

    out_sh = Path(args.out_sh)
    out_sh.write_text("\n".join(submit_lines) + "\n")
    out_sh.chmod(0o755)
    print(f"Generated submit script: {out_sh}")

    if args.wait:
        print(f"Submitting {len(job_scripts)} jobs and waiting for completion...")
        job_ids = []
        for script in job_scripts:
            job_id = submit_and_wait(script)
            if job_id:
                job_ids.append(job_id)
                print(f"Submitted job {job_id}: {script.name}")
            else:
                print(f"Failed to submit job: {script.name}")

        if job_ids:
            wait_for_jobs(job_ids, args.wait_interval, args.max_wait_hours)
            print("All jobs completed!")
        else:
            print("No jobs were successfully submitted.")
    else:
        print(f"Use '{out_sh}' to submit jobs, or run with --wait to submit and monitor automatically.")


if __name__ == '__main__':
    main()
