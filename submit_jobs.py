#!/usr/bin/env python3
import argparse
import yaml
import subprocess
import time
import re
from pathlib import Path


def load_cfg(path: str):
    with open(path) as f:
        return yaml.safe_load(f)


def submit_and_wait(script_path, wait_interval=30):
    """Submit job and return job ID"""
    try:
        result = subprocess.run(['sbatch', str(script_path)], 
                              capture_output=True, text=True, check=True)
        # Extract job ID from sbatch output: "Submitted batch job 12345"
        match = re.search(r'Submitted batch job (\d+)', result.stdout)
        if match:
            return int(match.group(1))
        else:
            print(f"Warning: Could not extract job ID from: {result.stdout}")
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job {script_path}: {e}")
        return None


def check_job_status(job_id):
    """Check if job is still running"""
    try:
        result = subprocess.run(['squeue', '-j', str(job_id)], 
                              capture_output=True, text=True)
        return str(job_id) in result.stdout
    except subprocess.CalledProcessError:
        return False


def wait_for_jobs(job_ids, wait_interval=30, max_wait_hours=48):
    """Wait for all jobs to complete"""
    if not job_ids:
        return
    
    print(f"Waiting for {len(job_ids)} jobs to complete...")
    start_time = time.time()
    max_wait_seconds = max_wait_hours * 3600
    
    while job_ids:
        time.sleep(wait_interval)
        
        # Check elapsed time
        elapsed = time.time() - start_time
        if elapsed > max_wait_seconds:
            print(f"Timeout after {max_wait_hours} hours. Remaining jobs: {job_ids}")
            break
            
        # Check job status
        completed_jobs = []
        for job_id in job_ids:
            if not check_job_status(job_id):
                completed_jobs.append(job_id)
                print(f"Job {job_id} completed")
        
        # Remove completed jobs
        for job_id in completed_jobs:
            job_ids.remove(job_id)
            
        if job_ids:
            print(f"Still waiting for {len(job_ids)} jobs... (elapsed: {elapsed/3600:.1f}h)")


def main():
    ap = argparse.ArgumentParser(description='Generate simple sbatch wrappers for DF-GSF v5')
    ap.add_argument('--config', required=True)
    ap.add_argument('--datasets', required=True, help='Comma-separated dataset keys')
    ap.add_argument('--traits', required=True, help='Comma-separated traits')
    ap.add_argument('--reps', type=int, default=None, help='Number of reps (default=from config)')
    ap.add_argument('--cpu-only', action='store_true', help='Submit to CPU partition only')
    ap.add_argument('--out-sh', required=True)
    ap.add_argument('--wait', action='store_true', help='Wait for all jobs to complete')
    ap.add_argument('--wait-interval', type=int, default=30, help='Job status check interval in seconds')
    ap.add_argument('--max-wait-hours', type=int, default=48, help='Maximum wait time in hours')
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    exp_root = Path(cfg['exp_root'])
    nrep = args.reps or cfg['experiment']['replicates']
    ds_list = [x.strip() for x in args.datasets.split(',') if x.strip()]
    tr_list = [x.strip() for x in args.traits.split(',') if x.strip()]

    job_dir = exp_root / 'slurm_jobs/scripts_v5'
    log_dir = exp_root / 'slurm_jobs/logs_v5'
    job_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    submit_lines = ['#!/bin/bash']
    job_scripts = []
    for ds in ds_list:
        dscfg = cfg['datasets'][ds]
        for trait in tr_list:
            if trait not in dscfg['traits']:
                continue
            for r in range(1, nrep + 1):
                rep = f'rep_{r:02d}'
                job = f'{ds}_{trait}_v5_{rep}'
                out = log_dir / f'{job}.out'
                err = log_dir / f'{job}.err'
                script = job_dir / f'{job}.sh'

                header = [
                    '#!/bin/bash',
                    f"#SBATCH -J {job}",
                    f"#SBATCH -p {cfg['slurm']['cpu_partition'] if args.cpu_only else cfg['slurm']['gpu_partition']}",
                    f"#SBATCH --nodes=1",
                    f"#SBATCH --ntasks=1",
                    f"#SBATCH --cpus-per-task={cfg['slurm']['cpus_per_task']}",
                    f"#SBATCH --mem={cfg['slurm']['mem']}",
                    *( [] if args.cpu_only else [f"#SBATCH --gres={cfg['slurm']['gres']}"] ),
                    f"#SBATCH -o {out}",
                    f"#SBATCH -e {err}",
                    f"#SBATCH --time={cfg['slurm']['time']}",
                ]
                body = [
                    f"cd {exp_root/'DF_GSF_v5'}",
                    f"{cfg['resources']['python_bin']} DF_GSF_v5.py run-all "
                    f"--config config/global_config.yaml "
                    f"--dataset {ds} --trait {trait} --rep {rep}"
                ]
                script.write_text("\n".join(header + body) + "\n")
                script.chmod(0o755)
                job_scripts.append(script)
                submit_lines.append(f"sbatch {script}")

    out_sh = Path(args.out_sh)
    out_sh.write_text("\n".join(submit_lines) + "\n")
    out_sh.chmod(0o755)
    print(f"Generated submit script: {out_sh}")
    
    # If --wait flag is provided, submit jobs and wait for completion
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

