"""
batch_job_manager.py

Manage SLURM batch jobs for experiments.
Check status, cancel jobs, view logs, etc.

Authors: Hannah Kim
Date: 2025-07-29
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def list_jobs():
    """List all running and pending jobs."""
    print("Current SLURM jobs:")
    print("-" * 50)
    subprocess.run(["squeue", "-u", "$USER"])

def check_job_status(job_dir="batch_jobs"):
    """Check status of experiment jobs."""
    if not os.path.exists(job_dir):
        print(f"Job directory not found: {job_dir}")
        return
    
    print(f"Experiment job status in {job_dir}:")
    print("-" * 60)
    
    experiments = [
        "Single_RandomEmbeddingEncoder",
        "Single_T5", 
        "Double_RandomEmbeddingEncoder",
        "Double_T5",
        "Mixed_RandomEmbeddingEncoder",
        "Mixed_T5"
    ]
    
    for exp_name in experiments:
        job_script = os.path.join(job_dir, f"job_{exp_name}.sh")
        config_file = os.path.join(job_dir, f"{exp_name}_config.json")
        
        status = []
        if os.path.exists(job_script):
            status.append("✓ Job script")
        else:
            status.append("✗ No job script")
            
        if os.path.exists(config_file):
            status.append("✓ Config")
        else:
            status.append("✗ No config")
        
        # Check for log files
        log_dir = os.path.join(job_dir, "logs")
        if os.path.exists(log_dir):
            log_files = [f for f in os.listdir(log_dir) if f.startswith(f"{exp_name}_")]
            if log_files:
                status.append(f"✓ {len(log_files)} log files")
            else:
                status.append("No logs")
        else:
            status.append("No logs dir")
        
        print(f"{exp_name:30} | {' | '.join(status)}")

def view_logs(experiment_name, job_dir="batch_jobs", lines=50):
    """View logs for a specific experiment."""
    log_dir = os.path.join(job_dir, "logs")
    
    if not os.path.exists(log_dir):
        print(f"Log directory not found: {log_dir}")
        return
    
    # Find log files for this experiment
    log_files = [f for f in os.listdir(log_dir) if f.startswith(f"{experiment_name}_")]
    
    if not log_files:
        print(f"No log files found for experiment: {experiment_name}")
        return
    
    # Show most recent log
    log_files.sort()
    latest_log = log_files[-1]
    log_path = os.path.join(log_dir, latest_log)
    
    print(f"Latest log for {experiment_name}: {latest_log}")
    print("-" * 60)
    
    # Show last N lines
    try:
        with open(log_path, 'r') as f:
            lines_content = f.readlines()
            for line in lines_content[-lines:]:
                print(line.rstrip())
    except Exception as e:
        print(f"Error reading log file: {e}")

def cancel_jobs(experiment_name=None):
    """Cancel jobs. If experiment_name is None, cancel all user jobs."""
    if experiment_name:
        print(f"Cancelling jobs for experiment: {experiment_name}")
        # You might need to implement job ID tracking for specific experiments
        print("Note: This will cancel all user jobs. Use 'squeue -u $USER' to see job IDs first.")
    else:
        print("Cancelling all user jobs...")
        response = input("Are you sure? (y/N): ")
        if response.lower() == 'y':
            subprocess.run(["scancel", "-u", "$USER"])
            print("All jobs cancelled.")
        else:
            print("Cancellation cancelled.")

def submit_jobs(job_dir="batch_jobs"):
    """Submit all jobs to SLURM."""
    submission_script = os.path.join(job_dir, "submit_all_jobs.sh")
    
    if not os.path.exists(submission_script):
        print(f"Submission script not found: {submission_script}")
        print("Run create_batch_jobs.py first to create job scripts.")
        return
    
    print("Submitting all experiment jobs to SLURM...")
    subprocess.run(["bash", submission_script], cwd=job_dir)

def create_job_summary(job_dir="batch_jobs"):
    """Create a summary of all jobs and their status."""
    summary_path = os.path.join(job_dir, "JOB_SUMMARY.md")
    
    summary_content = """# Batch Job Summary

## Job Status
| Experiment | Job Script | Config | Logs | Status |
|------------|------------|---------|------|---------|
"""
    
    experiments = [
        "Single_RandomEmbeddingEncoder",
        "Single_T5", 
        "Double_RandomEmbeddingEncoder",
        "Double_T5",
        "Mixed_RandomEmbeddingEncoder",
        "Mixed_T5"
    ]
    
    for exp_name in experiments:
        job_script = os.path.join(job_dir, f"job_{exp_name}.sh")
        config_file = os.path.join(job_dir, f"{exp_name}_config.json")
        log_dir = os.path.join(job_dir, "logs")
        
        job_exists = "✓" if os.path.exists(job_script) else "✗"
        config_exists = "✓" if os.path.exists(config_file) else "✗"
        
        if os.path.exists(log_dir):
            log_files = [f for f in os.listdir(log_dir) if f.startswith(f"{exp_name}_")]
            logs_status = f"{len(log_files)} files" if log_files else "None"
        else:
            logs_status = "No logs dir"
        
        summary_content += f"| {exp_name} | {job_exists} | {config_exists} | {logs_status} | Pending |\n"
    
    summary_content += f"""

## Commands

### Submit all jobs:
```bash
cd {job_dir}
bash submit_all_jobs.sh
```

### Check job status:
```bash
squeue -u $USER
```

### View logs for specific experiment:
```bash
python experimental_scripts/batch_job_manager.py --view-logs Single_T5
```

### Cancel all jobs:
```bash
scancel -u $USER
```
"""
    
    with open(summary_path, 'w') as f:
        f.write(summary_content)
    
    print(f"Job summary created: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Manage SLURM batch jobs")
    parser.add_argument("--list", action="store_true", help="List all user jobs")
    parser.add_argument("--status", action="store_true", help="Check experiment job status")
    parser.add_argument("--view-logs", type=str, help="View logs for specific experiment")
    parser.add_argument("--cancel", action="store_true", help="Cancel all user jobs")
    parser.add_argument("--submit", action="store_true", help="Submit all jobs")
    parser.add_argument("--summary", action="store_true", help="Create job summary")
    parser.add_argument("--job-dir", type=str, default="batch_jobs", help="Job directory")
    
    args = parser.parse_args()
    
    if args.list:
        list_jobs()
    elif args.status:
        check_job_status(args.job_dir)
    elif args.view_logs:
        view_logs(args.view_logs, args.job_dir)
    elif args.cancel:
        cancel_jobs()
    elif args.submit:
        submit_jobs(args.job_dir)
    elif args.summary:
        create_job_summary(args.job_dir)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python batch_job_manager.py --list")
        print("  python batch_job_manager.py --status")
        print("  python batch_job_manager.py --view-logs Single_T5")
        print("  python batch_job_manager.py --submit")

if __name__ == "__main__":
    main() 