"""
create_batch_jobs.py

Create SLURM batch job scripts for running experiments on a cluster.
Each experiment gets its own job script with appropriate resources.

Authors: Hannah Kim
Date: 2025-07-29
"""

import os
import sys
import json
import argparse
from pathlib import Path
import subprocess

def create_slurm_job_script(experiment, output_dir, job_name=None):
    """
    Create a SLURM job script for a single experiment.
    
    Parameters:
    - experiment: dict with experiment configuration
    - output_dir: str, directory to save job scripts
    - job_name: str, optional custom job name
    """
    
    config = experiment["config"]
    exp_name = experiment["name"]
    dataset_type = experiment["dataset_type"]
    encoder_type = experiment["encoder_type"]
    
    if job_name is None:
        job_name = f"exp_{exp_name}"
    
    # Create config file for this experiment
    config_path = os.path.join(output_dir, f"{exp_name}_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # SLURM job script template
    job_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output_dir}/logs/{exp_name}_%j.out
#SBATCH --error={output_dir}/logs/{exp_name}_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hjkim@g.harvard.edu

# Experiment: {exp_name}
# Dataset: {dataset_type}
# Encoder: {encoder_type}

set -e

echo "Starting SLURM job: {job_name}"
echo "Experiment: {exp_name}"
echo "Dataset type: {dataset_type}"
echo "Encoder type: {encoder_type}"
echo "Job ID: $SLURM_JOB_ID"

# Load modules (adjust for your cluster)
module load cuda/11.8
module load anaconda/2023a

# Activate conda environment
source activate torch2

# Set up environment
cd /n/home12/hjkim/Github/DiffusionObjectRelation

# Create logs directory
mkdir -p {output_dir}/logs

# Log start time
echo "$(date): Starting experiment {exp_name}" >> {output_dir}/logs/{exp_name}_$SLURM_JOB_ID.log

# Step 1: Data Generation
echo "$(date): Step 1 - Generating dataset..." >> {output_dir}/logs/{exp_name}_$SLURM_JOB_ID.log
python hannah/data_generation.py --config {config_path} 2>&1 | tee -a {output_dir}/logs/{exp_name}_$SLURM_JOB_ID.log

if [ $? -eq 0 ]; then
    echo "$(date): Data generation completed successfully" >> {output_dir}/logs/{exp_name}_$SLURM_JOB_ID.log
else
    echo "$(date): Data generation failed" >> {output_dir}/logs/{exp_name}_$SLURM_JOB_ID.log
    exit 1
fi

# Step 2: Training (uncomment when ready)
# echo "$(date): Step 2 - Starting training..." >> {output_dir}/logs/{exp_name}_$SLURM_JOB_ID.log
# python train_scripts/train_diffusers_pixart_objrel.py --config {config_path} 2>&1 | tee -a {output_dir}/logs/{exp_name}_$SLURM_JOB_ID.log

# Step 3: Evaluation (uncomment when ready)
# echo "$(date): Step 3 - Running evaluation..." >> {output_dir}/logs/{exp_name}_$SLURM_JOB_ID.log
# python experimental_scripts/posthoc_generation_eval_cli.py --config {config_path} 2>&1 | tee -a {output_dir}/logs/{exp_name}_$SLURM_JOB_ID.log

echo "$(date): Experiment {exp_name} completed successfully" >> {output_dir}/logs/{exp_name}_$SLURM_JOB_ID.log
echo "Experiment {exp_name} completed!"
"""
    
    job_script_path = os.path.join(output_dir, f"job_{exp_name}.sh")
    with open(job_script_path, 'w') as f:
        f.write(job_script)
    
    # Make script executable
    os.chmod(job_script_path, 0o755)
    
    return job_script_path

def create_submission_script(experiments, output_dir):
    """
    Create a script to submit all jobs to SLURM.
    """
    
    submission_script = f"""#!/bin/bash
# Batch job submission script
# Submits all experiments to SLURM

set -e

echo "Submitting {len(experiments)} experiments to SLURM..."

# Create logs directory
mkdir -p {output_dir}/logs

# Submit each job
"""
    
    for i, experiment in enumerate(experiments):
        exp_name = experiment["name"]
        job_script = os.path.join(output_dir, f"job_{exp_name}.sh")
        
        submission_script += f"""
echo "Submitting experiment {i+1}/{len(experiments)}: {exp_name}"
sbatch {job_script}
sleep 2  # Small delay between submissions
"""
    
    submission_script += """
echo "All jobs submitted!"
echo "Check job status with: squeue -u $USER"
"""
    
    submission_path = os.path.join(output_dir, "submit_all_jobs.sh")
    with open(submission_path, 'w') as f:
        f.write(submission_script)
    
    os.chmod(submission_path, 0o755)
    return submission_path

def create_experiment_configs():
    """
    Create all experiment configurations.
    Returns list of config dictionaries for each experiment.
    """
    
    base_config = {
        "num_images": 10000,
        "resolution": 128,
        "radius": 16,  # Always use radius=16 for consistency
        "model_max_length": 20,
        "pixart_dir": "/n/home12/hjkim/Github/DiffusionObjectRelation/PixArt-alpha",
        "save_dir": "/n/home12/hjkim/Github/DiffusionObjectRelation/output",
        "using_existing_img_txt": False,
        "validation_prompts": [
            "a red circle",
            "a blue square", 
            "a triangle",
            "circle is above square",
            "triangle is to the left of circle"
        ]
    }
    
    experiments = []
    
    # Dataset types
    dataset_types = ["Single", "Double", "Mixed"]
    
    # Embedding types
    embedding_types = ["RandomEmbeddingEncoder", "T5"]
    
    # Generate all combinations
    for dataset_type in dataset_types:
        for encoder_type in embedding_types:
            config = base_config.copy()
            config["ObjRelDataset"] = dataset_type
            config["encoder_type"] = encoder_type
            
            # Set dataset-specific parameters
            if dataset_type == "Mixed":
                config["single_ratio"] = 0.3
                config["dataset_name"] = f"objectRel_mixed_{encoder_type.lower()}_pilot"
            else:
                config["dataset_name"] = f"objectRel_{dataset_type.lower()}_{encoder_type.lower()}_pilot"
            
            experiments.append({
                "config": config,
                "name": f"{dataset_type}_{encoder_type}",
                "dataset_type": dataset_type,
                "encoder_type": encoder_type
            })
    
    return experiments

def main():
    parser = argparse.ArgumentParser(description="Create SLURM batch job scripts")
    parser.add_argument("--output_dir", type=str, default="batch_jobs", 
                       help="Directory to save job scripts")
    parser.add_argument("--submit", action="store_true",
                       help="Submit jobs after creating scripts")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Creating batch jobs in: {output_dir}")
    
    # Generate all experiment configurations
    experiments = create_experiment_configs()
    
    print(f"Generated {len(experiments)} experiments:")
    for exp in experiments:
        print(f"  - {exp['name']}: {exp['dataset_type']} + {exp['encoder_type']}")
    
    # Create individual job scripts
    job_scripts = []
    for experiment in experiments:
        job_script = create_slurm_job_script(experiment, output_dir)
        job_scripts.append(job_script)
        print(f"Created: {job_script}")
    
    # Create submission script
    submission_script = create_submission_script(experiments, output_dir)
    print(f"Created submission script: {submission_script}")
    
    print(f"\nBatch job setup complete!")
    print(f"To submit all jobs:")
    print(f"  cd {output_dir}")
    print(f"  bash submit_all_jobs.sh")
    print(f"\nOr submit individual jobs:")
    for script in job_scripts:
        print(f"  sbatch {script}")
    
    if args.submit:
        print(f"\nSubmitting jobs...")
        subprocess.run(["bash", submission_script], cwd=output_dir)

if __name__ == "__main__":
    main() 