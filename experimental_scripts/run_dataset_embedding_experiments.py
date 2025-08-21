"""
run_dataset_embedding_experiments.py

Systematic experiment runner for testing dataset types vs embedding types.
Runs all combinations of:
- Dataset Types: Single, Double, Mixed
- Embedding Types: Random, T5

Authors: Hannah Kim
Date: 2025-07-29
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

# Add paths for imports
sys.path.append("/n/home12/hjkim/Github/DiffusionObjectRelation/PixArt-alpha")
sys.path.append("/n/home12/hjkim/Github/DiffusionObjectRelation/")

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

def create_experiment_script(experiment, output_dir):
    """
    Create a script to run a single experiment.
    
    Parameters:
    - experiment: dict with experiment configuration
    - output_dir: str, directory to save experiment outputs
    """
    
    config = experiment["config"]
    exp_name = experiment["name"]
    
    # Create config file for this experiment
    config_path = os.path.join(output_dir, f"{exp_name}_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create experiment script
    script_content = f"""#!/bin/bash
# Experiment: {exp_name}
# Dataset: {experiment['dataset_type']}
# Encoder: {experiment['encoder_type']}

set -e

echo "Starting experiment: {exp_name}"
echo "Dataset type: {experiment['dataset_type']}"
echo "Encoder type: {experiment['encoder_type']}"

# Set up environment
cd /n/home12/hjkim/Github/DiffusionObjectRelation

# Run data generation
echo "Generating dataset..."
python hannah/data_generation.py --config {config_path}

# Run training (you'll need to adapt this to your training script)
echo "Starting training..."
# python train_scripts/train_diffusers_pixart_objrel.py --config {config_path}

# Run evaluation
echo "Running evaluation..."
# python experimental_scripts/posthoc_generation_eval_cli.py --config {config_path}

echo "Experiment {exp_name} completed!"
"""
    
    script_path = os.path.join(output_dir, f"run_{exp_name}.sh")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    os.chmod(script_path, 0o755)
    
    return script_path

def create_batch_script(experiments, output_dir):
    """
    Create a batch script to run all experiments.
    """
    
    batch_content = """#!/bin/bash
# Batch experiment runner
# Runs all dataset type vs embedding type combinations

set -e

echo "Starting batch experiments..."
echo "Total experiments: {len(experiments)}"

""".format(len=experiments))
    
    for i, experiment in enumerate(experiments):
        exp_name = experiment["name"]
        script_path = os.path.join(output_dir, f"run_{exp_name}.sh")
        
        batch_content += f"""
echo "Running experiment {i+1}/{len(experiments)}: {exp_name}"
echo "=================================================="
bash {script_path}
echo "=================================================="
echo ""

"""
    
    batch_content += """
echo "All experiments completed!"
"""
    
    batch_path = os.path.join(output_dir, "run_all_experiments.sh")
    with open(batch_path, 'w') as f:
        f.write(batch_content)
    
    os.chmod(batch_path, 0o755)
    return batch_path

def create_experiment_summary(experiments, output_dir):
    """
    Create a summary of all experiments.
    """
    
    summary_content = """# Experiment Summary

## Experiment Matrix
| Dataset Type | Encoder Type | Experiment Name | Status |
|-------------|--------------|-----------------|---------|
"""
    
    for experiment in experiments:
        exp_name = experiment["name"]
        dataset_type = experiment["dataset_type"]
        encoder_type = experiment["encoder_type"]
        
        summary_content += f"| {dataset_type} | {encoder_type} | {exp_name} | Pending |\n"
    
    summary_content += f"""

## Total Experiments: {len(experiments)}

## Dataset Types:
- Single: Single object images
- Double: Two objects with spatial relationships  
- Mixed: 30% single, 70% double objects

## Encoder Types:
- RandomEmbeddingEncoder: Random embeddings
- T5: T5 text encoder

## Output Directory: {output_dir}

## Running Experiments:
```bash
cd {output_dir}
# Run all experiments
bash run_all_experiments.sh

# Or run individual experiments
bash run_Single_RandomEmbeddingEncoder.sh
bash run_Single_T5.sh
# etc...
```
"""
    
    summary_path = os.path.join(output_dir, "EXPERIMENT_SUMMARY.md")
    with open(summary_path, 'w') as f:
        f.write(summary_content)
    
    return summary_path

def main():
    parser = argparse.ArgumentParser(description="Create experiment configurations")
    parser.add_argument("--output_dir", type=str, default="experiment_outputs", 
                       help="Directory to save experiment configurations")
    parser.add_argument("--create_only", action="store_true",
                       help="Only create configs, don't run experiments")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Creating experiments in: {output_dir}")
    
    # Generate all experiment configurations
    experiments = create_experiment_configs()
    
    print(f"Generated {len(experiments)} experiments:")
    for exp in experiments:
        print(f"  - {exp['name']}: {exp['dataset_type']} + {exp['encoder_type']}")
    
    # Create individual experiment scripts
    script_paths = []
    for experiment in experiments:
        script_path = create_experiment_script(experiment, output_dir)
        script_paths.append(script_path)
        print(f"Created: {script_path}")
    
    # Create batch script
    batch_path = create_batch_script(experiments, output_dir)
    print(f"Created batch script: {batch_path}")
    
    # Create summary
    summary_path = create_experiment_summary(experiments, output_dir)
    print(f"Created summary: {summary_path}")
    
    print(f"\nExperiment setup complete!")
    print(f"To run all experiments:")
    print(f"  cd {output_dir}")
    print(f"  bash run_all_experiments.sh")
    
    if not args.create_only:
        print(f"\nStarting experiments...")
        subprocess.run(["bash", batch_path], cwd=output_dir)

if __name__ == "__main__":
    main() 