"""
experiment_runner.py

Simple experiment runner for individual experiments.
Can run specific experiments or check status.

Authors: Hannah Kim
Date: 2025-07-29
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

def run_single_experiment(dataset_type, encoder_type, output_dir="experiment_outputs"):
    """
    Run a single experiment.
    
    Parameters:
    - dataset_type: str, "Single", "Double", or "Mixed"
    - encoder_type: str, "RandomEmbeddingEncoder" or "T5"
    - output_dir: str, directory for experiment outputs
    """
    
    exp_name = f"{dataset_type}_{encoder_type}"
    config_path = os.path.join(output_dir, f"{exp_name}_config.json")
    
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        print("Run run_dataset_embedding_experiments.py first to create experiment configs.")
        return
    
    print(f"Running experiment: {exp_name}")
    print(f"Dataset: {dataset_type}")
    print(f"Encoder: {encoder_type}")
    
    # Set up environment
    os.chdir("/n/home12/hjkim/Github/DiffusionObjectRelation")
    
    # Run data generation
    print("1. Generating dataset...")
    subprocess.run(["python", "hannah/data_generation.py", "--config", config_path], check=True)
    
    # TODO: Add training and evaluation steps
    print("2. Training (not implemented yet)")
    print("3. Evaluation (not implemented yet)")
    
    print(f"Experiment {exp_name} completed!")

def list_experiments(output_dir="experiment_outputs"):
    """
    List all available experiments and their status.
    """
    
    if not os.path.exists(output_dir):
        print(f"Output directory not found: {output_dir}")
        print("Run run_dataset_embedding_experiments.py first to create experiment configs.")
        return
    
    print(f"Experiments in {output_dir}:")
    print("-" * 50)
    
    experiments = [
        ("Single", "RandomEmbeddingEncoder"),
        ("Single", "T5"),
        ("Double", "RandomEmbeddingEncoder"),
        ("Double", "T5"),
        ("Mixed", "RandomEmbeddingEncoder"),
        ("Mixed", "T5")
    ]
    
    for dataset_type, encoder_type in experiments:
        exp_name = f"{dataset_type}_{encoder_type}"
        config_path = os.path.join(output_dir, f"{exp_name}_config.json")
        
        status = "✓ Config exists" if os.path.exists(config_path) else "✗ No config"
        print(f"{exp_name:30} | {status}")

def main():
    parser = argparse.ArgumentParser(description="Run individual experiments")
    parser.add_argument("--dataset", type=str, choices=["Single", "Double", "Mixed"],
                       help="Dataset type to run")
    parser.add_argument("--encoder", type=str, choices=["RandomEmbeddingEncoder", "T5"],
                       help="Encoder type to run")
    parser.add_argument("--list", action="store_true",
                       help="List all available experiments")
    parser.add_argument("--output_dir", type=str, default="experiment_outputs",
                       help="Directory containing experiment configs")
    
    args = parser.parse_args()
    
    if args.list:
        list_experiments(args.output_dir)
    elif args.dataset and args.encoder:
        run_single_experiment(args.dataset, args.encoder, args.output_dir)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python experiment_runner.py --list")
        print("  python experiment_runner.py --dataset Single --encoder T5")

if __name__ == "__main__":
    main() 