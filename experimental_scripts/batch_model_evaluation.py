#!/usr/bin/env python3
"""
Batch evaluation script for comparing all 9 model variants
"""

import subprocess
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Define all model configurations
MODEL_CONFIGS = [
    ("objrel_T5_DiT_B_pilot", "T5"),
    ("objrel_T5_DiT_mini_pilot", "T5"), 
    ("objrel_rndembdposemb_DiT_B_pilot", "RandomEmbeddingEncoder_wPosEmb"),
    ("objrel_rndembdposemb_DiT_micro_pilot", "RandomEmbeddingEncoder_wPosEmb"),
    ("objrel_rndembdposemb_DiT_nano_pilot", "RandomEmbeddingEncoder_wPosEmb"),
    ("objrel_rndembdposemb_DiT_mini_pilot", "RandomEmbeddingEncoder_wPosEmb"),
    ("objrel_rndemb_DiT_B_pilot", "RandomEmbeddingEncoder"),
    ("objrel_T5_DiT_B_pilot_WDecay", "T5"),
    ("objrel_T5_DiT_mini_pilot_WDecay", "T5"),
]

def run_evaluation_for_all_models():
    """Run evaluation for all model configurations"""
    results = []
    
    for model_name, encoder_type in MODEL_CONFIGS:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name} with {encoder_type}")
        print(f"{'='*60}")
        
        try:
            # Run the evaluation script
            cmd = [
                "python", "experimental_scripts/posthoc_generation_train_traj_eval_cli.py",
                "--model_run_name", model_name,
                "--text_encoder_type", encoder_type
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            
            if result.returncode == 0:
                print(f"✅ Successfully evaluated {model_name}")
                results.append({
                    "model_name": model_name,
                    "encoder_type": encoder_type,
                    "status": "success",
                    "output": result.stdout
                })
            else:
                print(f"❌ Failed to evaluate {model_name}")
                print(f"Error: {result.stderr}")
                results.append({
                    "model_name": model_name,
                    "encoder_type": encoder_type,
                    "status": "failed",
                    "error": result.stderr
                })
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout for {model_name}")
            results.append({
                "model_name": model_name,
                "encoder_type": encoder_type,
                "status": "timeout",
                "error": "Evaluation timed out"
            })
        except Exception as e:
            print(f"💥 Exception for {model_name}: {e}")
            results.append({
                "model_name": model_name,
                "encoder_type": encoder_type,
                "status": "exception",
                "error": str(e)
            })
    
    return results

def collect_results():
    """Collect all evaluation results from saved files"""
    all_results = []
    
    for model_name, encoder_type in MODEL_CONFIGS:
        # Path to evaluation results
        eval_dir = f"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/analysis_results/{model_name}/large_scale_eval_posthoc"
        
        # Look for trajectory CSV files
        traj_file = f"{eval_dir}/eval_df_all_train_traj_prompts.csv"
        
        if os.path.exists(traj_file):
            print(f"📊 Loading results for {model_name}")
            df = pd.read_csv(traj_file)
            
            # Add model metadata
            df['model_name'] = model_name
            df['encoder_type'] = encoder_type
            
            # Determine model characteristics
            if 'T5' in model_name:
                df['text_encoder'] = 'T5'
            elif 'rndembdposemb' in model_name:
                df['text_encoder'] = 'Random+Position'
            elif 'rndemb' in model_name:
                df['text_encoder'] = 'Random'
            
            if 'B_pilot' in model_name:
                df['model_size'] = 'Base'
            elif 'mini_pilot' in model_name:
                df['model_size'] = 'Mini'
            elif 'micro_pilot' in model_name:
                df['model_size'] = 'Micro'
            elif 'nano_pilot' in model_name:
                df['model_size'] = 'Nano'
            
            if 'WDecay' in model_name:
                df['training'] = 'With Weight Decay'
            else:
                df['training'] = 'Standard'
                
            all_results.append(df)
        else:
            print(f"⚠️  No results found for {model_name}")
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return None

def create_comparison_plots(combined_df):
    """Create comprehensive comparison plots"""
    
    # Create output directory
    output_dir = "Figures/model_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Overall performance comparison
    plt.figure(figsize=(12, 8))
    
    # Get latest step for each model
    latest_results = combined_df.groupby(['model_name', 'text_encoder', 'model_size', 'training']).agg({
        'overall': 'mean',
        'color': 'mean', 
        'shape': 'mean',
        'spatial_relationship': 'mean',
        'exist_binding': 'mean',
        'unique_binding': 'mean'
    }).reset_index()
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    metrics = ['overall', 'color', 'shape', 'spatial_relationship', 'exist_binding', 'unique_binding']
    
    for i, metric in enumerate(metrics):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        # Create grouped bar plot
        pivot_data = latest_results.pivot_table(
            index=['model_size', 'training'],
            columns='text_encoder',
            values=metric,
            aggfunc='mean'
        )
        
        pivot_data.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title(f'{metric.replace("_", " ").title()} Accuracy')
        ax.set_ylabel('Accuracy')
        ax.legend(title='Text Encoder')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison_metrics.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Training trajectory comparison
    plt.figure(figsize=(15, 10))
    
    # Plot training trajectories for each model
    for model_name in combined_df['model_name'].unique():
        model_data = combined_df[combined_df['model_name'] == model_name]
        
        # Get model characteristics for legend
        model_info = model_data.iloc[0]
        legend_label = f"{model_info['model_size']} + {model_info['text_encoder']} + {model_info['training']}"
        
        # Plot trajectory
        plt.plot(model_data['step_num'], model_data['overall'], 
                marker='o', label=legend_label, alpha=0.7)
    
    plt.xlabel('Training Step')
    plt.ylabel('Overall Accuracy')
    plt.title('Training Trajectory Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_trajectories.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Heatmap of final performance
    plt.figure(figsize=(12, 8))
    
    # Create pivot table for heatmap
    heatmap_data = latest_results.pivot_table(
        index=['model_size', 'training'],
        columns='text_encoder',
        values='overall',
        aggfunc='mean'
    )
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis', 
                cbar_kws={'label': 'Overall Accuracy'})
    plt.title('Final Model Performance Comparison')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Save summary statistics
    summary_stats = latest_results.groupby(['text_encoder', 'model_size', 'training']).agg({
        'overall': ['mean', 'std'],
        'color': ['mean', 'std'],
        'shape': ['mean', 'std'],
        'spatial_relationship': ['mean', 'std']
    }).round(3)
    
    summary_stats.to_csv(f"{output_dir}/model_comparison_summary.csv")
    
    return summary_stats

def main():
    """Main execution function"""
    print("🚀 Starting batch evaluation of all 9 models...")
    
    # Step 1: Run evaluations (uncomment if you want to run them)
    # results = run_evaluation_for_all_models()
    
    # Step 2: Collect existing results
    print("\n📊 Collecting existing evaluation results...")
    combined_df = collect_results()
    
    if combined_df is not None:
        print(f"✅ Collected results for {len(combined_df['model_name'].unique())} models")
        print(f"📈 Total data points: {len(combined_df)}")
        
        # Step 3: Create comparison plots
        print("\n📊 Creating comparison plots...")
        summary_stats = create_comparison_plots(combined_df)
        
        print("\n📋 Summary Statistics:")
        print(summary_stats)
        
        # Step 4: Save combined dataset
        combined_df.to_csv("Figures/model_comparison/combined_evaluation_results.csv", index=False)
        print("\n💾 Saved combined results to: Figures/model_comparison/combined_evaluation_results.csv")
        
    else:
        print("❌ No evaluation results found. Please run evaluations first.")

if __name__ == "__main__":
    main() 