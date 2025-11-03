#!/usr/bin/env python3
"""
Detailed model comparison analysis for the 9 model variants
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

def load_and_prepare_data():
    """Load all evaluation results and prepare for analysis"""
    
    # Define model configurations
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
    
    all_results = []
    
    for model_name, encoder_type in MODEL_CONFIGS:
        # Path to evaluation results
        eval_dir = f"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/analysis_results/{model_name}/large_scale_eval_posthoc"
        traj_file = f"{eval_dir}/eval_df_all_train_traj_prompts.csv"
        
        if os.path.exists(traj_file):
            print(f"📊 Loading {model_name}")
            df = pd.read_csv(traj_file)
            
            # Add model metadata
            df['model_name'] = model_name
            df['encoder_type'] = encoder_type
            
            # Parse model characteristics
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
            print(f"⚠️  No results for {model_name}")
    
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        print(f"✅ Loaded {len(combined_df)} data points from {len(combined_df['model_name'].unique())} models")
        return combined_df
    else:
        return None

def analyze_model_performance(combined_df):
    """Analyze performance differences between models"""
    
    # Get final performance (latest step) for each model
    final_performance = combined_df.groupby('model_name').agg({
        'step_num': 'max',
        'overall': 'mean',
        'color': 'mean',
        'shape': 'mean',
        'spatial_relationship': 'mean',
        'exist_binding': 'mean',
        'unique_binding': 'mean',
        'text_encoder': 'first',
        'model_size': 'first',
        'training': 'first'
    }).reset_index()
    
    print("\n" + "="*80)
    print("FINAL MODEL PERFORMANCE COMPARISON")
    print("="*80)
    
    # Sort by overall performance
    final_performance = final_performance.sort_values('overall', ascending=False)
    
    for idx, row in final_performance.iterrows():
        print(f"{row['model_name']:40} | "
              f"Overall: {row['overall']:.3f} | "
              f"Color: {row['color']:.3f} | "
              f"Shape: {row['shape']:.3f} | "
              f"Spatial: {row['spatial_relationship']:.3f} | "
              f"Size: {row['model_size']:6} | "
              f"Encoder: {row['text_encoder']:15} | "
              f"Training: {row['training']}")
    
    return final_performance

def analyze_encoder_effects(combined_df):
    """Analyze the effect of different text encoders"""
    
    print("\n" + "="*80)
    print("TEXT ENCODER COMPARISON")
    print("="*80)
    
    # Group by text encoder and calculate statistics
    encoder_stats = combined_df.groupby('text_encoder').agg({
        'overall': ['mean', 'std', 'count'],
        'color': ['mean', 'std'],
        'shape': ['mean', 'std'],
        'spatial_relationship': ['mean', 'std']
    }).round(3)
    
    print(encoder_stats)
    
    # Statistical significance test
    from scipy import stats
    
    encoders = combined_df['text_encoder'].unique()
    for i, enc1 in enumerate(encoders):
        for enc2 in encoders[i+1:]:
            data1 = combined_df[combined_df['text_encoder'] == enc1]['overall']
            data2 = combined_df[combined_df['text_encoder'] == enc2]['overall']
            
            t_stat, p_value = stats.ttest_ind(data1, data2)
            print(f"\n{enc1} vs {enc2}: t={t_stat:.3f}, p={p_value:.3f}")
    
    return encoder_stats

def analyze_model_size_effects(combined_df):
    """Analyze the effect of model size"""
    
    print("\n" + "="*80)
    print("MODEL SIZE COMPARISON")
    print("="*80)
    
    size_stats = combined_df.groupby('model_size').agg({
        'overall': ['mean', 'std', 'count'],
        'color': ['mean', 'std'],
        'shape': ['mean', 'std'],
        'spatial_relationship': ['mean', 'std']
    }).round(3)
    
    print(size_stats)
    
    return size_stats

def create_comprehensive_plots(combined_df, output_dir="Figures/model_analysis"):
    """Create comprehensive comparison plots"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Performance by encoder type
    plt.figure(figsize=(15, 10))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    metrics = ['overall', 'color', 'shape', 'spatial_relationship', 'exist_binding', 'unique_binding']
    
    for i, metric in enumerate(metrics):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        # Box plot by encoder type
        sns.boxplot(data=combined_df, x='text_encoder', y=metric, ax=ax)
        ax.set_title(f'{metric.replace("_", " ").title()} by Encoder')
        ax.set_ylabel('Accuracy')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/encoder_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Performance by model size
    plt.figure(figsize=(15, 10))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for i, metric in enumerate(metrics):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        # Box plot by model size
        sns.boxplot(data=combined_df, x='model_size', y=metric, ax=ax)
        ax.set_title(f'{metric.replace("_", " ").title()} by Model Size')
        ax.set_ylabel('Accuracy')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_size_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Training trajectory comparison
    plt.figure(figsize=(16, 10))
    
    # Plot trajectories for each model
    for model_name in combined_df['model_name'].unique():
        model_data = combined_df[combined_df['model_name'] == model_name]
        model_info = model_data.iloc[0]
        
        # Create legend label
        legend_label = f"{model_info['model_size']} + {model_info['text_encoder']}"
        if model_info['training'] == 'With Weight Decay':
            legend_label += " (WD)"
        
        # Plot with different colors for encoders
        color_map = {'T5': 'blue', 'Random+Position': 'red', 'Random': 'green'}
        plt.plot(model_data['step_num'], model_data['overall'], 
                marker='o', label=legend_label, color=color_map[model_info['text_encoder']], 
                alpha=0.7, linewidth=2)
    
    plt.xlabel('Training Step')
    plt.ylabel('Overall Accuracy')
    plt.title('Training Trajectories by Model Configuration')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_trajectories.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Heatmap of final performance
    plt.figure(figsize=(12, 8))
    
    # Get final performance for heatmap
    final_perf = combined_df.groupby(['model_name', 'text_encoder', 'model_size', 'training']).agg({
        'overall': 'mean'
    }).reset_index()
    
    heatmap_data = final_perf.pivot_table(
        index=['model_size', 'training'],
        columns='text_encoder',
        values='overall',
        aggfunc='mean'
    )
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis', 
                cbar_kws={'label': 'Overall Accuracy'})
    plt.title('Final Performance Heatmap')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()

def generate_summary_report(combined_df, final_performance, encoder_stats, size_stats):
    """Generate a comprehensive summary report"""
    
    report = []
    report.append("="*80)
    report.append("COMPREHENSIVE MODEL COMPARISON REPORT")
    report.append("="*80)
    
    # Best performing model
    best_model = final_performance.iloc[0]
    report.append(f"\n🏆 BEST PERFORMING MODEL: {best_model['model_name']}")
    report.append(f"   Overall Accuracy: {best_model['overall']:.3f}")
    report.append(f"   Configuration: {best_model['model_size']} + {best_model['text_encoder']} + {best_model['training']}")
    
    # Encoder comparison
    report.append(f"\n📊 TEXT ENCODER PERFORMANCE:")
    for encoder in combined_df['text_encoder'].unique():
        encoder_data = combined_df[combined_df['text_encoder'] == encoder]
        avg_perf = encoder_data['overall'].mean()
        report.append(f"   {encoder}: {avg_perf:.3f}")
    
    # Model size comparison
    report.append(f"\n📏 MODEL SIZE PERFORMANCE:")
    for size in combined_df['model_size'].unique():
        size_data = combined_df[combined_df['model_size'] == size]
        avg_perf = size_data['overall'].mean()
        report.append(f"   {size}: {avg_perf:.3f}")
    
    # Key insights
    report.append(f"\n💡 KEY INSIGHTS:")
    
    # Find best encoder
    best_encoder = encoder_stats['overall']['mean'].idxmax()
    report.append(f"   • Best text encoder: {best_encoder}")
    
    # Find best model size
    best_size = size_stats['overall']['mean'].idxmax()
    report.append(f"   • Best model size: {best_size}")
    
    # Training effect
    training_comparison = combined_df.groupby('training')['overall'].mean()
    if len(training_comparison) > 1:
        best_training = training_comparison.idxmax()
        report.append(f"   • Best training strategy: {best_training}")
    
    # Save report
    with open("Figures/model_analysis/comparison_report.txt", "w") as f:
        f.write("\n".join(report))
    
    print("\n".join(report))
    
    return report

def main():
    """Main analysis function"""
    
    print("🔍 Starting comprehensive model comparison analysis...")
    
    # Load data
    combined_df = load_and_prepare_data()
    
    if combined_df is None:
        print("❌ No data found. Please run evaluations first.")
        return
    
    # Run analyses
    final_performance = analyze_model_performance(combined_df)
    encoder_stats = analyze_encoder_effects(combined_df)
    size_stats = analyze_model_size_effects(combined_df)
    
    # Create plots
    create_comprehensive_plots(combined_df)
    
    # Generate report
    generate_summary_report(combined_df, final_performance, encoder_stats, size_stats)
    
    # Save processed data
    combined_df.to_csv("Figures/model_analysis/processed_comparison_data.csv", index=False)
    final_performance.to_csv("Figures/model_analysis/final_performance_summary.csv", index=False)
    
    print("\n✅ Analysis complete! Check Figures/model_analysis/ for results.")

if __name__ == "__main__":
    main() 