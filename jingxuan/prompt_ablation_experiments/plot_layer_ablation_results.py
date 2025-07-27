import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from collections import defaultdict

def parse_condition_key(key, ablation_type='layer'):
    """
    Parse condition key to extract layer/head index and semantic part
    
    Args:
        key: String like 'step_5_mask_semantic_parts_attention_colors' (layer)
             or 'step_2_5_mask_semantic_parts_attention_colors' (head)
        ablation_type: 'layer' or 'head' to determine parsing strategy
        
    Returns:
        tuple: (index, semantic_part) or None if baseline/unparseable
        For layer ablation: index is layer_idx
        For head ablation: index is head_idx (layer is ignored)
    """
    if key == 'baseline':
        return None, None
    
    parts = key.split('_')
    
    if ablation_type == 'layer':
        # Expected format: step_{layer}_{mask_func_name}_{part_to_mask}
        # Example: step_5_mask_semantic_parts_attention_colors
        if len(parts) >= 6 and parts[0] == 'step':
            try:
                layer_idx = int(parts[1])
                # The semantic part is the last part
                semantic_part = parts[-1]
                return layer_idx, semantic_part
            except (ValueError, IndexError):
                return None, None
    
    elif ablation_type == 'head':
        # Expected format: step_{layer}_{head}_{mask_func_name}_{part_to_mask}
        # Example: step_2_5_mask_semantic_parts_attention_colors
        if len(parts) >= 7 and parts[0] == 'step':
            try:
                layer_idx = int(parts[1])  # We'll ignore this for plotting
                head_idx = int(parts[2])   # This is what we'll plot
                # The semantic part is the last part
                semantic_part = parts[-1]
                return head_idx, semantic_part
            except (ValueError, IndexError):
                return None, None
    
    return None, None

def detect_ablation_type(json_file_path):
    """
    Detect whether this is layer or head ablation based on filename
    
    Args:
        json_file_path: Path to the JSON file
        
    Returns:
        str: 'layer' or 'head'
    """
    filename = os.path.basename(json_file_path)
    if 'head_masking.json' in filename:
        return 'head'
    elif 'layer_masking.json' in filename:
        return 'layer'
    else:
        # Default to layer for backward compatibility
        print(f"Warning: Could not detect ablation type from filename '{filename}', defaulting to 'layer'")
        return 'layer'

def load_and_parse_results(json_file_path):
    """
    Load JSON results and parse into structured format
    
    Args:
        json_file_path: Path to the layer_masking.json or head_masking.json file
        
    Returns:
        tuple: (baseline_scores, parsed_results, ablation_type)
        parsed_results has structure {semantic_part: {index: scores}}
        where index is layer_idx for layer ablation or head_idx for head ablation
    """
    # Detect ablation type from filename
    ablation_type = detect_ablation_type(json_file_path)
    
    with open(json_file_path, 'r') as f:
        raw_results = json.load(f)
    
    # Get baseline scores
    baseline_scores = raw_results.get('baseline', {})
    
    # Parse results
    parsed_results = defaultdict(dict)  # {semantic_part: {index: scores}}
    
    for key, scores in raw_results.items():
        index, semantic_part = parse_condition_key(key, ablation_type)
        
        if index is not None and semantic_part is not None:
            parsed_results[semantic_part][index] = scores
    
    return baseline_scores, dict(parsed_results), ablation_type

def plot_ablation_results(json_file_path, save_path=None, show_plot=True):
    """
    Create comprehensive plots showing how scores change across layers/heads for different semantic parts
    
    Args:
        json_file_path: Path to the layer_masking.json or head_masking.json file
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
    """
    
    # Load and parse results
    baseline_scores, ablation_results, ablation_type = load_and_parse_results(json_file_path)
    
    if not ablation_results:
        print(f"No {ablation_type} ablation results found in the JSON file!")
        return
    
    # Define score types and their display names
    score_types = {
        'shape_match': 'Shape Match',
        'color_binding': 'Color Binding', 
        'spatial_color_relation': 'Spatial Color Relation',
        'spatial_shape_relation': 'Spatial Shape Relation',
        'overall_score': 'Overall Score'
    }
    
    # Define colors for different semantic parts
    semantic_colors = {
        'colors': 'tab:blue',    # Blue
        'objects': 'tab:orange', # Orange
        'spatial': 'tab:green',  # Green
        'all': '#96CEB4',       # Green (if testing all parts together)
        'baseline': 'gray'      # Gray to match text_prompt_ablation_analysis.py
    }
    
    # Create subplots - vertical stacking like text_prompt_ablation_analysis.py
    num_metrics = len(score_types)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 4 * num_metrics))
    if num_metrics == 1:
        axes = [axes]  # Ensure axes is iterable
    
    # Get all semantic parts and indices (layers or heads)
    all_semantic_parts = list(ablation_results.keys())
    all_indices = sorted(set().union(*[index_dict.keys() for index_dict in ablation_results.values()]))
    
    # Set appropriate labels based on ablation type
    if ablation_type == 'head':
        index_label = 'Head Index'
        title_suffix = 'Head Ablation'
    else:
        index_label = 'Layer Index'
        title_suffix = 'Layer Ablation'
    
    print(f"Found semantic parts: {all_semantic_parts}")
    print(f"Found {ablation_type}s: {all_indices}")
    
    # Plot each score type
    for ax, (score_key, score_name) in zip(axes, score_types.items()):
        
        # Plot baseline as horizontal line if available
        if baseline_scores and score_key in baseline_scores:
            baseline_val = baseline_scores[score_key]
            ax.axhline(baseline_val, color=semantic_colors['baseline'], 
                      linestyle='--', label='Baseline', linewidth=2)
        
        # Plot each semantic part
        for semantic_part in all_semantic_parts:
            if semantic_part in ablation_results:
                indices = []
                scores = []
                
                for index in all_indices:
                    if index in ablation_results[semantic_part]:
                        index_scores = ablation_results[semantic_part][index]
                        if score_key in index_scores:
                            indices.append(index)
                            scores.append(index_scores[score_key])
                
                if indices and scores:
                    color = semantic_colors.get(semantic_part, '#808080')
                    ax.plot(indices, scores, 'o-', color=color, 
                           label=f'{semantic_part.capitalize()} Masked', 
                           linewidth=2, markersize=6)
        
        # Customize subplot to match text_prompt_ablation_analysis.py style
        ax.set_title(f'{score_name}', fontsize=12)
        ax.set_xlabel(index_label)
        ax.set_ylabel(score_key)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set x-axis to show all indices
        if all_indices:
            ax.set_xticks(all_indices)
            ax.set_xlim(min(all_indices) - 0.5, max(all_indices) + 0.5)
    
    # Overall title
    prompt_name = Path(json_file_path).parent.parent.name.replace('_', ' ')
    fig.suptitle(f'{title_suffix} Results: "{prompt_name}"', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot if requested - match text_prompt_ablation_analysis.py style
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()  # Close the figure to free memory if not showing
    
    return fig

def create_summary_table(json_file_path, save_path=None):
    """
    Create a summary table showing the best and worst performing layers/heads for each semantic part
    
    Args:
        json_file_path: Path to the layer_masking.json or head_masking.json file
        save_path: Path to save the table (optional)
    """
    
    baseline_scores, ablation_results, ablation_type = load_and_parse_results(json_file_path)
    
    if not ablation_results:
        print(f"No {ablation_type} ablation results found!")
        return
    
    # Set appropriate column names based on ablation type
    if ablation_type == 'head':
        index_col_best = 'Best Head'
        index_col_worst = 'Worst Head'
    else:
        index_col_best = 'Best Layer'
        index_col_worst = 'Worst Layer'
    
    summary_data = []
    
    for semantic_part, index_data in ablation_results.items():
        if not index_data:
            continue
            
        # Find best and worst performing indices for overall score
        overall_scores = {index: scores.get('overall_score', 0) 
                         for index, scores in index_data.items() 
                         if 'overall_score' in scores}
        
        if overall_scores:
            best_index = max(overall_scores, key=overall_scores.get)
            worst_index = min(overall_scores, key=overall_scores.get)
            
            best_score = overall_scores[best_index]
            worst_score = overall_scores[worst_index]
            
            # Compare with baseline
            baseline_overall = baseline_scores.get('overall_score', 0)
            
            summary_data.append({
                'Semantic Part': semantic_part.capitalize(),
                index_col_best: best_index,
                'Best Score': f"{best_score:.3f}",
                index_col_worst: worst_index, 
                'Worst Score': f"{worst_score:.3f}",
                'Baseline Score': f"{baseline_overall:.3f}",
                'Best vs Baseline': f"{best_score - baseline_overall:+.3f}",
                'Score Range': f"{best_score - worst_score:.3f}"
            })
    
    # Create DataFrame and display
    df = pd.DataFrame(summary_data)
    title = f"{ablation_type.upper()} ABLATION SUMMARY"
    print("\n" + "="*80)
    print(title)
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    # Save table if requested
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"\nSummary table saved to: {save_path}")
    
    return df

def analyze_ablation_patterns(json_file_path):
    """
    Analyze patterns in the layer/head ablation results
    
    Args:
        json_file_path: Path to the layer_masking.json or head_masking.json file
    """
    
    baseline_scores, ablation_results, ablation_type = load_and_parse_results(json_file_path)
    
    title = f"{ablation_type.upper()} ABLATION PATTERN ANALYSIS"
    print("\n" + "="*60)
    print(title)
    print("="*60)
    
    baseline_overall = baseline_scores.get('overall_score', 0)
    print(f"Baseline Overall Score: {baseline_overall:.3f}")
    
    # Set appropriate labels based on ablation type
    if ablation_type == 'head':
        unit_name = 'heads'
        unit_singular = 'Head'
    else:
        unit_name = 'layers'
        unit_singular = 'Layer'
    
    for semantic_part, index_data in ablation_results.items():
        print(f"\n{semantic_part.upper()} MASKING:")
        print("-" * 40)
        
        # Calculate statistics
        overall_scores = [scores.get('overall_score', 0) for scores in index_data.values()]
        
        if overall_scores:
            mean_score = np.mean(overall_scores)
            std_score = np.std(overall_scores)
            
            print(f"  Mean Score: {mean_score:.3f} (±{std_score:.3f})")
            print(f"  vs Baseline: {mean_score - baseline_overall:+.3f}")
            
            # Find indices that improve/hurt performance
            improved_indices = []
            hurt_indices = []
            
            for index, scores in index_data.items():
                overall_score = scores.get('overall_score', 0)
                if overall_score > baseline_overall:
                    improved_indices.append((index, overall_score))
                elif overall_score < baseline_overall:
                    hurt_indices.append((index, overall_score))
            
            if improved_indices:
                improved_indices.sort(key=lambda x: x[1], reverse=True)
                indices_list = [index for index, _ in improved_indices]
                print(f"  {unit_singular}s that improve performance: {indices_list}")
                
            if hurt_indices:
                hurt_indices.sort(key=lambda x: x[1])
                indices_list = [index for index, _ in hurt_indices]
                print(f"  {unit_singular}s that hurt performance: {indices_list}")

def main():
    parser = argparse.ArgumentParser(description='Plot layer or head ablation results')
    parser.add_argument('--json_path', type=str, required=True,
                       help='Path to layer_masking.json or head_masking.json file')
    parser.add_argument('--save_plot', type=str, default=None,
                       help='Path to save the plot (optional)')
    parser.add_argument('--save_table', type=str, default=None, 
                       help='Path to save summary table (optional)')
    parser.add_argument('--no_show', action='store_true',
                       help='Don\'t display the plot')
    parser.add_argument('--analysis_only', action='store_true',
                       help='Only run analysis, skip plotting')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.json_path):
        print(f"Error: JSON file not found at {args.json_path}")
        return
    
    print(f"Loading results from: {args.json_path}")
    
    # Detect and display ablation type
    ablation_type = detect_ablation_type(args.json_path)
    print(f"Detected ablation type: {ablation_type}")
    
    # Run analysis
    analyze_ablation_patterns(args.json_path)
    
    # Create summary table
    create_summary_table(args.json_path, args.save_table)
    
    # Create plots unless analysis_only is specified
    if not args.analysis_only:
        plot_ablation_results(
            json_file_path=args.json_path,
            save_path=args.save_plot, 
            show_plot=not args.no_show
        )

if __name__ == "__main__":
    main()

# Example usage:
# For layer ablation:
# python plot_layer_ablation_results.py --json_path /path/to/prompt_dir/saved_metrics/layer_masking.json
# python plot_layer_ablation_results.py --json_path /path/to/layer_masking.json --save_plot layer_results.png --save_table layer_summary.csv

# For head ablation:
# python plot_layer_ablation_results.py --json_path /path/to/prompt_dir/saved_metrics/head_masking.json
# python plot_layer_ablation_results.py --json_path /path/to/head_masking.json --save_plot head_results.png --save_table head_summary.csv

# Analysis only (no plots):
# python plot_layer_ablation_results.py --json_path /path/to/layer_masking.json --analysis_only 
# python plot_layer_ablation_results.py --json_path /path/to/head_masking.json --analysis_only 