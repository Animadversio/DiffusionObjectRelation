import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from collections import defaultdict
import glob

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


def load_multiple_prompt_results(save_dir, ablation_type='layer'):
    """
    Load results from multiple prompts in a directory structure
    
    Args:
        save_dir: Directory containing prompt subfolders
        ablation_type: 'layer' or 'head' to determine which JSON files to load
        
    Returns:
        tuple: (all_baseline_scores, all_ablation_results, prompt_names)
        all_baseline_scores: list of baseline scores from each prompt
        all_ablation_results: list of parsed results from each prompt
        prompt_names: list of prompt names
    """
    if ablation_type == 'layer':
        json_pattern = 'layer_masking.json'
    else:
        json_pattern = 'head_masking.json'
    
    # Find all JSON files in the directory structure
    json_files = glob.glob(os.path.join(save_dir, '**', json_pattern), recursive=True)
    
    if not json_files:
        print(f"No {json_pattern} files found in {save_dir}")
        return [], [], []
    
    print(f"Found {len(json_files)} {json_pattern} files")
    
    all_baseline_scores = []
    all_ablation_results = []
    prompt_names = []
    
    for json_file in json_files:
        try:
            # Extract prompt name from path
            prompt_name = Path(json_file).parent.parent.name
            prompt_names.append(prompt_name)
            
            print(f"Loading: {prompt_name}")
            
            # Load and parse results
            baseline_scores, ablation_results, detected_type = load_and_parse_results(json_file)
            
            # Verify ablation type matches
            if detected_type != ablation_type:
                print(f"Warning: Expected {ablation_type} ablation but found {detected_type} in {json_file}")
                continue
            
            all_baseline_scores.append(baseline_scores)
            all_ablation_results.append(ablation_results)
            
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
    
    return all_baseline_scores, all_ablation_results, prompt_names

def compute_averages_with_errors(all_baseline_scores, all_ablation_results, ablation_type='layer'):
    """
    Compute averages and standard errors across multiple prompts
    
    Args:
        all_baseline_scores: list of baseline scores from each prompt
        all_ablation_results: list of parsed results from each prompt
        ablation_type: 'layer' or 'head'
        
    Returns:
        tuple: (avg_baseline_scores, avg_ablation_results, std_errors)
    """
    if not all_ablation_results:
        return {}, {}, {}
    
    # Get all semantic parts and indices across all prompts
    all_semantic_parts = set()
    all_indices = set()
    
    for ablation_results in all_ablation_results:
        all_semantic_parts.update(ablation_results.keys())
        for semantic_part_data in ablation_results.values():
            all_indices.update(semantic_part_data.keys())
    
    all_semantic_parts = sorted(all_semantic_parts)
    all_indices = sorted(all_indices)
    
    # Define score types
    score_types = {
        'color': 'Color Match',
        'shape': 'Shape Match', 
        'unique_binding': 'Unique Binding',
        'spatial_relationship': 'Spatial Relationship',
    }
    
    # Compute averages and standard errors
    avg_baseline_scores = {}
    avg_ablation_results = defaultdict(dict)
    std_errors = defaultdict(dict)
    
    # Baseline scores
    for score_type in score_types.keys():
        baseline_values = []
        for baseline_scores in all_baseline_scores:
            if score_type in baseline_scores:
                baseline_values.append(baseline_scores[score_type])
        
        if baseline_values:
            avg_baseline_scores[score_type] = np.mean(baseline_values)
    
    # Ablation results
    for semantic_part in all_semantic_parts:
        for index in all_indices:
            for score_type in score_types.keys():
                values = []
                for ablation_results in all_ablation_results:
                    if (semantic_part in ablation_results and 
                        index in ablation_results[semantic_part] and
                        score_type in ablation_results[semantic_part][index]):
                        values.append(ablation_results[semantic_part][index][score_type])
                
                if values:
                    avg_ablation_results[semantic_part][index] = avg_ablation_results[semantic_part].get(index, {})
                    avg_ablation_results[semantic_part][index][score_type] = np.mean(values)
                    
                    std_errors[semantic_part][index] = std_errors[semantic_part].get(index, {})
                    std_errors[semantic_part][index][score_type] = np.std(values) / np.sqrt(len(values))
    
    return dict(avg_baseline_scores), dict(avg_ablation_results), dict(std_errors)

def plot_multi_prompt_averages(save_dir, save_path=None, show_plot=True, ablation_type='layer'):
    """
    Create plots showing averages with standard errors across multiple prompts
    
    Args:
        save_dir: Directory containing prompt subfolders
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
        ablation_type: 'layer' or 'head'
    """
    
    # Load results from all prompts
    all_baseline_scores, all_ablation_results, prompt_names = load_multiple_prompt_results(save_dir, ablation_type)
    
    if not all_ablation_results:
        print(f"No valid {ablation_type} ablation results found!")
        return
    
    print(f"Loaded results from {len(prompt_names)} prompts")
    
    # Compute averages and standard errors
    avg_baseline_scores, avg_ablation_results, std_errors = compute_averages_with_errors(
        all_baseline_scores, all_ablation_results, ablation_type
    )
    
    if not avg_ablation_results:
        print("No averaged results to plot!")
        return
    
    # Define score types and their display names
    score_types = {
        'color': 'Color Match',
        'shape': 'Shape Match', 
        'unique_binding': 'Unique Binding',
        'spatial_relationship': 'Spatial Relationship',
    }
    
    # Define colors for different semantic parts
    semantic_colors = {
        'colors': 'tab:blue',    # Blue
        'objects': 'tab:orange', # Orange
        'spatial': 'tab:green',  # Green
        'all': '#96CEB4',       # Green (if testing all parts together)
        'baseline': 'gray'      # Gray
    }
    
    # Create subplots
    num_metrics = len(score_types)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 4 * num_metrics))
    if num_metrics == 1:
        axes = [axes]  # Ensure axes is iterable
    
    # Get all semantic parts and indices
    all_semantic_parts = list(avg_ablation_results.keys())
    all_indices = sorted(set().union(*[index_dict.keys() for index_dict in avg_ablation_results.values()]))
    
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
        if avg_baseline_scores and score_key in avg_baseline_scores:
            baseline_val = avg_baseline_scores[score_key]
            ax.axhline(baseline_val, color=semantic_colors['baseline'], 
                      linestyle='--', label='Baseline', linewidth=2)
        
        # Plot each semantic part with error bars
        for semantic_part in all_semantic_parts:
            if semantic_part in avg_ablation_results:
                indices = []
                scores = []
                errors = []
                
                for index in all_indices:
                    if (index in avg_ablation_results[semantic_part] and
                        score_key in avg_ablation_results[semantic_part][index]):
                        indices.append(index)
                        scores.append(avg_ablation_results[semantic_part][index][score_key])
                        
                        # Get standard error
                        if (semantic_part in std_errors and 
                            index in std_errors[semantic_part] and
                            score_key in std_errors[semantic_part][index]):
                            errors.append(std_errors[semantic_part][index][score_key])
                        else:
                            errors.append(0)
                
                if indices and scores:
                    color = semantic_colors.get(semantic_part, '#808080')
                    ax.errorbar(indices, scores, yerr=errors, fmt='o-', color=color,
                               label=f'{semantic_part.capitalize()} Masked', 
                               linewidth=2, markersize=6, capsize=5, capthick=2)
        
        # Customize subplot
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
    prompt_count = len(prompt_names)
    fig.suptitle(f'{title_suffix} Results: Average Across {prompt_count} Prompts', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_path:
        base, _ = os.path.splitext(save_path)
        pdf_path = base + '.pdf'
        dir_name = os.path.dirname(pdf_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        print(f"Plot saved to: {pdf_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='Plot layer or head ablation results')
    parser.add_argument('--save_dir', type=str, default="/n/home13/xupan/sompolinsky_lab/object_relation/layer_ablation/rndposemb_mini/",
                       help='Directory containing prompt subfolders (multi-prompt mode)')
    parser.add_argument('--save_plot', type=str, default='plots/multi_prompt_attention_rndposemb_mini_layer_ablation.pdf',
                       help='Path to save the plot (optional)')
    parser.add_argument('--no_show', action='store_true',
                       help='Don\'t display the plot')
    parser.add_argument('--ablation_type', type=str, choices=['layer', 'head'], default='layer',
                       help='Type of ablation to analyze (default: layer)')
    
    args = parser.parse_args()
    
    # Multi-prompt mode
    if not os.path.exists(args.save_dir):
        print(f"Error: Save directory not found at {args.save_dir}")
        return
    
    print(f"Loading results from directory: {args.save_dir}")
    print(f"Using ablation type: {args.ablation_type}")
    
    # Create plots
    plot_multi_prompt_averages(
        save_dir=args.save_dir,
        save_path=args.save_plot,
        show_plot=not args.no_show,
        ablation_type=args.ablation_type
    )
    

if __name__ == "__main__":
    main()

# Example usage:
# For layer ablation:
# python plot_layer_ablation_results.py --json_path /path/to/prompt_dir/saved_metrics/layer_masking.json
# python plot_layer_ablation_results.py --json_path /path/to/layer_masking.json --save_plot layer_results.pdf

# For head ablation:
# python plot_layer_ablation_results.py --json_path /path/to/prompt_dir/saved_metrics/head_masking.json
# python plot_layer_ablation_results.py --json_path /path/to/head_masking.json --save_plot head_results.pdf

# Multi-prompt mode (new functionality):
# python plot_layer_ablation_results.py --save_dir /path/to/layer_ablation --ablation_type layer
# python plot_layer_ablation_results.py --save_dir /path/to/head_ablation --ablation_type head --save_plot multi_prompt_results.pdf 