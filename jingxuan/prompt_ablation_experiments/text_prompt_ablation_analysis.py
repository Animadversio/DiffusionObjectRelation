import json
import matplotlib.pyplot as plt
from collections import defaultdict
import re
import os
import argparse
import numpy as np

def extract_prompt_components(prompt_name):
    """
    Extract color and object components from prompt name.
    Expected format: {color1}_{object1}_is_{relation}_{of}_{color2}_{object2}
    
    Parameters:
    -----------
    prompt_name : str
        The prompt name (e.g., "red_triangle_is_to_the_right_of_blue_square")
    
    Returns:
    --------
    tuple: (color1, object1, color2, object2) or None if parsing fails
    """
    # Replace underscores with spaces and split
    parts = prompt_name.replace('_', ' ').split()
    
    try:
        # Find the index of "is" to split the prompt
        is_index = parts.index("is")
        
        # First part: color1 + object1
        first_part = parts[:is_index]
        if len(first_part) >= 2:
            color1 = first_part[0]
            object1 = first_part[1]
        else:
            return None
            
        # Find "of" to locate the second object
        of_index = None
        for i in range(is_index + 1, len(parts)):
            if parts[i] == "of":
                of_index = i
                break
        
        if of_index is None:
            return None
            
        # Second part: color2 + object2 (after "of")
        second_part = parts[of_index + 1:]
        if len(second_part) >= 2:
            color2 = second_part[0]
            object2 = second_part[1]
        else:
            return None
            
        return (color1, object1, color2, object2)
    except (ValueError, IndexError):
        return None

def meets_criteria(prompt_name):
    """
    Check if prompt meets the criteria:
    - First and second color differ
    - First and second object differ
    
    Parameters:
    -----------
    prompt_name : str
        The prompt name to check
    
    Returns:
    --------
    bool: True if meets criteria, False otherwise
    """
    components = extract_prompt_components(prompt_name)
    if components is None:
        return False
    
    color1, object1, color2, object2 = components
    
    # Check if colors differ and objects differ
    return color1 != color2 and object1 != object2

def plot_manipulation_curves(save_dir, name="one_step.json"):
    """
    Load metrics from multiple JSON files in prompt directories and plot subplots for each metric:
      - 'original' values are shown as a horizontal line.
      - Other keys of the form 'step_<n>_<manipulation>' are grouped by manipulation
        and plotted as a line over step numbers.
    Only includes prompts where first/second colors differ AND first/second objects differ.
    
    Parameters:
    -----------
    save_dir : str
        Path to the directory containing prompt folders.
    name : str
        Name of the JSON file (including .json extension).
    """
    # Collect all qualifying data
    all_data = {}
    qualifying_prompts = []
    
    # Loop through all directories in save_dir
    for prompt_folder in os.listdir(save_dir):
        prompt_path = os.path.join(save_dir, prompt_folder)
        
        # Skip if not a directory
        if not os.path.isdir(prompt_path):
            continue
            
        # Check if prompt meets criteria
        if not meets_criteria(prompt_folder):
            print(f"Skipping {prompt_folder}: does not meet criteria")
            continue
            
        # Check if saved_metrics/one_step.json exists
        json_path = os.path.join(prompt_path, 'saved_metrics', name)
        if not os.path.exists(json_path):
            print(f"Skipping {prompt_folder}: no {json_path} found")
            continue
            
        # Load the JSON data
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            all_data[prompt_folder] = data
            qualifying_prompts.append(prompt_folder)
            print(f"Loaded data from {prompt_folder}")
        except Exception as e:
            print(f"Error loading {json_path}: {e}")
            continue
    
    if not all_data:
        print("No qualifying data found!")
        return
    
    print(f"Found {len(qualifying_prompts)} qualifying prompts: {qualifying_prompts}")
    
    # Combine data from all prompts
    combined_original = {}
    combined_manip_data = defaultdict(list)
    
    # Aggregate manipulation data first to get metric keys
    for prompt_name, prompt_data in all_data.items():
        for key, metrics in prompt_data.items():
            if key == 'original':
                continue
            match = re.match(r'step_(\d+)_(.+)', key)
            if not match:
                continue
            step_num = int(match.group(1))
            manipulation = match.group(2)
            combined_manip_data[manipulation].append((step_num, metrics, prompt_name))

    # Get metric keys from manipulation data or original data
    first_data = next(iter(all_data.values()))
    metric_keys = []
    
    # Try to get metric keys from original data first
    if 'original' in first_data and first_data['original']:
        metric_keys = list(first_data['original'].keys())
        print(f"Metric keys found in original data: {metric_keys}")
    
    # If no original data or empty, get metric keys from manipulation data
    if not metric_keys and combined_manip_data:
        # Get metric keys from the first manipulation entry
        for manipulation_list in combined_manip_data.values():
            if manipulation_list:
                _, first_metrics, _ = manipulation_list[0]
                metric_keys = list(first_metrics.keys())
                break
        print(f"Metric keys found in manipulation data: {metric_keys}")
    
    # Check if we have any metrics
    if not metric_keys:
        print("Error: No metrics found in either original or manipulation data.")
        return
    
    # Aggregate original values if available (take average across prompts)
    has_original_data = False
    combined_original_std = {}
    for metric in metric_keys:
        values = []
        for prompt_data in all_data.values():
            if 'original' in prompt_data and metric in prompt_data['original']:
                values.append(prompt_data['original'][metric])
        if values:
            combined_original[metric] = np.mean(values)
            if len(values) > 1:
                combined_original_std[metric] = np.std(values, ddof=1) / np.sqrt(len(values))
            else:
                combined_original_std[metric] = 0
            has_original_data = True
    
    if has_original_data:
        print("Original baseline data found and will be plotted")
    else:
        print("No original baseline data found - only plotting manipulation curves")

    # Create subplots
    num_metrics = len(metric_keys)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 4 * num_metrics))
    if num_metrics == 1:
        axes = [axes]  # Ensure axes is iterable

    for ax, metric in zip(axes, metric_keys):
        # Plot original baseline if available
        if metric in combined_original:
            baseline_mean = combined_original[metric]
            baseline_se = combined_original_std.get(metric, 0)
            
            # Plot baseline as horizontal line
            ax.axhline(baseline_mean, color='gray', linestyle='--', label='original', linewidth=2)
            
            # Add shaded region for standard error if available
            if baseline_se > 0:
                ax.axhspan(baseline_mean - baseline_se, baseline_mean + baseline_se, 
                          color='gray', alpha=0.2, zorder=0)

        # Plot each manipulation curve (averaged across prompts)
        for manipulation, values in combined_manip_data.items():
            # Group by step number and collect all values across prompts
            step_groups = defaultdict(list)
            for step_num, metrics, prompt_name in values:
                if metric in metrics:
                    step_groups[step_num].append(metrics[metric])
            
            # Calculate averages and standard errors for each step
            steps = []
            scores = []
            errors = []
            for step_num in sorted(step_groups.keys()):
                step_values = step_groups[step_num]
                if step_values:
                    steps.append(step_num)
                    mean_score = np.mean(step_values)
                    scores.append(mean_score)
                    # Calculate standard error (std / sqrt(n))
                    if len(step_values) > 1:
                        std_error = np.std(step_values, ddof=1) / np.sqrt(len(step_values))
                    else:
                        std_error = 0
                    errors.append(std_error)
            
            if steps:
                # Plot line with error bars
                ax.errorbar(steps, scores, yerr=errors, marker='o', label=manipulation, 
                           linewidth=2, markersize=6, capsize=4, capthick=1)

        ax.set_title(f'{metric} (mean ± SE across {len(qualifying_prompts)} prompts)', fontsize=12)
        ax.set_xlabel('Step Number')
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Save the plot to current directory
    plot_filename = f'multi_prompt_{os.path.splitext(name)[0]}_manipulation_curves.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    
    print(f"Plot saved to: {plot_filename}")
    print(f"Included prompts: {', '.join(qualifying_prompts)}")


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--save_dir', type=str, default='/n/home13/xupan/sompolinsky_lab/object_relation/prompt_ablation',
                       help='Directory containing prompt folders with saved_metrics')
    parser.add_argument('--name', type=str, default='one_step.json',
                       help='Name of the JSON file (including .json extension)')
    
    args = parser.parse_args()
    
    # Validate that the directory exists
    if not os.path.isdir(args.save_dir):
        print(f"Error: Directory '{args.save_dir}' does not exist.")
        return 1
    
    plot_manipulation_curves(args.save_dir, args.name)


if __name__ == '__main__':
    main()


