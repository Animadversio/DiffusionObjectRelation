import json
import os
from collections import defaultdict

# List of seed files to process
seed_files = [
    "T2I-CompBench-main/examples/labels/2025-05-06_custom_epochunknown_stepunknown_scale4.5_step14_size512_bs8_sampdpm-solver_seed1/annotation_obj_detection_2d/detailed_scores.json",
    "T2I-CompBench-main/examples/labels/2025-05-06_custom_epochunknown_stepunknown_scale4.5_step14_size512_bs8_sampdpm-solver_seed2/annotation_obj_detection_2d/detailed_scores.json",
    "T2I-CompBench-main/examples/labels/2025-05-06_custom_epochunknown_stepunknown_scale4.5_step14_size512_bs8_sampdpm-solver_seed3/annotation_obj_detection_2d/detailed_scores.json",
    "T2I-CompBench-main/examples/labels/2025-05-06_custom_epochunknown_stepunknown_scale4.5_step14_size512_bs8_sampdpm-solver_seed4/annotation_obj_detection_2d/detailed_scores.json",
    "T2I-CompBench-main/examples/labels/2025-05-06_custom_epochunknown_stepunknown_scale4.5_step14_size512_bs8_sampdpm-solver_seed5/annotation_obj_detection_2d/detailed_scores.json"
]

# Dictionary to store entries where both objects are correctly generated
correct_entries = defaultdict(dict)
# Dictionary to track which entries have both objects detected in each seed
detection_status = defaultdict(lambda: defaultdict(bool))

# First pass: collect detection status for all entries across all seeds
for seed_file in seed_files:
    seed_num = int(seed_file.split('seed')[-1].split('/')[0])
    
    with open(seed_file, 'r') as f:
        data = json.load(f)
        
    for entry in data:
        key = f"{entry['prompt']}_{entry['image']}"
        detection_status[key][f'seed_{seed_num}'] = entry['obj1_detected'] and entry['obj2_detected']

# Second pass: only process entries that have both objects detected in ALL seeds
for seed_file in seed_files:
    seed_num = int(seed_file.split('seed')[-1].split('/')[0])
    
    with open(seed_file, 'r') as f:
        data = json.load(f)
        
    for entry in data:
        key = f"{entry['prompt']}_{entry['image']}"
        
        # Check if this entry has both objects detected in ALL seeds
        if all(detection_status[key].values()):
            if key not in correct_entries:
                # Initialize the entry with the first occurrence
                correct_entries[key] = {
                    'question_id': entry['question_id'],
                    'image': entry['image'],
                    'prompt': entry['prompt'],
                    'locality': entry['locality'],
                    'obj1': entry['obj1'],
                    'obj2': entry['obj2'],
                    'obj1_detected': entry['obj1_detected'],
                    'obj2_detected': entry['obj2_detected'],
                    'obj1_confidence': entry['obj1_confidence'],
                    'obj2_confidence': entry['obj2_confidence'],
                    'spatial_relationship_score': {},
                    'final_score': entry['final_score']
                }
            
            # Add the spatial relationship score for this seed
            correct_entries[key]['spatial_relationship_score'][f'seed_{seed_num}'] = entry['spatial_relationship_score']

# Filter entries to keep only those with varying spatial relationship scores
filtered_entries = {}
for key, entry in correct_entries.items():
    spatial_scores = list(entry['spatial_relationship_score'].values())
    # Check if scores show variation (not all above 0.5 or all below 0.5)
    if not (all(score > 0.5 for score in spatial_scores) or all(score < 0.5 for score in spatial_scores)):
        filtered_entries[key] = entry

# Convert the dictionary to a list
combined_scores = list(filtered_entries.values())

# Save the combined scores
output_file = "T2I-CompBench-main/examples/labels/2025-05-06_custom_epochunknown_stepunknown_scale4.5_step14_size512_bs8_sampdpm-solver_seed1/annotation_obj_detection_2d/combined_detailed_scores_correct_obj.json"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, 'w') as f:
    json.dump(combined_scores, f, indent=2)

print(f"Processed {len(combined_scores)} entries where both objects are correctly generated across ALL seeds and show variation in spatial relationship scores")
print(f"Results saved to {output_file}") 