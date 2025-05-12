import os
import re
import pickle
import numpy as np
import torch
import argparse
from pathlib import Path
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description='Calculate mean latents for successful and failed cases')
    parser.add_argument('--block_num', type=int, required=True, help='Block number (0-27)')
    parser.add_argument('--base_dir', type=str, default="/n/home13/xupan/sompolinsky_lab/object_relation/latents",
                      help='Base directory containing seed folders')
    parser.add_argument('--save_dir', type=str, default="classifier_prep_latents",
                      help='Directory to save the mean latents')
    parser.add_argument('--success_map', type=str, default="binary_mask.pkl",
                      help='Path to the success map pickle file')
    
    args = parser.parse_args()
    
    if not 0 <= args.block_num <= 28:
        raise ValueError("Block number must be between 0 and 28")

    # Create save directory if it doesn't exist
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load success/failure map
    with open(args.success_map, "rb") as f:
        success_map = pickle.load(f)

    # Get all prompt IDs from success_map
    valid_prompt_ids = set(success_map.keys())

    # Initialize dictionaries to store arrays
    succ_arrays = defaultdict(list)
    fail_arrays = defaultdict(list)

    # Iterate seed folders & collect arrays
    seed_dirs = sorted(d for d in os.listdir(args.base_dir) if d.startswith("seed"))
    pattern = re.compile(r"spatial_img_latent_residual_allblocks_prompt_(\d+)_seed_(\d+)\.pkl")

    for seed_dir in seed_dirs:
        seed_num = int(seed_dir.replace("seed", ""))
        folder = os.path.join(args.base_dir, seed_dir)
        for fname in os.listdir(folder):
            m = pattern.match(fname)
            if not m:
                continue

            prompt_id = int(m.group(1))
            # Skip if prompt_id is not in success_map
            if prompt_id not in valid_prompt_ids:
                continue

            path = os.path.join(folder, fname)
            
            # Skip empty files
            if os.path.getsize(path) == 0:
                print(f"Skipping empty file: {path}")
                continue

            # Try to unpickle safely
            try:
                with open(path, "rb") as f:
                    latent = pickle.load(f)
            except EOFError:
                print(f"Skipping corrupted/truncated file: {path}")
                continue

            # Get 4D array and average over middle dimensions
            latent_array = latent[f'block_{args.block_num}_residual_spatial_state_traj'][:,1,:,:,:].to(torch.float32).numpy()
            latent_2d = np.mean(latent_array, axis=(1, 2))  # Average over middle dimensions

            is_success = prompt_id in success_map and success_map[prompt_id][seed_num-1] == 1

            if is_success:
                succ_arrays[(prompt_id, seed_num)].append(latent_2d)
            else:
                fail_arrays[(prompt_id, seed_num)].append(latent_2d)

    # Convert lists to numpy arrays for each (prompt_id, seed) pair
    succ_arrays = {k: np.array(v) for k, v in succ_arrays.items()}
    fail_arrays = {k: np.array(v) for k, v in fail_arrays.items()}

    print(f"Number of successful (prompt, seed) pairs: {len(succ_arrays)}")
    print(f"Number of failed (prompt, seed) pairs: {len(fail_arrays)}")

    # Save the arrays as pickle files
    save_path_succ = save_dir / f"succ_arrays_block_{args.block_num}.pkl"
    save_path_fail = save_dir / f"fail_arrays_block_{args.block_num}.pkl"
    
    # Save successful cases
    with open(save_path_succ, 'wb') as f:
        pickle.dump(succ_arrays, f)
    
    # Save failed cases
    with open(save_path_fail, 'wb') as f:
        pickle.dump(fail_arrays, f)
    
    print(f"Saved successful arrays to: {save_path_succ}")
    print(f"Saved failed arrays to: {save_path_fail}")

if __name__ == "__main__":
    main()