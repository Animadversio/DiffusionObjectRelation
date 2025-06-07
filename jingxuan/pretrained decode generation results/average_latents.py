import os
import re
import pickle
import numpy as np
import torch
import argparse
from pathlib import Path

def update_running_mean(running_mean, count, new_sample):
    """
    One‐step Welford update for the mean.
    running_mean : np.ndarray  (same shape as new_sample)
    count        : int         (number of samples seen so far)
    new_sample   : np.ndarray  (the incoming sample)
    Returns: (updated_mean, updated_count)
    """
    count += 1
    # Δ = x - μ_prev
    delta = new_sample - running_mean
    # μ_new = μ_prev + Δ / count
    running_mean += delta / count
    return running_mean, count

def main():
    parser = argparse.ArgumentParser(description='Calculate mean latents for successful and failed cases')
    parser.add_argument('--block_num', type=int, required=True, help='Block number (0-27)')
    parser.add_argument('--base_dir', type=str, default="/n/home13/xupan/sompolinsky_lab/object_relation/latents",
                      help='Base directory containing seed folders')
    parser.add_argument('--save_dir', type=str, default="mean_latents",
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

    # Initialize storage for running means & counts
    mean_succ = None
    count_succ = 0
    mean_fail = None
    count_fail = 0

    # Iterate seed folders & update per-prompt stats
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

            latent_array = latent[f'block_{args.block_num}_residual_spatial_state_traj'][:,1,:,:,:].to(torch.float32).numpy()
            latent = latent_array

            is_success = prompt_id in success_map and seed_num in success_map[prompt_id]

            if is_success:
                if mean_succ is None:
                    mean_succ = np.zeros_like(latent)
                mean_succ, count_succ = update_running_mean(mean_succ, count_succ, latent)
            else:
                if mean_fail is None:
                    mean_fail = np.zeros_like(latent)
                mean_fail, count_fail = update_running_mean(mean_fail, count_fail, latent)

    print(f"Succeeded seeds aggregated: {count_succ}")
    print(f"Failed    seeds aggregated: {count_fail}")

    # Save the mean latents and counts as pickle files
    save_path_succ = save_dir / f"mean_succ_block_{args.block_num}.pkl"
    save_path_fail = save_dir / f"mean_fail_block_{args.block_num}.pkl"
    
    # Save successful cases
    with open(save_path_succ, 'wb') as f:
        pickle.dump({
            'mean': mean_succ,
            'count': count_succ
        }, f)
    
    # Save failed cases
    with open(save_path_fail, 'wb') as f:
        pickle.dump({
            'mean': mean_fail,
            'count': count_fail
        }, f)
    
    print(f"Saved successful mean latents to: {save_path_succ}")
    print(f"Saved failed mean latents to: {save_path_fail}")

if __name__ == "__main__":
    main()