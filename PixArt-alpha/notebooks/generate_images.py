import os
import argparse
import torch
from itertools import islice
from diffusers import PixArtAlphaPipeline
from diffusion.utils.misc import read_config, set_random_seed, \
    init_random_seed, DebugUnderflowOverflow
from utils.pixart_utils import construct_diffuser_pipeline_from_config, construct_pixart_transformer_from_config, state_dict_convert


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-generate images with PixArt-α, controlling sub-batch size."
    )
    parser.add_argument("--prompts_file",  type=str, default="../asset/spatial_test.txt",
                        help="Path to .txt file with one prompt per line.")
    parser.add_argument("--output_dir",    type=str, default="../imgs_temp/",
                        help="Directory to save generated images.")
    parser.add_argument("--cfg_scale",     type=float, default=4.5,
                        help="Classifier-free guidance scale.")
    parser.add_argument("--seed",          type=int,   default=1,
                        help="Random seed for reproducibility.")
    parser.add_argument("--num_inference_steps", type=int, default=14,
                        help="Number of diffusion steps.")
    parser.add_argument("--model_path",    type=str,   default="PixArt-alpha/PixArt-XL-2-512x512",
                        help="Model repo ID for pretrained model.")
    parser.add_argument("--config_path",   type=str,   default=None,
                        help="Path to config.py for local checkpoint.")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to local checkpoint file (.pth). Required if config_path is provided.")
    parser.add_argument("--num_images_per_prompt", type=int, default=1,
                        help="Number of images to generate per prompt.")
    parser.add_argument("--batch_size",    type=int,   default=8,
                        help="How many prompts to process in one sub-batch.")
    return parser.parse_args()

def chunks(iterable, size):
    """Yield successive `size`-sized lists from `iterable`."""
    it = iter(iterable)
    while True:
        batch = list(islice(it, size))
        if not batch:
            return
        yield batch

def main():
    args = parse_args()

    # Device and reproducible seed setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load pipeline either from pretrained or construct from config
    if args.config_path is not None:
        if args.checkpoint_path is None:
            raise ValueError("checkpoint_path must be provided when using config_path")
        if not os.path.exists(args.config_path):
            raise ValueError(f"Config file not found at {args.config_path}")
        if not os.path.exists(args.checkpoint_path):
            raise ValueError(f"Checkpoint file not found at {args.checkpoint_path}")
            
        # Load config and construct pipeline
        config = read_config(args.config_path)
        pipe = construct_diffuser_pipeline_from_config(config, pipeline_class=PixArtAlphaPipeline)
        
        # Load checkpoint
        ckpt = torch.load(args.checkpoint_path)
        pipe.transformer.load_state_dict(state_dict_convert(ckpt['state_dict_ema']))
        print(f"Loaded weights from checkpoint: {args.checkpoint_path}")
    else:
        # Load from pretrained
        pipe = PixArtAlphaPipeline.from_pretrained(args.model_path).to(device)
    
    pipe = pipe.to(device)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    # Read prompts
    with open(args.prompts_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if not prompts:
        raise ValueError("No valid prompts found in prompts_file.")

    os.makedirs(args.output_dir, exist_ok=True)

    idx_offset = 0
    # Process in controlled sub-batches
    for sub_prompts in chunks(prompts, args.batch_size):
        # Single call for this sub-batch (list→batch_size submits B prompts → B×M images) :contentReference[oaicite:5]{index=5}
        output = pipe(
            prompt=sub_prompts,
            guidance_scale=args.cfg_scale,
            num_inference_steps=args.num_inference_steps,
            num_images_per_prompt=args.num_images_per_prompt,
            generator=gen,
        )
        images = output.images  # List[PIL.Image], length = len(sub_prompts)*num_images_per_prompt :contentReference[oaicite:6]{index=6}

        # Save
        for i, prompt in enumerate(sub_prompts):
            for j in range(args.num_images_per_prompt):
                img = images[i * args.num_images_per_prompt + j]
                safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in prompt)[:50]
                filename = f"{safe}_{idx_offset + i:06d}.png"
                path = os.path.join(args.output_dir, filename)
                img.save(path)
                print(f"Saved: {path}")
        idx_offset += len(sub_prompts)

    print("All sub-batches complete. Generated", idx_offset * args.num_images_per_prompt, "images.")

if __name__ == "__main__":
    main()
