import os
import argparse
import torch
from itertools import islice
from diffusers import PixArtAlphaPipeline

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-generate images with PixArt-α, controlling sub-batch size."
    )
    parser.add_argument("--prompts_file",  type=str, default="../asset/spatial_sample.txt",
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
                        help="Model repo ID or local path for PixArt-α.")
    parser.add_argument("--num_images_per_prompt", type=int, default=1,
                        help="Number of images to generate per prompt.")
    parser.add_argument("--batch_size",    type=int,   default=8,
                        help="How many prompts to process in one sub-batch.")  # new
    return parser.parse_args()

def chunks(lst, n):
    it = iter(lst)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pipeline once
    pipe = PixArtAlphaPipeline.from_pretrained(args.model_path).to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Prepare directories
    seed_dir = os.path.join(args.output_dir, f"seed_{args.seed}")
    os.makedirs(seed_dir, exist_ok=True)
    intermediates_dir = os.path.join(seed_dir, "intermediates")
    os.makedirs(intermediates_dir, exist_ok=True)

    # Read prompts
    with open(args.prompts_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    idx_offset = 0
    for sub_prompts in chunks(prompts, args.batch_size):
        # Define a callback that captures the current offset
        def callback_fn(step_idx, timestep, latents):
            # Decode latents 🔄
            scaled = latents / pipe.vae.config.scaling_factor
            decoded, = pipe.vae.decode(scaled, return_dict=False)
            imgs = pipe.image_processor.postprocess(
                decoded,
                output_type="pil",
                do_denormalize=[True] * decoded.shape[0]
            )

            # Save each intermediate image
            step_dir = os.path.join(intermediates_dir, f"step_{step_idx:04d}")
            os.makedirs(step_dir, exist_ok=True)
            for i, img in enumerate(imgs):
                global_idx = idx_offset + i
                prompt = sub_prompts[i]
                safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in prompt)[:50]
                filename = f"{safe}_{global_idx:06d}.png"
                img.save(os.path.join(step_dir, filename))

        # Run the pipeline for this sub-batch, invoking our callback every step
        output = pipe(
            prompt=sub_prompts,
            guidance_scale=args.cfg_scale,
            num_inference_steps=args.num_inference_steps,
            num_images_per_prompt=args.num_images_per_prompt,
            generator=generator,
            callback=callback_fn,              # attach callback :contentReference[oaicite:9]{index=9}
            callback_steps=1                   # every step :contentReference[oaicite:10]{index=10}
        )

        # Save final images as before
        for i, prompt in enumerate(sub_prompts):
            for j, img in enumerate(output.images[i * args.num_images_per_prompt:(i+1) * args.num_images_per_prompt]):
                safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in prompt)[:50]
                filename = f"{safe}_{idx_offset + i:06d}.png"
                img.save(os.path.join(seed_dir, filename))

        idx_offset += len(sub_prompts)

    print("Done! Final images in", seed_dir,
          "and intermediates in", intermediates_dir)

if __name__ == "__main__":
    main()
