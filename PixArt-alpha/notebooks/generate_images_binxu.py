from diffusers import PixArtAlphaPipeline
import torch
from diffusers import Transformer2DModel
from PIL import Image
import os
from transformers import T5Tokenizer
import argparse
import sys
sys.path.append('..')  # Add parent directory to path to import from utils
from utils.pixart_sampling_utils import PixArtAlphaPipeline_custom, visualize_prompts_with_traj_pretrained, visualize_single_prompt_with_traj


def generate_and_save_images(prompts, pipe, output_dir, cfg_scale=7.0, seed=None, num_inference_steps=50, callback=None, callback_steps=1):
    """
    Generate and save images based on a list of prompts using a pretrained model.

    Args:
        prompts (list of str): List of prompts to generate images.
        pipe: The pretrained model pipeline (e.g., from diffusers).
        output_dir (str): Directory to save the generated images.
        cfg_scale (float): Classifier-free guidance scale. Higher values make the image more closely follow the prompt.
        seed (int, optional): Random seed for reproducibility. If None, a random seed will be used.
        num_inference_steps (int): Number of denoising steps.
        callback (callable, optional): Function to call during inference.
        callback_steps (int): Number of steps between callback calls.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, prompt in enumerate(prompts, start=1):
        # Set random seed if provided
        if seed is not None:
            generator = torch.Generator("cuda").manual_seed(seed)
        else:
            generator = None

        # Generate the image with additional parameters
        output = pipe(
            prompt=prompt,
            generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=cfg_scale,
            callback=callback,
            callback_steps=callback_steps,
            return_sample_pred_traj=True
        )
        
        # Get the image from the output
        image = output[0].images[0] if isinstance(output, tuple) else output.images[0]
        
        # Create a valid filename from the prompt with numbering
        filename_base = "".join(c if c.isalnum() else "_" for c in prompt)
        filename = f"{filename_base}_{idx:06d}.png"
        filepath = os.path.join(output_dir, filename)
        
        # Save the image
        image.save(filepath)
        print(f"Image saved: {filepath}")

def parse_args():
    parser = argparse.ArgumentParser(description='Generate images using PixArt-alpha model')
    
    # Required arguments
    parser.add_argument('--prompts', nargs='+', required=True,
                      help='List of prompts to generate images from')
    parser.add_argument('--output_dir', type=str, default="/n/home13/xupan/sompolinsky_lab/object_relation/t2ibench_imgs_with_intermediates/",
                      help='Directory to save generated images')
    
    # Optional arguments
    parser.add_argument('--cfg_scale', type=float, default=7.0,
                      help='Classifier-free guidance scale (default: 7.0)')
    parser.add_argument('--seed', type=int, default=None,
                      help='Random seed for reproducibility (default: None)')
    parser.add_argument('--num_inference_steps', type=int, default=50,
                      help='Number of denoising steps (default: 50)')
    parser.add_argument('--callback_steps', type=int, default=1,
                      help='Number of steps between callback calls (default: 1)')
    parser.add_argument('--model_path', type=str, default="PixArt-alpha/PixArt-XL-2-512x512",
                      help='Path to the pretrained model (default: PixArt-alpha/PixArt-XL-2-512x512)')
    parser.add_argument('--use_trajectory', action='store_true',
                      help='Use trajectory visualization function instead of basic generation')
    parser.add_argument('--num_images_per_prompt', type=int, default=1,
                      help='Number of images to generate per prompt (default: 1)')

    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Initialize the custom pipeline
    pipe = PixArtAlphaPipeline_custom.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    
    # Example callback function
    def progress_callback(step, timestep, latents):
        print(f"Step {step}/{args.num_inference_steps} completed")
    
    if args.use_trajectory:
        for prompt in args.prompts:
            visualize_single_prompt_with_traj(
                pipeline=pipe,
                prompt=prompt,
                save_dir=args.output_dir,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.cfg_scale,
                device="cuda",
                random_seed=args.seed,
                weight_dtype=torch.float16
            )
    else:
        # Use the basic generation function
        generate_and_save_images(
            prompts=args.prompts,
            pipe=pipe,
            output_dir=args.output_dir,
            cfg_scale=args.cfg_scale,
            seed=args.seed,
            num_inference_steps=args.num_inference_steps,
            callback=progress_callback,
            callback_steps=args.callback_steps
        )

if __name__ == "__main__":
    main() 