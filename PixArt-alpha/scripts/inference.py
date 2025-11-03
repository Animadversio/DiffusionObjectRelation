import os
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
import warnings
warnings.filterwarnings("ignore")  # ignore warning
import re
import argparse
from datetime import datetime
from tqdm import tqdm
import torch
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
import spacy

from diffusion.model.utils import prepare_prompt_ar
from diffusion import IDDPM, DPMS, SASolverSampler
from tools.download import find_model
from diffusion.model.nets import PixArtMS_XL_2, PixArt_XL_2
from diffusion.model.t5 import T5Embedder
from diffusion.data.datasets import get_chunks, ASPECT_RATIO_512_TEST, ASPECT_RATIO_1024_TEST


def detect_spatial_relationships(prompt):
    """
    Detect spatial relationship terms in a prompt.
    
    Args:
        prompt: The text prompt
    
    Returns:
        List of spatial relationship terms found in the prompt
    """
    spatial_patterns = [
        r"on the top of",
        r"on the bottom of", 
        r"on the left of",
        r"on the right of",
        r"next to",
        r"near",
        r"on side of"
    ]
    
    spatial_terms = []
    for pattern in spatial_patterns:
        matches = re.finditer(pattern, prompt, re.IGNORECASE)
        for match in matches:
            spatial_terms.append(match.group())
    
    return spatial_terms


def mask_words_in_prompt(prompt, words_to_mask, t5_embedder):
    """
    Takes a prompt and masks specified words by setting their attention mask to zero.
    
    Args:
        prompt: The text prompt
        words_to_mask: List of words to mask in the prompt
        t5_embedder: The T5 text embedder with tokenizer
    
    Returns:
        Modified prompt_attn_mask with specified words masked, and mask indices
    """
    if isinstance(words_to_mask, str):
        words_to_mask = [words_to_mask]
    
    # Get the tokenizer from t5_embedder
    tokenizer = t5_embedder.tokenizer
    
    # OLD METHOD: Tokenize the prompt (without special tokens)
    # tokens = tokenizer.tokenize(prompt)
    # print(f"Tokens: {tokens}")
    # token_ids = tokenizer.convert_tokens_to_ids(tokens)
    # print(f"Token IDs: {token_ids}")
    
    # NEW METHOD: Tokenize the prompt using the same method as T5Embedder
    text_tokens_and_mask = tokenizer(
        [prompt],  # Pass as list like T5Embedder does
        max_length=t5_embedder.model_max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )
    #print(f"Text tokens and mask: {text_tokens_and_mask}")
    token_ids = text_tokens_and_mask['input_ids'][0].tolist()  # Get first (and only) sequence
    #print(f"Full tokenized IDs with special tokens: {token_ids}")
    
    # Create initial attention mask (all ones)
    prompt_attn_mask = torch.ones(1, len(token_ids), dtype=torch.bool)
    mask_idx = []
    
    # Find token IDs for words to mask
    for word_to_mask in words_to_mask:
        word_token_ids = tokenizer.encode(word_to_mask, add_special_tokens=False)
        print(f"Word '{word_to_mask}' token IDs: {word_token_ids}")
        # Find indices of token IDs to mask
        for i, token_id in enumerate(token_ids):
            if token_id in word_token_ids:
                prompt_attn_mask[0, i] = 0
                mask_idx.append(i)
    
    mask_idx = sorted(mask_idx)
    print(f"Mask indices: {mask_idx}")
    return prompt_attn_mask, mask_idx


def mask_eos_in_prompt(prompt, t5_embedder):
    """
    Find and mask end-of-sentence (EOS) token(s) in the tokenized prompt.
    Returns an attention mask and list of indices corresponding to EOS tokens.
    """
    tokenizer = t5_embedder.tokenizer
    text_tokens_and_mask = tokenizer(
        [prompt],
        max_length=t5_embedder.model_max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )
    token_ids = text_tokens_and_mask['input_ids'][0].tolist()
    eos_id = tokenizer.eos_token_id
    prompt_attn_mask = torch.ones(1, len(token_ids), dtype=torch.bool)
    mask_idx = []
    for i, token_id in enumerate(token_ids):
        if token_id == eos_id:
            prompt_attn_mask[0, i] = 0
            mask_idx.append(i)
    return prompt_attn_mask, mask_idx


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', default=512, type=int)
    parser.add_argument('--t5_path', default='../output/pretrained_models/t5_ckpts', type=str)
    parser.add_argument('--tokenizer_path', default='../output/pretrained_models/sd-vae-ft-ema', type=str)
    parser.add_argument('--txt_file', default='../asset/high_accuracy_spatial_prompt.txt', type=str)
    parser.add_argument('--model_path', default='../output/pretrained_models/PixArt-XL-2-512x512.pth', type=str)
    parser.add_argument('--bs', default=8, type=int)
    parser.add_argument('--cfg_scale', default=4.5, type=float)
    parser.add_argument('--sampling_algo', default='dpm-solver', type=str, choices=['iddpm', 'dpm-solver', 'sa-solver'])
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--dataset', default='custom', type=str)
    parser.add_argument('--step', default=14, type=int)
    parser.add_argument('--img_save_dir', default='/n/home13/xupan/sompolinsky_lab/object_relation/t2ibench_imgs_with_eos_masking/', type=str)
    parser.add_argument('--enable_masking', action='store_true', help='Enable spatial relationship masking')
    parser.add_argument('--mask_mode', default='spatial', type=str, choices=['spatial', 'object1', 'object2', 'eos'], help="Which part to mask when masking is enabled ('spatial', 'object1', 'object2', 'eos')")
    #parser.add_argument('--save_name', default='test_sample', type=str)

    return parser.parse_args()


def set_env(seed=0):
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    for _ in range(30):
        torch.randn(1, 4, args.image_size, args.image_size)


@torch.inference_mode()
def visualize(items, bs, sample_steps, cfg_scale, enable_masking=False):

    # Add a counter for image numbering
    image_counter = 0
    
    for chunk in tqdm(list(get_chunks(items, bs)), unit='batch'):

        prompts = []
        masked_prompts = []  # Store info about masking for filenames
        
        if bs == 1:
            prompt_clean, _, hw, ar, custom_hw = prepare_prompt_ar(chunk[0], base_ratios, device=device, show=False)  # ar for aspect ratio
            if args.image_size == 1024:
                latent_size_h, latent_size_w = int(hw[0, 0] // 8), int(hw[0, 1] // 8)
            else:
                hw = torch.tensor([[args.image_size, args.image_size]], dtype=torch.float, device=device).repeat(bs, 1)
                ar = torch.tensor([[1.]], device=device).repeat(bs, 1)
                latent_size_h, latent_size_w = latent_size, latent_size
            prompts.append(prompt_clean.strip())
        else:
            hw = torch.tensor([[args.image_size, args.image_size]], dtype=torch.float, device=device).repeat(bs, 1)
            ar = torch.tensor([[1.]], device=device).repeat(bs, 1)
            for prompt in chunk:
                prompts.append(prepare_prompt_ar(prompt, base_ratios, device=device, show=False)[0].strip())
            latent_size_h, latent_size_w = latent_size, latent_size

        null_y = model.y_embedder.y_embedding[None].repeat(len(prompts), 1, 1)[:, None]

        with torch.no_grad():
            # Handle masking if enabled
            if enable_masking:
                caption_embs_list = []
                emb_masks_list = []
                
                # Lazy-load spaCy model once
                nlp = getattr(visualize, "_spacy_nlp", None)
                if nlp is None:
                    try:
                        nlp = spacy.load("en_core_web_sm")
                    except Exception:
                        nlp = None
                    setattr(visualize, "_spacy_nlp", nlp)
                
                for prompt in prompts:
                    # Detect spatial relationships
                    spatial_terms = detect_spatial_relationships(prompt)
                    #print(f"Spatial terms: {spatial_terms}")
                    
                    # Extract objects relative to spatial term if possible
                    obj1_text, obj2_text = None, None
                    if spatial_terms:
                        # Use the first spatial phrase occurrence
                        first_match = None
                        for term in spatial_terms:
                            m = re.search(re.escape(term), prompt, flags=re.IGNORECASE)
                            if m:
                                first_match = (m.start(), m.end(), term)
                                break
                        if first_match is not None:
                            left_text = prompt[:first_match[0]].strip()
                            right_text = prompt[first_match[1]:].strip()
                            # heuristic cleanup
                            def simple_tokens(text):
                                return [t for t in re.split(r"[^A-Za-z0-9'-]+", text) if t]
                            if nlp is not None:
                                try:
                                    left_doc = nlp(left_text)
                                    right_doc = nlp(right_text)
                                    # last noun/proper noun on left, first on right
                                    left_nouns = [t.text for t in left_doc if t.pos_ in ['NOUN','PROPN']]
                                    right_nouns = [t.text for t in right_doc if t.pos_ in ['NOUN','PROPN']]
                                    if len(left_nouns) > 0:
                                        obj1_text = left_nouns[-1]
                                    if len(right_nouns) > 0:
                                        obj2_text = right_nouns[0]
                                except Exception:
                                    pass
                            if obj1_text is None:
                                ltoks = simple_tokens(left_text)
                                if len(ltoks) > 0:
                                    obj1_text = ltoks[-1]
                            if obj2_text is None:
                                rtoks = simple_tokens(right_text)
                                if len(rtoks) > 0:
                                    obj2_text = rtoks[0]
                    else:
                        # Fallback: try to pick first and last nouns from entire prompt
                        if nlp is not None:
                            try:
                                doc = nlp(prompt)
                                nouns = [t.text for t in doc if t.pos_=='NOUN']
                                if len(nouns) >= 1:
                                    obj1_text = nouns[0]
                                if len(nouns) >= 2:
                                    obj2_text = nouns[-1]
                            except Exception:
                                pass
                    
                    # Decide which tokens/words to mask based on mode
                    mode = getattr(args, 'mask_mode', 'spatial')
                    if mode == 'eos':
                        # Mask EOS token(s)
                        prompt_attn_mask, mask_idx = mask_eos_in_prompt(prompt, t5)
                        caption_emb, emb_mask = t5.get_text_embeddings([prompt])
                        if len(mask_idx) > 0:
                            for idx in mask_idx:
                                if idx < caption_emb.shape[1]:
                                    caption_emb[:, idx, :] = 0
                            masked_prompts.append(f"{prompt}_MASKED_{mode}")
                        else:
                            masked_prompts.append(f"{prompt}_NO_MASK_{mode}")
                    else:
                        words_to_mask = []
                        if mode == 'spatial' and spatial_terms:
                            words_to_mask = spatial_terms
                        elif mode == 'object1' and obj1_text is not None:
                            words_to_mask = [obj1_text]
                        elif mode == 'object2' and obj2_text is not None:
                            words_to_mask = [obj2_text]

                        if len(words_to_mask) > 0:
                            # Create masked version
                            prompt_attn_mask, mask_idx = mask_words_in_prompt(prompt, words_to_mask, t5)
                            caption_emb, emb_mask = t5.get_text_embeddings([prompt])
                            if len(mask_idx) > 0:
                                for idx in mask_idx:
                                    if idx < caption_emb.shape[1]:
                                        caption_emb[:, idx, :] = 0
                            masked_prompts.append(f"{prompt}_MASKED_{mode}")
                        else:
                            # No terms to mask for this mode
                            caption_emb, emb_mask = t5.get_text_embeddings([prompt])
                            masked_prompts.append(f"{prompt}_NO_MASK_{mode}")
                    
                    caption_embs_list.append(caption_emb)
                    emb_masks_list.append(emb_mask)
                
                # Concatenate all embeddings
                caption_embs = torch.cat(caption_embs_list, dim=0)
                emb_masks = torch.cat(emb_masks_list, dim=0) if emb_masks_list[0] is not None else None
            else:
                # Original behavior - no masking
                caption_embs, emb_masks = t5.get_text_embeddings(prompts)
                masked_prompts = [f"{prompt}_NO_MASKING" for prompt in prompts]
            
            caption_embs = caption_embs.float()[:, None]
            print('finish embedding')

            if args.sampling_algo == 'iddpm':
                # Create sampling noise:
                n = len(prompts)
                z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device).repeat(2, 1, 1, 1)
                model_kwargs = dict(y=torch.cat([caption_embs, null_y]),
                                    cfg_scale=cfg_scale, data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
                diffusion = IDDPM(str(sample_steps))
                # Sample images:
                samples = diffusion.p_sample_loop(
                    model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
                    device=device
                )
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            elif args.sampling_algo == 'dpm-solver':
                # Create sampling noise:
                n = len(prompts)
                z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device)
                model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
                dpm_solver = DPMS(model.forward_with_dpmsolver,
                                  condition=caption_embs,
                                  uncondition=null_y,
                                  cfg_scale=cfg_scale,
                                  model_kwargs=model_kwargs)
                samples = dpm_solver.sample(
                    z,
                    steps=sample_steps,
                    order=2,
                    skip_type="time_uniform",
                    method="multistep"
                )
            elif args.sampling_algo == 'sa-solver':
                # Create sampling noise:
                n = len(prompts)
                model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
                sa_solver = SASolverSampler(model.forward_with_dpmsolver, device=device)
                samples = sa_solver.sample(
                    S=25,
                    batch_size=n,
                    shape=(4, latent_size_h, latent_size_w),
                    eta=1,
                    conditioning=caption_embs,
                    unconditional_conditioning=null_y,
                    unconditional_guidance_scale=cfg_scale,
                    model_kwargs=model_kwargs
                )[0]
        samples = vae.decode(samples / 0.18215).sample
        torch.cuda.empty_cache()
        # Save images:
        os.umask(0o000)  # file permission: 666; dir permission: 777
        for i, sample in enumerate(samples):
            # Increment counter for each image
            image_counter += 1
            
            # Create 6-digit suffix with leading zeros
            suffix = f"_{image_counter:06d}"
            
            # Use masked prompt info for filename
            filename_base = masked_prompts[i][:100] if enable_masking else prompts[i][:100]
            
            # Create filename with the suffix and .png extension
            save_path = os.path.join(save_root, f"{filename_base}{suffix}.png")
            print("Saving path: ", save_path)
            save_image(sample, save_path, nrow=1, normalize=True, value_range=(-1, 1))


if __name__ == '__main__':
    args = get_args()
    # Setup PyTorch:
    seed = args.seed
    set_env(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert args.sampling_algo in ['iddpm', 'dpm-solver', 'sa-solver']

    # only support fixed latent size currently
    latent_size = args.image_size // 8
    lewei_scale = {512: 1, 1024: 2}     # trick for positional embedding interpolation
    sample_steps_dict = {'iddpm': 100, 'dpm-solver': 20, 'sa-solver': 25}
    sample_steps = args.step if args.step != -1 else sample_steps_dict[args.sampling_algo]
    weight_dtype = torch.bfloat16
    print(f"Inference with {weight_dtype}")

    # model setting
    if args.image_size == 512:
        model = PixArt_XL_2(input_size=latent_size, lewei_scale=lewei_scale[args.image_size]).to(device)
    else:
        model = PixArtMS_XL_2(input_size=latent_size, lewei_scale=lewei_scale[args.image_size]).to(device)

    print(f"Generating sample from ckpt: {args.model_path}")
    state_dict = find_model(args.model_path)
    del state_dict['state_dict']['pos_embed']
    missing, unexpected = model.load_state_dict(state_dict['state_dict'], strict=False)
    print('Missing keys: ', missing)
    print('Unexpected keys', unexpected)
    model.eval()
    model.to(weight_dtype)
    base_ratios = eval(f'ASPECT_RATIO_{args.image_size}_TEST')

    vae = AutoencoderKL.from_pretrained(args.tokenizer_path).to(device)
    t5 = T5Embedder(device="cuda", local_cache=True, cache_dir=args.t5_path, torch_dtype=torch.float)
    #work_dir = os.path.join(*args.model_path.split('/')[:-2])
    #work_dir = f'/{work_dir}' if args.model_path[0] == '/' else work_dir

    # data setting
    with open(args.txt_file, 'r') as f:
        items = [item.strip() for item in f.readlines()]

    # img save setting
    try:
        epoch_name = re.search(r'.*epoch_(\d+).*.pth', args.model_path).group(1)
        step_name = re.search(r'.*step_(\d+).*.pth', args.model_path).group(1)
    except Exception:
        epoch_name = 'unknown'
        step_name = 'unknown'
    #img_save_dir = os.path.join(work_dir, 'vis')
    img_save_dir = args.img_save_dir
    os.umask(0o000)  # file permission: 666; dir permission: 777
    os.makedirs(img_save_dir, exist_ok=True)

    save_root = os.path.join(img_save_dir, f"{datetime.now().date()}_{args.dataset}_epoch{epoch_name}_step{step_name}_scale{args.cfg_scale}_step{sample_steps}_size{args.image_size}_bs{args.bs}_samp{args.sampling_algo}_seed{seed}{'_MASKED' if args.enable_masking else ''}")
    os.makedirs(save_root, exist_ok=True)
    visualize(items, args.bs, sample_steps, args.cfg_scale, args.enable_masking)