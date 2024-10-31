# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from os.path import join
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from datetime import datetime
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter
from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


from torch.utils.data import Dataset, DataLoader
class ShapesDatasetCached(Dataset):
    filename = "shapes_dataset_pilot.pth"
    savedir = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/Diffusion_ObjectRelation"
    def __init__(self, transform=None):
        """
        Initializes the dataset.

        Parameters:
        - transform: Optional torchvision transforms to apply to the images.
        """
        self.transform = transform
        self.data = torch.load(join(self.savedir, "dataset", self.filename))
        self.images = self.data["images"]
        self.shape1 = self.data["shape1"]
        self.location1 = self.data["location1"]
        self.shape2 = self.data["shape2"]
        self.location2 = self.data["location2"]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        shape1 = self.shape1[idx]
        location1 = self.location1[idx]
        shape2 = self.shape2[idx]
        location2 = self.location2[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, (shape1, location1, shape2, location2)

def get_max_index(folder_path):
    import re
    indexes = []
    for item in os.listdir(folder_path):
        match = re.match(r"(\d+)-", item)
        if match:
            index = int(match.group(1))
            indexes.append(index)
    if indexes:
        return max(indexes)
    else:
        return 0
#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        # experiment_index = len(glob(f"{args.results_dir}/*"))
        experiment_index = get_max_index(args.results_dir) + 1
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        run_id = datetime.now().strftime("%Y%m%d-%H%M")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{args.dataset}-{'cond' if args.cond else 'uncond'}-{model_string_name}-{args.expname}_{run_id}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        samples_dir = f"{experiment_dir}/samples"  # Stores generated samples
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(samples_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        writer = SummaryWriter(log_dir=join(experiment_dir, "tensorboard_logs"))
    else:
        logger = create_logger(None)
    
    is_conditional = args.cond
    if is_conditional:
        num_classes = args.num_classes
        class_dropout_prob = 0.1
    else: # unconditional
        num_classes = 0
        class_dropout_prob = 1.0
    # Create model:
    # assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size #// 8
    model = DiT_models[args.model](
        input_size=latent_size,
        in_channels=3,
        num_classes=num_classes,
        class_dropout_prob=class_dropout_prob
    )
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    diffusion_eval = create_diffusion(timestep_respacing=args.eval_sampler)  # default: ddim100, linear noise schedule
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Setup data:
    transform = transforms.Compose([
        # transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        # transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    # dataset = ImageFolder(args.data_path, transform=transform)
    dataset = ShapesDatasetCached(transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        writer.add_scalar('epoch', epoch, train_steps)
        for x, y in loader:
            x = x.to(device)
            if is_conditional:
                raise NotImplementedError("Conditional training not supported yet. Need to fix encoding.")
                y = y.to(device)
            else:
                if isinstance(y, torch.Tensor):
                    y = torch.zeros_like(y).to(device)
                else:
                    y = torch.zeros(x.shape[0], dtype=torch.int, device=device)
            # with torch.no_grad():
            #     # Map input images to latent space + normalize latents:
            #     x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                writer.add_scalar('Loss/average', avg_loss, train_steps)
                writer.add_scalar('Loss/batch', loss.detach().item(), train_steps)
                writer.add_scalar('Speed/steps_per_sec', steps_per_sec, train_steps)
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()
            # save samples 
            if train_steps % args.save_samples_every == 0 or \
                (train_steps == 1) :
                # or (train_steps == args.total_steps)
                if rank == 0:
                    model.eval() 
                    if is_conditional:  # conditional case
                        y = torch.randint(0, args.num_classes, (args.num_eval_sample,), device=device)
                    else:               # unconditional case
                        y = args.num_classes * torch.ones((args.num_eval_sample,), dtype=torch.int, device=device)
                    model_kwargs = dict(y=y)
                    # Sample images:
                    z = torch.randn(args.num_eval_sample, model.module.in_channels, latent_size, latent_size, device=device)
                    with torch.no_grad():
                        samples = diffusion_eval.ddim_sample_loop(
                            model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
                        )
                    model.train() 
                    samples = (0.5 * samples + 0.5).clamp(0, 1)
                    # Save and display images:
                    nrow = int(args.num_eval_sample ** 0.5)
                    save_image(samples, f"{samples_dir}/{train_steps:07d}.png", nrow=nrow, ) #normalize=True, value_range=(-1, 1))
                    
                    # sample_fmt = format_samples(samples, args.encoding)
                    # c3_list, c2_list, rule_col = infer_rule_from_sample_batch(sample_fmt)
                    # c3_cnt, c2_cnt, anyvalid_cnt, total = compute_rule_statistics(c3_list, c2_list, rule_col)
                    # c3_vec, c2_vec, rule_vec = pool_rules(c3_list, c2_list, rule_col)
                    # torch.save({'c3_list': c3_list, 'c2_list': c2_list, 'rule_col': rule_col, 
                    #             'c3_cnt': c3_cnt, 'c2_cnt': c2_cnt, 'anyvalid_cnt': anyvalid_cnt, 'total': total},
                    #         f'{samples_dir}/sample_rule_eval_{train_steps}.pt')
                    # # use this dict to log at progress bar. 
                    # eval_dict = {"c3": c3_cnt / total, "c2": c2_cnt / total, "valid": anyvalid_cnt / total / 3}
                    # logger.info(f"(step={train_steps:07d}) Eval: C3: {eval_dict['c3']:.4f}, C2: {eval_dict['c2']:.4f}, AnyValid: {eval_dict['valid']:.4f}")
                
                    # writer.add_scalar('Rules/c3_cnt', c3_cnt, train_steps)
                    # writer.add_scalar('Rules/c2_cnt', c2_cnt, train_steps)
                    # writer.add_scalar('Rules/anyvalid_cnt', anyvalid_cnt, train_steps)
                    
                    # writer.add_scalar('Rules/c3', c3_cnt / total, train_steps)
                    # writer.add_scalar('Rules/c2', c2_cnt / total, train_steps)
                    # writer.add_scalar('Rules/anyvalid', anyvalid_cnt / total / 3, train_steps)
                    # if c3_cnt > 0:
                    #     writer.add_histogram('Rules/c3_vec', c3_vec, train_steps, bins=range(41))
                    # if c2_cnt > 0:
                    #     writer.add_histogram('Rules/c2_vec', c2_vec, train_steps, bins=range(41))
                    # if anyvalid_cnt > 0:
                    #     writer.add_histogram('Rules/rule_vec', rule_vec, train_steps, bins=range(41))

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()
    writer.close()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/Diffusion_ObjectRelation/DiT/results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-S/4")
    parser.add_argument("--image-size", type=int, choices=[64, 128], default=64)
    parser.add_argument("--dataset", type=str, default="shapes_2obj_excl_pilot")
    parser.add_argument("--num-classes", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    # parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--save-samples-every", type=int, default=2500)
    parser.add_argument("--eval_sampler", type=str, default="ddim100")
    parser.add_argument("--num_eval_sample", type=int, default=256)
    parser.add_argument("--cond", action="store_true")
    parser.add_argument("--expname", default="", type=str, help='Experiment name for current run') 
    args = parser.parse_args()
    main(args)
