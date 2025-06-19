from custom_ddim_scheduler import CustomDDIMScheduler
from rewards import reward_function

from accelerate.logging import get_logger
from accelerate import Accelerator
from diffusers import UNet2DModel
from tqdm import tqdm
from PIL import Image

import torch.distributed as dist
import numpy as np
import argparse
import logging
import torch
import wandb
import json
import math
import os

def generate_batch(
    model,   
    scheduler,
    batch_size: int,  
    device="cuda:0",
    generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a batch of images from the model.

    Args:
        model:       diffusion network
        scheduler:   DDIM scheduler
        batch_size:  number of samples in the batch
        device:      device on which to place the tensors
        generator:   optional private RNG (keeps global RNG untouched)  # NEW
    Returns:
        latents:      (B, T, C, H, W)
        next_latents: (B, T, C, H, W)
        log_probs:    (B, T)
        timesteps:    (T,)
    """

    # Start from pure noise
    n_channels = model.config.in_channels
    image_size = model.config.sample_size
    latents = torch.randn(
        (batch_size, n_channels, image_size, image_size),
        device=device,
        generator=generator
    )

    # Initialize the arrays
    log_probs_list = [] #shape: (T, B)
    latents_list = [] #shape: (T, B, C, H, W)
    next_latents_list = [] #shape: (T, B, C, H, W)
    timesteps_list = [] #shape: (T)

    # Generate trajectory by iterating through the diffusion process
    for t in scheduler.timesteps:
        # Append the latents to the list
        latents_list.append(latents.cpu())

        # Disable gradient calculation since these samples are not part of the training loop
        with torch.no_grad():
            # Get the model prediction
            pred_noise = model(latents, t).sample

            # Step the scheduler to get the next latents
            scheduler_output, log_prob = scheduler.step(pred_noise, t, latents, eta=1.0, generator=generator)
            latents = scheduler_output.prev_sample

        # Append the log_prob and new latents to the lists
        log_probs_list.append(log_prob.cpu())
        next_latents_list.append(latents.cpu())
        timesteps_list.append(t.cpu())

    # Convert the lists to tensors and reshape them
    latents = torch.stack(latents_list).permute(1, 0, 2, 3, 4) #shape: (B, T, C, H, W)
    next_latents = torch.stack(next_latents_list).permute(1, 0, 2, 3, 4) #shape: (B, T, C, H, W)
    log_probs = torch.stack(log_probs_list).permute(1, 0) #shape: (B, T)
    timesteps = torch.tensor(timesteps_list) #shape: (T)

    return latents, next_latents, log_probs, timesteps

def rescore_batch(
    model,
    scheduler,
    latents: torch.Tensor, #shape: (B, C, H, W)
    next_latents: torch.Tensor, #shape: (B, C, H, W)
    timesteps: torch.Tensor #shape: (B,)
) -> torch.Tensor:
    """
    Compute log p(next_latents | latents) with `model` and return (B,).

    Args:
        model: the diffusion model
        scheduler: the scheduler
        latents: shape (B, C, H, W)
        next_latents: shape (B, C, H, W)
        timesteps: shape (B,)
    """
    # Get the model prediction
    pred_noise = model(latents, timesteps).sample

    # Step the scheduler to get the log prob of next_latents given latents  
    _, log_prob = scheduler.step(pred_noise, timesteps, latents, next_latents, eta=1.0)

    return log_prob

def check_model_sync(accelerator, model, tol=1e-6):
    """
    Check if model parameters are synced across GPUs.
    
    Args:
        accelerator: The accelerator object
        model: The model to check
    """
    model = accelerator.unwrap_model(model)
    device = next(model.parameters()).device

    max_diff = torch.tensor(0.0, device=device)
    for p in model.parameters():
        # Copy local data into two buffers
        local = p.data
        global_max = local.clone()
        global_min = local.clone()

        # Compute per‐element max and min across all ranks
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX)
        dist.all_reduce(global_min, op=dist.ReduceOp.MIN)

        # Largest abs difference on this tensor
        diff = (global_max - global_min).abs().max()
        max_diff = torch.max(max_diff, diff)

    if accelerator.is_main_process:
        print(f"Max |Δparam| across all ranks = {max_diff:.3e}")
        if max_diff <= tol:
            print(f"✅ Parameters agree within ±{tol}")
        else:
            print(f"❌ Some params differ by more than ±{tol}")

def evaluate_model(
    step: int,
    model: UNet2DModel,
    scheduler: CustomDDIMScheduler,
    num_samples: int,
    batch_size: int,
    device: torch.device,
    accelerator: Accelerator,
    fixed_seed: int = 1234,
    save_dir: str = "eval_images",
):
    """
    Deterministic evaluation that does not touch the global RNG
    (→ training randomness proceeds as usual).
    """
    os.makedirs(save_dir, exist_ok=True)

    was_training = model.training
    model.eval()

    with torch.no_grad():
        num_batches = math.ceil(num_samples / (batch_size * accelerator.num_processes))
        all_rewards = []
        all_metrics = {k: [] for k in [
            'ir_person', 'sex_score', 'sex_score_binary', 'aesthetics_score'
        ]}

        for i in range(num_batches):
            # private generator = no impact on global RNG
            gen = torch.Generator(device=device).manual_seed(fixed_seed + i + accelerator.process_index)

            latents, next_latents, _, _ = generate_batch(
                model, scheduler, batch_size, device=device, generator=gen
            )

            rewards, scores = reward_function(next_latents[:, -1])
            rewards = rewards.to(device)

            all_rewards.append(accelerator.gather(rewards))
            for k in all_metrics:
                all_metrics[k].append(
                    accelerator.gather(scores[k].to(device)).cpu().flatten()
                )

            # collect the images from all ranks
            gathered = accelerator.gather(next_latents[:, -1].to(device, non_blocking=True))      

            if accelerator.is_main_process:                            # only rank-0 logs
                imgs = gathered.cpu().permute(0, 2, 3, 1)
                imgs = ((imgs + 1.0) * 127.5).numpy().astype(np.uint8)

                wandb_imgs = []
                for idx, arr in enumerate(imgs):
                    img = Image.fromarray(arr)
                    fname = os.path.join(save_dir, f"step_{step:08d}_{idx:05d}.png")
                    img.save(fname)
                    wandb_imgs.append(wandb.Image(img))

                accelerator.log({"eval/samples": wandb_imgs}, step=step)

        # aggregate & log
        all_rewards = torch.cat(all_rewards)
        metrics = {
            "eval/reward":     all_rewards.mean().item(),
            "eval/reward_std": all_rewards.std(unbiased=False).item(),
        }
        for k, v in all_metrics.items():
            vals = torch.cat(v).float()
            metrics[f"eval/{k}"]     = vals.mean().item()
            metrics[f"eval/{k}_std"] = vals.std().item()

        if accelerator.is_main_process:
            accelerator.log(metrics, step=step)

    if was_training:
        model.train()

    torch.cuda.empty_cache()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fine-tune diffusion model with RL')
    parser.add_argument('--per_gpu_batch_size', type=int, default=5, help='Batch size per GPU')
    parser.add_argument('--inference_timesteps', type=int, default=50, help='Number of inference timesteps')
    parser.add_argument('--full_epochs', type=int, default=10, help='Total number of epochs')
    parser.add_argument('--epochs_per_sampling', type=int, default=2, help='Number of training epochs per sampling')
    parser.add_argument('--samples_per_epoch', type=int, default=100, help='Number of samples per epoch')
    parser.add_argument('--first_train_step', type=int, default=0, help='First timestep to train on (inclusive)')
    parser.add_argument('--last_train_step', type=int, default=48, help='Last timestep to train on (inclusive)')
    parser.add_argument('--num_train_timesteps', type=int, default=None, help='Number of timesteps to uniformly sample for training (default: use all timesteps in range)')
    parser.add_argument('--learning_rate', type=float, default=1e-6, help='Learning rate for optimizer')
    parser.add_argument('--eval_every_steps', type=int, default=20, help='Run evaluation every N optimiser steps')
    parser.add_argument('--eval_samples', type=int, default=20, help='Total #samples drawn in each evaluation')
    args = parser.parse_args()
    
    # Create a logger
    logger = get_logger(__name__, log_level="INFO")
    logging.basicConfig(level=logging.INFO) 

    # Initialize the accelerator
    accelerator = Accelerator(gradient_accumulation_steps=args.last_train_step - args.first_train_step + 1, log_with="wandb")
    device = accelerator.device
    logger.info(f"Using device: {device}")

    # Initialize wandb
    accelerator.init_trackers(project_name="diffusion-finetune", config=args)

    # Define number of samples and batches per GPU
    num_batches_per_gpu = math.ceil(args.samples_per_epoch / (args.per_gpu_batch_size * accelerator.num_processes))

    # Load the model and scheduler
    scheduler = CustomDDIMScheduler.from_pretrained("google/ddpm-celebahq-256", use_safetensors = True)
    pretrained_model = UNet2DModel.from_pretrained("google/ddpm-celebahq-256").to(device)

    # Define the optimizer
    optimizer = torch.optim.AdamW(pretrained_model.parameters(), lr=args.learning_rate)

    # Prepare the model for DDP
    pretrained_model, optimizer = accelerator.prepare(pretrained_model, optimizer)                   

    # Set the timesteps and move the alphas_cumprod to the device
    scheduler.set_timesteps(args.inference_timesteps, device=device)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)

    # Initialize the optimiation steps
    global_step = 0

    # Evaluate model before fine-tuning
    logger.info("Evaluating model before fine-tuning")
    evaluate_model(
        step=global_step,
        model=pretrained_model.module,
        scheduler=scheduler,
        num_samples=args.eval_samples,
        batch_size=args.per_gpu_batch_size,
        device=device,
        accelerator=accelerator
    )

    # Sampling + Optimization loop
    for epoch in tqdm(range(args.full_epochs), desc="Training Epochs", disable=not accelerator.is_main_process):
        # Initialize lists to store rewards and batches
        batches = []
        all_rewards = []
        all_metrics = {
            'ir_person': [],
            'sex_score': [],
            'sex_score_binary': [],
            'aesthetics_score': [],
        }

        # Sampling loop
        for _ in tqdm(range(num_batches_per_gpu), desc=f"Generating batches of size {args.per_gpu_batch_size}", disable=not accelerator.is_main_process):
            latents, next_latents, log_probs, timesteps = generate_batch(pretrained_model.module, scheduler, args.per_gpu_batch_size, device)
            
            # Get the rewards in the current GPU
            rewards, scores = reward_function(next_latents[:, -1])
            rewards = rewards.to(device)

            # Gather the rewards and metrics from all GPUs
            all_batch_rewards = accelerator.gather(rewards)
            all_rewards.append(all_batch_rewards)

            # Gather and store metrics from all GPUs
            for k in all_metrics:
                metric_tensor = scores[k].to(device)
                gathered_metric = accelerator.gather(metric_tensor)
                all_metrics[k].append(gathered_metric.cpu().flatten())

            # Append the batch to the list
            batches.append((latents, next_latents, log_probs, timesteps, rewards))
            torch.cuda.empty_cache()

        # Compute the reward avg and std in this sampling epoch
        all_rewards = torch.cat(all_rewards)
        global_mean = all_rewards.mean()
        global_std  = all_rewards.std(unbiased=False)

        # Log the metrics for this epoch
        if accelerator.is_main_process:
            logger.info(f"Average reward: {global_mean.item()}")

            # Concatenate and compute mean for each metric
            metrics_to_log = {'train/reward': global_mean.item(), 'train/reward_std': global_std.item()}
            for k in all_metrics:
                if all_metrics[k]:  # list of tensors
                    all_values = torch.cat(all_metrics[k])
                    metrics_to_log[f"train/{k}"] = all_values.float().mean().item()
                    metrics_to_log[f"train/{k}_std"] = all_values.float().std().item()
                    logger.info(f"Average {k}: {metrics_to_log[f'train/{k}']}")
            accelerator.log(metrics_to_log)

        # Training loop
        for inner_epoch in range(args.epochs_per_sampling):
            
            # Sample timesteps for this epoch
            if args.num_train_timesteps is None:
                # Use all timesteps in the range
                train_timesteps = list(range(args.first_train_step, args.last_train_step + 1))
            else:
                # Uniformly sample K timesteps from the range
                all_timesteps = list(range(args.first_train_step, args.last_train_step + 1))
                train_timesteps = torch.randperm(len(all_timesteps))[:args.num_train_timesteps].tolist()
                train_timesteps = [all_timesteps[i] for i in train_timesteps]

            for b, batch in enumerate(batches):
                logger.info(f"Training step {inner_epoch * len(batches) + b + 1}/{args.epochs_per_sampling * len(batches)} (Inner epoch {inner_epoch+1}/{args.epochs_per_sampling}, Batch {b+1}/{len(batches)})")

                # Unpack the batch
                latents, next_latents, log_probs, timesteps, rewards = batch 
                
                #Compute the normalized rewards, i.e. advantage
                advantages = (rewards - global_mean) / global_std

                # Accumulate gradients for each selected timestep of the current batch
                for t in train_timesteps:
                    with accelerator.accumulate(pretrained_model):
                        # Get new likelihoods
                        lat_gpu = latents[:, t].to(device, non_blocking=True)
                        nxt_gpu = next_latents[:, t].to(device, non_blocking=True)
                        t_gpu = timesteps[t].to(device, non_blocking=True)
                        new_log_probs = rescore_batch(
                            pretrained_model, 
                            scheduler, 
                            lat_gpu, 
                            nxt_gpu, 
                            t_gpu
                        )

                        # Importance Sampling Ratio
                        importance_ratio = torch.exp(new_log_probs - log_probs[:, t].to(device))

                        # PPO clipping
                        clipped_ratio = torch.clamp(importance_ratio, 1 - 1e-4, 1 + 1e-4)
                        loss_clip = torch.min(importance_ratio * advantages, clipped_ratio * advantages)

                        # Compute the total loss
                        loss = -loss_clip.mean()

                        # Backpropagate and clear the cache
                        accelerator.backward(loss)
                        del lat_gpu, nxt_gpu, t_gpu, loss, new_log_probs, importance_ratio, clipped_ratio
                        torch.cuda.empty_cache()
                    
                        # Step the optimizer after the loss was backpropagated for all the timesteps in all GPUs 
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()

                        # Update the global step
                        if accelerator.sync_gradients:
                            global_step += 1
                            if global_step % args.eval_every_steps == 0:
                                logger.info(f"Evaluating model at step {global_step}")
                                evaluate_model(
                                    step=global_step,
                                    model=pretrained_model.module,
                                    scheduler=scheduler,
                                    num_samples=args.eval_samples,
                                    batch_size=args.per_gpu_batch_size,
                                    device=device,
                                    accelerator=accelerator
                                )

                # Synchronize the processes
                torch.cuda.synchronize()
                accelerator.wait_for_everyone()

    # Check if models are synced across GPUs
    check_model_sync(accelerator, pretrained_model)

    # Save the model and arguments
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if accelerator.is_main_process:
        model_to_save = accelerator.unwrap_model(pretrained_model)
        model_dir = f"./final_model_{timestamp}"
        model_to_save.save_pretrained(model_dir)
        
        # Save the arguments as JSON
        args_dict = vars(args)
        with open(f"{model_dir}/training_args.json", "w") as f:
            json.dump(args_dict, f, indent=2)
        
        logger.info(f"Saved model to {model_dir}")
        logger.info(f"Saved training arguments to {model_dir}/training_args.json")