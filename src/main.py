import torch
import matplotlib.pyplot as plt
import torch.distributed as dist

def generate_batch(
    model,   
    scheduler,
    batch_size: int,  
    device="cuda:0"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a batch of images from the model.
    Returns:
        latents: shape (B, T, C, H, W)
        next_latents: shape (B, T, C, H, W)
        log_probs: shape (B, T)
        timesteps: shape (T)
    """

    # Start from pure noise
    n_channels = model.config.in_channels
    image_size = model.config.sample_size
    latents = torch.randn((batch_size, n_channels, image_size, image_size), device=device)

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
            scheduler_output, log_prob = scheduler.step(pred_noise, t, latents, eta=1.0)
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

if __name__ == "__main__":
    import argparse
    import json
    from diffusers import UNet2DModel
    from custom_ddim_scheduler import CustomDDIMScheduler
    from utils import display_sample
    from accelerate import Accelerator
    from rewards import reward_function

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fine-tune diffusion model with RL')
    parser.add_argument('--per_gpu_batch_size', type=int, default=5, help='Batch size per GPU')
    parser.add_argument('--inference_timesteps', type=int, default=50, help='Number of inference timesteps')
    parser.add_argument('--full_epochs', type=int, default=10, help='Total number of epochs')
    parser.add_argument('--epochs_per_sampling', type=int, default=2, help='Number of training epochs per sampling')
    parser.add_argument('--samples_per_epoch', type=int, default=100, help='Number of samples per epoch')
    parser.add_argument('--first_train_step', type=int, default=0, help='First timestep to train on (inclusive)')
    parser.add_argument('--last_train_step', type=int, default=48, help='Last timestep to train on (inclusive)')
    parser.add_argument('--learning_rate', type=float, default=1e-6, help='Learning rate for optimizer')
    
    args = parser.parse_args()
    
    # Initialize the accelerator
    accelerator = Accelerator(gradient_accumulation_steps=args.last_train_step - args.first_train_step + 1)
    device = accelerator.device
    print(f"Using device: {device}")

    # Define number of samples and batches per GPU
    num_samples_per_gpu = args.samples_per_epoch // accelerator.num_processes
    num_batches_per_gpu = num_samples_per_gpu // args.per_gpu_batch_size

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

    # Initialize list to store global rewards for plotting
    global_rewards_history = []

    for epoch in range(args.full_epochs):
        if accelerator.is_main_process:
            print(f"Epoch {epoch}")

        # Check if models are synced across GPUs
        check_model_sync(accelerator, pretrained_model)

        # Generate a batch of images
        batches = []
        all_rewards = []
        print(f"Generating {num_samples_per_gpu} samples in {num_batches_per_gpu} batches of size {args.per_gpu_batch_size}")
        for _ in range(num_batches_per_gpu):
            latents, next_latents, log_probs, timesteps = generate_batch(pretrained_model.module, scheduler, args.per_gpu_batch_size, device)
            
            # Get the rewards
            rewards, scores = reward_function(next_latents[:, -1])
            print(f"scores={scores}")
            rewards = rewards.to(device)
            all_batch_rewards = accelerator.gather(rewards)    
            all_rewards.append(all_batch_rewards)
            if accelerator.is_main_process: # only print on the main process
                print(f"rewards={all_batch_rewards}")

            batches.append((latents, next_latents, log_probs, timesteps, rewards))
            torch.cuda.empty_cache()

        # Compute reward avg and std
        all_rewards = torch.cat(all_rewards)
        global_mean = all_rewards.mean()
        global_std  = all_rewards.std(unbiased=False)
        print(f"device={device}, global_mean={global_mean}, global_std={global_std}")

        # Store the average reward for this epoch
        if accelerator.is_main_process:
            global_rewards_history.append(global_mean.item())
            print(f"Epoch {epoch} average reward: {global_mean.item()}")

        for inner_epoch in range(args.epochs_per_sampling):
            for b, batch in enumerate(batches):
                if accelerator.is_main_process:
                    print(f"Batch {b+1}/{len(batches)}")

                # Unpack the batch
                latents, next_latents, log_probs, timesteps, rewards = batch 
                
                #Compute the normalized rewards / advantage
                advantages = (rewards - global_mean) / global_std

                # Backpropagate accumulating gradients for each timestep within the training range
                for t in range(args.first_train_step, args.last_train_step + 1):
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
                        if accelerator.sync_gradients:
                            #print(f"({device}) Syncing gradients in timestep {t}")
                            pass
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()

                torch.cuda.synchronize()
                accelerator.wait_for_everyone()

    # Check if models are synced across GPUs
    check_model_sync(accelerator, pretrained_model)

    # Save some samples
    print(f"Saving {num_samples_per_gpu} samples in device {device}")
    latents, next_latents, log_probs, timesteps = generate_batch(pretrained_model.module, scheduler, num_samples_per_gpu, device)
    
    for i in range(num_samples_per_gpu):
        display_sample(next_latents[i:i+1, -1], f"Device {device} Final sample {i}")

    # Plot the global rewards history
    if accelerator.is_main_process and global_rewards_history:
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(global_rewards_history)), global_rewards_history, 'b-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Average Global Reward')
        plt.title('Global Reward Progress During Training')
        plt.grid(True, alpha=0.3)
        plt.savefig('global_rewards_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved reward plot to global_rewards_plot.png")

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
        
        print(f"Saved model to {model_dir}")
        print(f"Saved training arguments to {model_dir}/training_args.json")