import torch
import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    from diffusers import UNet2DModel
    from custom_ddim_scheduler import CustomDDIMScheduler
    from utils import display_sample
    from accelerate import Accelerator
    from rewards import reward_function

    # Define hyperparameters
    PER_GPU_BATCH_SIZE = 5
    INFERENCE_TIMESTEPS = 50
    FULL_EPOCHS = 3
    EPOCHS_PER_SAMPLING = 2
    SAMPLES_PER_EPOCH = 100
    
    # Initialize the accelerator
    accelerator = Accelerator()
    device = accelerator.device
    print(f"Using device: {device}")

    # Define number of samples and batches per GPU
    num_samples_per_gpu = SAMPLES_PER_EPOCH // accelerator.num_processes
    num_batches_per_gpu = num_samples_per_gpu // PER_GPU_BATCH_SIZE

    # Load the model and scheduler
    scheduler = CustomDDIMScheduler.from_pretrained("google/ddpm-celebahq-256", use_safetensors = True)
    pretrained_model = UNet2DModel.from_pretrained("google/ddpm-celebahq-256").to(device)

    # Define the optimizer
    optimizer = torch.optim.AdamW(pretrained_model.parameters(), lr=1e-6)

    # Prepare the model for DDP
    pretrained_model, optimizer = accelerator.prepare(pretrained_model, optimizer)                   

    # Set the timesteps and move the alphas_cumprod to the device
    scheduler.set_timesteps(INFERENCE_TIMESTEPS, device=device)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)

    # Initialize list to store global rewards for plotting
    global_rewards_history = []

    for epoch in range(FULL_EPOCHS):
        if accelerator.is_main_process:
            print(f"Epoch {epoch}")

        # Generate a batch of images
        batches = []
        all_rewards = []
        print(f"Generating {num_samples_per_gpu} samples in {num_batches_per_gpu} batches of size {PER_GPU_BATCH_SIZE}")
        for _ in range(num_batches_per_gpu):
            latents, next_latents, log_probs, timesteps = generate_batch(pretrained_model.module, scheduler, PER_GPU_BATCH_SIZE, device)
            
            # Get the rewards
            rewards, _ = reward_function(next_latents[:, -1])
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

        # Store the average reward for this epoch
        if accelerator.is_main_process:
            global_rewards_history.append(global_mean.item())
            print(f"Epoch {epoch} average reward: {global_mean.item()}")

        for inner_epoch in range(EPOCHS_PER_SAMPLING):
            for b, batch in enumerate(batches):
                if accelerator.is_main_process:
                    print(f"Batch {b+1}/{len(batches)}")

                # Unpack the batch
                latents, next_latents, log_probs, timesteps, rewards = batch 
                
                #Compute the normalized rewards / advantage
                advantages = (rewards - global_mean) / global_std

                # Backpropagate for each timestep
                for t in range(timesteps.shape[0]-1):
                    # Get new likelihoods
                    lat_gpu = latents[:, t].to(device, non_blocking=True)
                    nxt_gpu = next_latents[:, t].to(device, non_blocking=True)
                    t_gpu = timesteps[t].to(device, non_blocking=True)

                    new_log_probs = rescore_batch(
                        pretrained_model.module, 
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
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()

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
