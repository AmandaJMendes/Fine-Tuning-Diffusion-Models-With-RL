import torch

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

    # Define hyperparameters
    PER_GPU_BATCH_SIZE = 6
    INFERENCE_TIMESTEPS = 50
    
    # Initialize the accelerator
    accelerator = Accelerator()
    device = accelerator.device
    print(f"Using device: {device}")

    # Load the model and scheduler
    scheduler = CustomDDIMScheduler.from_pretrained("google/ddpm-celebahq-256", use_safetensors = True)
    pretrained_model = UNet2DModel.from_pretrained("google/ddpm-celebahq-256").to(device)
    
    # Prepare the model for DDP
    pretrained_model = accelerator.prepare(pretrained_model)                   

    # Set the timesteps and move the alphas_cumprod to the device
    scheduler.set_timesteps(INFERENCE_TIMESTEPS, device=device)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)

    # Generate a batch of images
    latents, next_latents, log_probs, timesteps = generate_batch(pretrained_model.module, scheduler, PER_GPU_BATCH_SIZE, device)
    torch.cuda.empty_cache()
    print(latents.shape, next_latents.shape, log_probs.shape, timesteps.shape)

    # Display some images
    for i in range(3):
        for t in range(40, 50, 2):
            display_sample(latents[i, t:], f"Latent {i} at t={t}")

    # Rescore the batch and backpropagate for each timestep
    for t in range(timesteps.shape[0]):
        new_log_probs = rescore_batch(
            pretrained_model.module, 
            scheduler, 
            latents[:, t].to(device), 
            next_latents[:, t].to(device), 
            timesteps[t].to(device)
        )
        
        loss = new_log_probs.sum()
        print(f"device={device} t={t}  loss={loss.item():.4f}")
        accelerator.backward(loss)
        torch.cuda.empty_cache()
       
        #backpropagate
    # step