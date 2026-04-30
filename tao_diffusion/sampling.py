import torch
from torch import nn

from tao_diffusion.custom_ddim_scheduler import CustomDDIMScheduler


def generate_batch(
    model: nn.Module,
    scheduler: CustomDDIMScheduler,
    batch_size: int,
    device: str | torch.device = "cuda:0",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a batch of images from the model.

    Args:
        model:       diffusion network
        scheduler:   DDIM scheduler
        batch_size:  number of samples in the batch
        device:      device on which to place the tensors
        generator:   optional private RNG (keeps global RNG untouched)
    Returns:
        latents:      (B, T, C, H, W)
        next_latents: (B, T, C, H, W)
        log_probs:    (B, T)
        timesteps:    (T,)
    """
    n_channels = model.config.in_channels
    image_size = model.config.sample_size
    latents = torch.randn(
        (batch_size, n_channels, image_size, image_size), device=device, generator=generator
    )

    log_probs_list = []
    latents_list = []
    next_latents_list = []
    timesteps_list = []

    for t in scheduler.timesteps:
        latents_list.append(latents.cpu())

        with torch.no_grad():
            pred_noise = model(latents, t).sample
            scheduler_output, log_prob = scheduler.step(
                pred_noise, t, latents, eta=1.0, generator=generator
            )
            latents = scheduler_output.prev_sample

        log_probs_list.append(log_prob.cpu())
        next_latents_list.append(latents.cpu())
        timesteps_list.append(t.cpu())

    latents = torch.stack(latents_list).permute(1, 0, 2, 3, 4)  # (B, T, C, H, W)
    next_latents = torch.stack(next_latents_list).permute(1, 0, 2, 3, 4)  # (B, T, C, H, W)
    log_probs = torch.stack(log_probs_list).permute(1, 0)  # (B, T)
    timesteps = torch.tensor(timesteps_list)  # (T,)

    return latents, next_latents, log_probs, timesteps
