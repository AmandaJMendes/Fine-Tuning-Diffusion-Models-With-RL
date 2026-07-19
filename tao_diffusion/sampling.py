import torch
from torch import nn

from tao_diffusion.custom_ddim_scheduler import CustomDDIMScheduler


@torch.no_grad()
def generate_eval_samples(
    model: nn.Module,
    scheduler: CustomDDIMScheduler,
    global_indices: list[int],
    device: torch.device,
    base_seed: int,
) -> torch.Tensor:
    """
    Generate denoised samples x_0 with **portable** RNG for evaluation.

    Each image with global index g in *global_indices* is driven by its own
    CPU generator seeded (base_seed + g).  Because the RNG runs on CPU
    rather than on the CUDA device:

      * the initial noise x_T and every per-step stochastic DDIM variance
        noise depend only on g, not on the GPU architecture (the CUDA RNG
        is architecture-dependent; the CPU Philox RNG is not);
      * the image set is independent of world size — the global index, not
        the rank, identifies the image;
      * the global / training RNG is never touched.

    Unlike ``generate_batch`` this returns only the final ``x_0`` (eval does
    not need the trajectory or log-probs) and uses one CPU generator per
    image so that each image's noise stream is invariant to batch
    composition and world size.

    Returns the final x_0 tensor of shape (len(global_indices), C, H, W).
    """
    n_channels = model.config.in_channels
    image_size = model.config.sample_size
    shape = (n_channels, image_size, image_size)

    gens = [
        torch.Generator(device="cpu").manual_seed(base_seed + g)
        for g in global_indices
    ]

    # x_T drawn on CPU, then moved to device — keeps noise GPU-independent
    latents = torch.stack(
        [torch.randn(shape, generator=g) for g in gens]
    ).to(device)

    for t in scheduler.timesteps:
        # Per-step stochastic DDIM variance noise, also CPU-seeded, fed
        # in explicitly so the scheduler never touches the CUDA RNG.
        var_noise = torch.stack(
            [torch.randn(shape, generator=g) for g in gens]
        ).to(device)
        pred_noise = model(latents, t).sample
        out, _ = scheduler.step(
            pred_noise, t, latents, eta=1.0, variance_noise=var_noise
        )
        latents = out.prev_sample

    return latents


@torch.no_grad()
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
