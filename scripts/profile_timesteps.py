import argparse
import csv
import os

import numpy as np
import torch
from diffusers import UNet2DModel
from tqdm import tqdm

from src.custom_ddim_scheduler import CustomDDIMScheduler
from src.sampling import generate_batch
from src.rewards import reward_function, tensor_batch_to_pil_images

SCORE_KEYS = ["ir_person", "sex_score", "sex_score_binary", "aesthetics_score"]


def corrupt_to_timestep(x0, n_corruptions, scheduler, timestep, device, generator=None):
    """
    Forward diffusion: produces n_corruptions independent noisy samples
    x_t^(i,j) ~ q(x_t | x_0^(i)) at the given timestep.
    x0: [C,H,W] or [B,C,H,W], assumed in [-1,1]
    n_corruptions: int, number of independent corruptions to produce
    timestep: int or torch.LongTensor
    device: torch.device
    generator: torch.Generator, optional
    Returns: [n_corruptions, C, H, W]
    """
    if x0.dim() == 3:
        x0 = x0.unsqueeze(0)
    x0 = x0.repeat(n_corruptions, 1, 1, 1).to(device)

    t_scalar = int(timestep.item() if isinstance(timestep, torch.Tensor) else timestep)
    t = torch.full((x0.shape[0],), t_scalar, device=device, dtype=torch.long)

    noise = torch.randn(x0.shape, device=device, generator=generator)
    return scheduler.add_noise(x0, noise, t)



@torch.inference_mode()
def reconstruct_from_timestep(
    corrupted_imgs, model, scheduler, batch_size=None, timestep=None, device=None, eta=0.0
):
    """
    Deterministic reconstruction: denoises x_t^(i,j) back to x̂_0^(i,j)
    ~ p_θ(x̂_0 | x_t^(i,j)), starting from the given timestep.

    Args:
        corrupted_imgs: noisy images x_t^(i,j), shape (n_corruptions, C, H, W)
        model: The diffusion model
        scheduler: The DDIM scheduler
        batch_size: Batch size for processing (default: n_corruptions, i.e., all at once)
        timestep: Corruption timestep to denoise from
        device: torch.device
        eta: stochasticity parameter for scheduler.step; 0 is deterministic (default)

    Returns:
        Reconstructed images x̂_0^(i,j), shape (n_corruptions, C, H, W)
    """
    corrupted_imgs = corrupted_imgs.to(device)
    if corrupted_imgs.dim() == 3:
        corrupted_imgs = corrupted_imgs.unsqueeze(0)
    num_samples = corrupted_imgs.shape[0]
    if batch_size is None:
        batch_size = num_samples

    all_latents = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        latents = corrupted_imgs[start_idx:end_idx].clone()  # Shape: [current_batch_size, C, H, W]

        for t in scheduler.timesteps:
            if timestep is not None and t > timestep:
                continue
            pred_noise = model(latents, t).sample
            scheduler_output, _ = scheduler.step(pred_noise, t, latents, eta=eta)
            latents = scheduler_output.prev_sample

        all_latents.append(latents.cpu())

    return torch.cat(all_latents, dim=0)


def write_score_row(writer, timestep, image_idx, sample_idx, reward, scores):
    """
    Helper to format and write a single row to CSV.
    """
    row = {
        "timestep": timestep,
        "image_idx": int(image_idx),
        "sample_idx": int(sample_idx) if sample_idx is not None else None,
    }
    if isinstance(reward, (torch.Tensor, np.ndarray)):
        row["reward"] = float(reward)
    else:
        row["reward"] = reward

    for k in SCORE_KEYS:
        v = scores.get(k)
        if v is None:
            row[k] = None
        elif isinstance(v, (torch.Tensor, np.ndarray)):
            row[k] = float(v)
        else:
            row[k] = v

    writer.writerow(row)


if __name__ == "__main__":
    print("Computing reward-aware timestep metrics...")

    parser = argparse.ArgumentParser(
        description=(
            "Measures reward sensitivity ΔR(t) and reward variance σ²R(t) across diffusion "
            "timesteps. Generates N images, corrupts each to every timestep t, and reconstructs "
            "M times deterministically to record rewards."
        )
    )
    parser.add_argument(
        "--num_gen_images", type=int, default=2, help="Number of images to use for analysis"
    )
    parser.add_argument(
        "--num_reconstructions",
        type=int,
        default=5,
        help="Number of independent reconstructions per image per timestep",
    )
    parser.add_argument(
        "--output_dir", type=str, default="artifacts/timestep_profiles", help="Directory to save CSVs and visualizations"
    )
    parser.add_argument(
        "--batch_size", type=int, default=10, help="Batch size for denoising samples"
    )
    parser.add_argument("--plot_every_k", type=int, default=10, help="Plot every k timesteps")
    parser.add_argument(
        "--num_plot_samples",
        type=int,
        default=3,
        help="Number of denoised samples to plot at each plotted timestep (m)",
    )
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    plot_dir = os.path.join(args.output_dir, f"worker_{local_rank}")

    # Set device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print(f"[{local_rank}] Using device: {device}")

    # Set seed
    torch.manual_seed(23 + local_rank)
    np.random.seed(23 + local_rank)
    torch.cuda.manual_seed_all(23 + local_rank)
    print(f"[{local_rank}] Set random seeds to 23 + {local_rank}")

    # Create directories
    os.makedirs(plot_dir, exist_ok=True)
    recon_plots_dir = os.path.join(plot_dir, "recon_plots")
    original_images_dir = os.path.join(plot_dir, "original_images")
    os.makedirs(recon_plots_dir, exist_ok=True)
    os.makedirs(original_images_dir, exist_ok=True)

    csv_keys = ["timestep", "image_idx", "sample_idx", "reward"] + SCORE_KEYS
    csv_path = os.path.join(plot_dir, "scores_per_image_timestep.csv")
    # Write header
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_keys)
        writer.writeheader()

    # Load model and scheduler
    print(f"[{local_rank}] Loading model and scheduler...")
    scheduler = CustomDDIMScheduler.from_pretrained(
        "google/ddpm-celebahq-256", use_safetensors=True
    )
    model = UNet2DModel.from_pretrained("google/ddpm-celebahq-256").to(device)
    scheduler.set_timesteps(50)

    # Generate images
    print(f"[{local_rank}] Generating {args.num_gen_images} images...")
    _, next_latents, _, timesteps = generate_batch(
        model, scheduler, args.num_gen_images, device
    )  # next_latents: [num_gen_images, T, C, H, W] ; timesteps: [T]
    original_images = next_latents[:, -1]

    print(f"[{local_rank}] Saving all generated images as files...")
    for img_idx, orig_img in enumerate(original_images):
        save_img_path = os.path.join(original_images_dir, f"gen_image_{img_idx}.png")
        tensor_batch_to_pil_images(orig_img.unsqueeze(0))[0].save(save_img_path)
    print(f"[{local_rank}] Saved all generated images.")

    # Compute and save original reward/IR/gender/aesthetics/sex_score scores
    # for all chosen/generated images.
    print(f"[{local_rank}] Saving original scores for generated images...")
    original_rewards, original_scores = reward_function(
        original_images
    )  # original_images: [num_gen_images, C, H, W]

    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_keys)
        for img_idx in range(args.num_gen_images):
            per_sample_scores = {}
            for k in SCORE_KEYS:
                arr = original_scores.get(k)
                per_sample_scores[k] = arr[img_idx] if arr is not None else None
            write_score_row(
                writer,
                timestep=None,
                image_idx=img_idx,
                sample_idx=None,
                reward=original_rewards[img_idx],
                scores=per_sample_scores,
            )
    print(f"[{local_rank}] Original scores for all generated images saved.")

    # For each timestep, corrupt each image num_reconstructions times and reconstruct deterministically
    print(f"[{local_rank}] Computing reconstructions across all timesteps...")
    with open(csv_path, "a", newline="", buffering=1) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_keys)
        for t_idx, timestep in enumerate(
            tqdm(timesteps, desc=f"Timesteps [{local_rank}]")
        ):
            plot_this_timestep = (
                t_idx % args.plot_every_k == 0 or t_idx == len(timesteps) - 1
            )  # also plot at the final timestep

            for img_idx in range(args.num_gen_images):
                corrupted = corrupt_to_timestep(
                    original_images[img_idx], args.num_reconstructions, scheduler, timestep, device
                )  # x_t^(i,j): [num_reconstructions, C, H, W]

                denoised = reconstruct_from_timestep(
                    corrupted,
                    model,
                    scheduler,
                    batch_size=args.batch_size,
                    timestep=timestep,
                    device=device,
                    eta=0.0,
                )  # x̂_0^(i,j): [num_reconstructions, C, H, W]

                # Compute reward for batch of denoised samples
                rewards, scores = reward_function(denoised)

                if plot_this_timestep:
                    vis_samples = min(args.num_plot_samples, denoised.shape[0])
                    for sample_idx in range(vis_samples):
                        save_path = os.path.join(
                            recon_plots_dir, f"img{img_idx}_t{timestep}_sample{sample_idx}.png"
                        )
                        tensor_batch_to_pil_images(denoised[sample_idx].unsqueeze(0))[0].save(save_path)

                for sample_idx in range(args.num_reconstructions):
                    per_sample_scores = {}
                    for k in SCORE_KEYS:
                        arr = scores.get(k)
                        per_sample_scores[k] = arr[sample_idx] if arr is not None else None
                    write_score_row(
                        writer,
                        timestep=int(timestep),
                        image_idx=img_idx,
                        sample_idx=sample_idx,
                        reward=rewards[sample_idx],
                        scores=per_sample_scores,
                    )

            torch.cuda.empty_cache()

    print(f"[{local_rank}] Timestep profiling completed.")
