import argparse
import csv
import logging
import os

import numpy as np
import torch
from diffusers import UNet2DModel
from tqdm import tqdm

from tao_diffusion.custom_ddim_scheduler import CustomDDIMScheduler
from tao_diffusion.rewards import DEFAULT_REWARD_PROMPT, reward_function, tensor_batch_to_pil_images
from tao_diffusion.sampling import generate_batch

SCORE_KEYS = ["ir_person", "sex_score", "sex_score_binary", "aesthetics_score"]


def corrupt_to_timestep(
    x0: torch.Tensor,
    n_corruptions: int,
    scheduler: CustomDDIMScheduler,
    timestep: int | torch.Tensor,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
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
    corrupted_imgs: torch.Tensor,
    model: UNet2DModel,
    scheduler: CustomDDIMScheduler,
    batch_size: int | None = None,
    timestep: int | torch.Tensor | None = None,
    device: torch.device | None = None,
    eta: float = 0.0,
) -> torch.Tensor:
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


def write_score_row(
    writer: csv.DictWriter,
    timestep: int | None,
    image_idx: int,
    sample_idx: int | None,
    reward: float | torch.Tensor | np.ndarray,
    scores: dict[str, torch.Tensor | np.ndarray | None],
) -> None:
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
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    logger.info("Computing reward-aware timestep metrics...")

    parser = argparse.ArgumentParser(
        description=(
            "Measures reward sensitivity ΔR(t) and reward variance σ²R(t) across diffusion "
            "timesteps. Generates N images, corrupts each to every timestep t, and reconstructs "
            "M times deterministically to record rewards."
        )
    )
    # Model
    parser.add_argument(
        "--model_id",
        type=str,
        default="google/ddpm-celebahq-256",
        help="HuggingFace model ID for the pretrained diffusion model",
    )

    # Profiling
    parser.add_argument(
        "--num_denoising_steps",
        type=int,
        default=50,
        help="Number of DDIM denoising steps used to generate each trajectory",
    )
    parser.add_argument(
        "--num_images", type=int, default=2, help="Number of images to use for analysis"
    )
    parser.add_argument(
        "--num_reconstructions",
        type=int,
        default=5,
        help="Number of independent reconstructions per image per timestep",
    )
    parser.add_argument(
        "--batch_size", type=int, default=10, help="Batch size for denoising samples"
    )

    # Reward
    parser.add_argument(
        "--reward_prompt",
        type=str,
        default=DEFAULT_REWARD_PROMPT,
        help="Text prompt passed to ImageReward for scoring",
    )
    parser.add_argument(
        "--gender_threshold",
        type=float,
        default=0.8,
        help="Male-probability threshold for the binary gender reward term",
    )
    parser.add_argument(
        "--gender_weight",
        type=float,
        default=2.0,
        help="Weight on the gender term in the combined reward",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/timestep_profiles",
        help="Directory to save CSVs and visualizations",
    )
    parser.add_argument(
        "--save_visualizations",
        action="store_true",
        help="Save all reconstructed images to disk alongside the CSV",
    )
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    plot_dir = os.path.join(args.output_dir, f"worker_{local_rank}")

    # Set device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(23 + local_rank)
    logger.info("[%d] Using device: %s", local_rank, device)

    # Set seed
    torch.manual_seed(23 + local_rank)
    np.random.seed(23 + local_rank)
    logger.info("[%d] Set random seeds to 23 + %d", local_rank, local_rank)

    # Create directories
    os.makedirs(plot_dir, exist_ok=True)
    if args.save_visualizations:
        reconstructions_dir = os.path.join(plot_dir, "reconstructions")
        original_images_dir = os.path.join(plot_dir, "original_images")
        os.makedirs(reconstructions_dir, exist_ok=True)
        os.makedirs(original_images_dir, exist_ok=True)

    csv_keys = ["timestep", "image_idx", "sample_idx", "reward"] + SCORE_KEYS
    csv_path = os.path.join(plot_dir, "reconstruction_scores.csv")
    with open(csv_path, "w", newline="", buffering=1) as csvfile:
        csv.DictWriter(csvfile, fieldnames=csv_keys).writeheader()

    # Load model and scheduler
    logger.info("[%d] Loading model and scheduler...", local_rank)
    scheduler = CustomDDIMScheduler.from_pretrained(args.model_id, use_safetensors=True)
    model = UNet2DModel.from_pretrained(args.model_id).to(device).eval()
    scheduler.set_timesteps(args.num_denoising_steps)

    # Generate images
    logger.info("[%d] Generating %d images...", local_rank, args.num_images)
    _, next_latents, _, timesteps = generate_batch(
        model, scheduler, args.num_images, device
    )  # next_latents: [num_images, T, C, H, W] ; timesteps: [T]
    original_images = next_latents[:, -1]
    del next_latents
    torch.cuda.empty_cache()

    if args.save_visualizations:
        for img_idx, orig_img in enumerate(original_images):
            tensor_batch_to_pil_images(orig_img.unsqueeze(0))[0].save(
                os.path.join(original_images_dir, f"image_{img_idx}.png")
            )

    logger.info("[%d] Computing original scores for generated images...", local_rank)
    original_rewards, original_scores = reward_function(
        original_images,
        prompt=args.reward_prompt,
        male_threshold=args.gender_threshold,
        gender_weight=args.gender_weight,
    )  # original_images: [num_images, C, H, W]

    with open(csv_path, "a", newline="", buffering=1) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_keys)
        for img_idx in range(args.num_images):
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
    logger.info("[%d] Original scores for all generated images saved.", local_rank)

    # For each timestep, corrupt each image num_reconstructions times and reconstruct
    logger.info("[%d] Computing reconstructions across all timesteps...", local_rank)
    with open(csv_path, "a", newline="", buffering=1) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_keys)
        for timestep in tqdm(timesteps, desc=f"Timesteps [{local_rank}]"):
            for img_idx in range(args.num_images):
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

                rewards, scores = reward_function(
                    denoised,
                    prompt=args.reward_prompt,
                    male_threshold=args.gender_threshold,
                    gender_weight=args.gender_weight,
                )

                if args.save_visualizations:
                    for sample_idx in range(args.num_reconstructions):
                        fname = f"image_{img_idx}_t{timestep}_{sample_idx}.png"
                        tensor_batch_to_pil_images(denoised[sample_idx].unsqueeze(0))[0].save(
                            os.path.join(reconstructions_dir, fname)
                        )

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

    logger.info("[%d] Timestep profiling completed.", local_rank)
