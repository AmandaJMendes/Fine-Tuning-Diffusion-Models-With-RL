import sys
import os
import argparse
import csv

# Use a fallback if __file__ is not defined (e.g., in Jupyter)
try:
    notebook_dir = os.path.dirname(__file__)
except NameError:
    notebook_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(notebook_dir, "..")))
from src.main import generate_batch 
from src.custom_ddim_scheduler import CustomDDIMScheduler
from src.rewards import reward_function

from diffusers import UNet2DModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

def add_noise(x0, n_samples, scheduler, timestep, device, generator=None):
    """
    Applies the forward diffusion process using scheduler.add_noise.
    x0: [C,H,W] or [B,C,H,W], assumed in [-1,1]
    n_samples: int, number of samples to generate
    timestep: int or torch.LongTensor, ascending (forward)
    device: torch.device
    generator: torch.Generator, optional
    Returns: [B, C, H, W]
    """
    if x0.dim() == 3:
        x0 = x0.unsqueeze(0)  # [1, C, H, W]
    if n_samples is not None:
        x0 = x0.repeat(n_samples, 1, 1, 1)  # [n_samples, C, H, W]

    x0 = x0.to(device)

    # Convert timestep to a tensor of shape [B] on device
    if isinstance(timestep, torch.Tensor):
        t_scalar = int(timestep.item())
    else:
        t_scalar = int(timestep)
    t = torch.full((x0.shape[0],), t_scalar, device=device, dtype=torch.long)

    # Forward noise
    noise = torch.randn(x0.shape, device=device, generator=generator)

    # Use scheduler.add_noise if available
    return scheduler.add_noise(x0, noise, t)

def save_tensor_as_image(tensor, path):
    """
    Save a single image tensor ([3,H,W] in [-1,1]) as a PNG image.
    """
    img = ((tensor + 1.0) * 127.5).clamp(0, 255).byte()
    img = img.permute(1, 2, 0).cpu().numpy()
    img = Image.fromarray(img)
    img.save(path)

@torch.inference_mode()
def denoise_corrupted_batched(corrupted_imgs, model, scheduler, batch_size=None, t_start=None, device=None, eta=0.0):
    """
    Optimized batched version that processes all samples together, with configurable batch size.

    Args:
        corrupted_imgs: Tensor of corrupted images (num_samples, 3, 256, 256)
        model: The diffusion model
        scheduler: The DDIM scheduler
        batch_size: Batch size for processing (default: num_samples, i.e., all at once)
        t_start: Timestep to start denoising from
        device: torch.device
        eta: float, eta parameter for scheduler.step. 0 is deterministic by default.

    Returns:
        Tensor of denoised latents of shape (num_samples, 3, 256, 256)
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

        for t in scheduler.timesteps:  # Timesteps are in descending order
            if t_start is not None and t > t_start:
                continue
            with torch.no_grad():
                pred_noise = model(latents, t).sample
                scheduler_output, log_prob = scheduler.step(pred_noise, t, latents, eta=eta)
                latents = scheduler_output.prev_sample

        all_latents.append(latents.cpu())

    # Concatenate all batches into a single tensor of shape (num_samples, 3, 256, 256)
    return torch.cat(all_latents, dim=0)

# Make sure to always write these four scores (or None if they are missing)
SCORE_KEYS = ["ir_person", "sex_score", "sex_score_binary", "aesthetics_score"]

def write_score_row(writer, timestep, image_idx, sample_idx, reward, scores):
    """
    Helper to format and write a single row to CSV.
    """
    row = {
        "timestep": timestep,
        "image_idx": int(image_idx),
        "sample_idx": int(sample_idx) if sample_idx is not None else None,
    }
    val_reward = reward
    if isinstance(val_reward, torch.Tensor) or isinstance(val_reward, np.ndarray):
        row["reward"] = float(val_reward)
    else:
        row["reward"] = val_reward

    for k in SCORE_KEYS:
        v = scores.get(k)
        if v is None:
            row[k] = None
        elif isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
            row[k] = float(v)
        else:
            row[k] = v

    writer.writerow(row)


if __name__ == "__main__":
    print("🚀 Starting Perceptual Analysis of Denoising at Different Timesteps...")

    parser = argparse.ArgumentParser(description="Perceptual analysis of denoising at different timesteps.")
    parser.add_argument("--num_gen_images", type=int, default=2, help="Number of images to keep for analysis (n)")
    parser.add_argument("--num_denoised_samples", type=int, default=5, help="Number of denoised samples per image per timestep")
    parser.add_argument("--plot_dir", type=str, default=".", help="Directory to save resulting plots")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for denoising samples")
    parser.add_argument("--plot_every_k", type=int, default=10, help="Plot every k timesteps")
    parser.add_argument("--num_plot_samples", type=int, default=3, help="Number of denoised samples to plot at each plotted timestep (m)")
    parser.add_argument("--eta", type=float, default=0.0, help="Eta value (stochasticity parameter) to pass to scheduler.step; 0 is deterministic (default)")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    num_gen_images = args.num_gen_images 
    num_denoised_samples = args.num_denoised_samples
    plot_dir = os.path.join(args.plot_dir, f"worker_{local_rank}")
    batch_size = args.batch_size
    plot_every_k = args.plot_every_k
    num_plot_samples = args.num_plot_samples
    eta = args.eta

    # Set device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print(f"🖥️ [{local_rank}] Using device: {device}")

    # Set seed
    torch.manual_seed(23 + local_rank)
    np.random.seed(23 + local_rank)
    torch.cuda.manual_seed_all(23 + local_rank)
    print(f"🌱 [{local_rank}] Set random seeds to 23 + {local_rank}")

    # Create directories
    os.makedirs(plot_dir, exist_ok=True)
    recon_plots_dir = os.path.join(plot_dir, "recon_plots")
    original_images_dir = os.path.join(plot_dir, "original_images")
    os.makedirs(recon_plots_dir, exist_ok=True)
    os.makedirs(original_images_dir, exist_ok=True)

    csv_keys = ["timestep", "image_idx", "sample_idx", "reward"] + SCORE_KEYS
    csv_path = os.path.join(plot_dir, "scores_per_image_timestep.csv")
    # Write header
    with open(csv_path, "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_keys)
        writer.writeheader()

    # Load model and scheduler
    print(f"🧠 [{local_rank}] Loading model and scheduler...")
    scheduler = CustomDDIMScheduler.from_pretrained("google/ddpm-celebahq-256", use_safetensors = True)
    model = UNet2DModel.from_pretrained("google/ddpm-celebahq-256").to(device)
    scheduler.set_timesteps(50)

    # Generate images
    print(f"🖼️ [{local_rank}] Generating {num_gen_images} images...")
    _, next_latents, _, timesteps = generate_batch(model, scheduler, num_gen_images, device) # next_latents: [num_gen_images, timesteps, C, H, W] ; timesteps: [timesteps]
    original_images = next_latents[:, -1]

    # SAVE ALL GENERATED IMAGES
    print(f"💾 [{local_rank}] Saving all generated images as files...")
    for img_idx, orig_img in enumerate(original_images):
        save_img_path = os.path.join(original_images_dir, f"gen_image_{img_idx}.png")
        save_tensor_as_image(orig_img.cpu(), save_img_path)
    print(f"✅ [{local_rank}] Saved all generated images.")

    # Compute and save original reward/IR/gender/aesthetics/sex_score scores for all chosen/generated images
    print(f"💾 [{local_rank}] Saving original scores for generated images...")
    original_rewards, original_scores = reward_function(original_images)  # original_images: [num_gen_images, C, H, W]

    with open(csv_path, "a", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_keys)
        for img_idx in range(num_gen_images):
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
                scores=per_sample_scores
            )
    print(f"💾 [{local_rank}] Original scores for all generated images saved.")

    # Denoise from intermediate timesteps, for each image
    print(f"✨ [{local_rank}] Denoising latents for all generated images and timesteps...")
    # Open the CSV file once for writing denoised sample scores
    with open(csv_path, "a", newline='', buffering=1) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_keys)
        for t_idx, timestep in enumerate(tqdm(timesteps, desc=f"Denoising latents for all images [{local_rank}]")):
            plot_this_timestep = (t_idx % plot_every_k == 0 or t_idx == len(timesteps) - 1)  # Also plot at final timestep

            for img_idx in range(num_gen_images):
                destroyed_img_to_t = add_noise(original_images[img_idx], num_denoised_samples, scheduler, timestep, device) # shape: [num_samples, C, H, W]
                
                denoised = denoise_corrupted_batched(
                    destroyed_img_to_t, model, scheduler, batch_size=batch_size, t_start=timestep, device=device, eta=eta
                )  # denoised: [num_samples, C, H, W]


                # Compute reward for batch of denoised samples
                rewards, scores = reward_function(denoised)

                if plot_this_timestep:
                    # Save up to num_plot_samples reconstructed denoised images for this image and timestep,
                    # each sample as its own file, identified by img_idx, timestep, and sample_idx in the filename
                    vis_samples = min(num_plot_samples, denoised.shape[0])
                    for sample_idx in range(vis_samples):
                        save_path = os.path.join(
                            recon_plots_dir,
                            f"img{img_idx}_t{timestep}_sample{sample_idx}.png"
                        )
                        save_tensor_as_image(denoised[sample_idx].cpu(), save_path)

                # Save per-sample values for every denoised sample (img_idx, t_idx)
                den_n = denoised.shape[0]
                for sample_idx in range(den_n):
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
                        scores=per_sample_scores
                    )

            # 🧹 Free cached GPU memory before next timestep
            torch.cuda.empty_cache()
    
    print(f"✅ [{local_rank}] Perceptual analysis completed.")