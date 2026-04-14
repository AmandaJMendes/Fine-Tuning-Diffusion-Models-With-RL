import argparse
import os
import sys

# Use a fallback if __file__ is not defined (e.g., in Jupyter)
try:
    notebook_dir = os.path.dirname(__file__)
except NameError:
    notebook_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(notebook_dir, "..")))
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import UNet2DModel
from tqdm import tqdm

from src.custom_ddim_scheduler import CustomDDIMScheduler
from src.main import generate_batch
from src.rewards import reward_function


def plot_latents(latents, nrows=2, suptitle=None, title=None):
    """
    latents: list of tensors shaped [3,H,W] in [-1,1]
    """
    n = len(latents)
    ncols = int(np.ceil(n / nrows))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))

    # axes could be 2D or 1D depending on nrows/ncols
    axes = np.array(axes).reshape(-1)

    for i, latent in enumerate(latents):
        img = ((latent + 1.0) * 127.5).clamp(0, 255).byte()
        img = img.permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].axis("off")
        if title:
            axes[i].set_title(f"{title} {i}")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    if suptitle:
        fig.suptitle(suptitle)
    plt.tight_layout()
    plt.show()


def denoise_corrupted_batched(
    corrupted_img, model, scheduler, num_samples=10, batch_size=None, t_start=None, device=None
):
    """
    Optimized batched version that processes all samples together, with configurable batch size.

    Args:
        corrupted_img: Single corrupted image tensor
        model: The diffusion model
        scheduler: The DDIM scheduler
        num_samples: Number of samples to generate
        batch_size: Batch size for processing (default: num_samples, i.e., all at once)
        t_start: Timestep to start denoising from
        device: torch.device

    Returns:
        Tensor of denoised latents of shape (num_samples, 3, 256, 256)
    """
    corrupted_img = corrupted_img.to(device)
    if corrupted_img.dim() == 3:
        corrupted_img = corrupted_img.unsqueeze(0)
    if batch_size is None:
        batch_size = num_samples

    all_latents = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
        latents = corrupted_img.repeat(current_batch_size, 1, 1, 1)  # [current_batch_size, C, H, W]

        for t in scheduler.timesteps:  # Timesteps are in descending order
            if t_start is not None and t > t_start:
                continue
            with torch.no_grad():
                pred_noise = model(latents, t).sample
                scheduler_output, log_prob = scheduler.step(pred_noise, t, latents, eta=1.0)
                latents = scheduler_output.prev_sample

        all_latents.append(latents.cpu())

    # Concatenate all batches into a single tensor of shape (num_samples, 3, 256, 256)
    return torch.cat(all_latents, dim=0)


if __name__ == "__main__":
    print("🚀 Starting Perceptual Analysis of Denoising at Different Timesteps...")

    parser = argparse.ArgumentParser(
        description="Perceptual analysis of denoising at different timesteps."
    )
    parser.add_argument(
        "--num_gen_images", type=int, default=2, help="Number of images to generate"
    )
    parser.add_argument(
        "--num_denoised_samples",
        type=int,
        default=5,
        help="Number of denoised samples per image per timestep",
    )
    parser.add_argument(
        "--plot_dir", type=str, default=".", help="Directory to save resulting plots"
    )
    parser.add_argument(
        "--batch_size", type=int, default=10, help="Batch size for denoising samples"
    )
    args = parser.parse_args()

    num_gen_images = args.num_gen_images
    num_denoised_samples = args.num_denoised_samples
    plot_dir = args.plot_dir
    batch_size = args.batch_size

    os.makedirs(plot_dir, exist_ok=True)

    # Load model and scheduler
    print("🧠 Loading model and scheduler...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scheduler = CustomDDIMScheduler.from_pretrained(
        "google/ddpm-celebahq-256", use_safetensors=True
    )
    model = UNet2DModel.from_pretrained("google/ddpm-celebahq-256").to(device)
    scheduler.set_timesteps(50)

    # Generate images
    print(f"🖼️ Generating {num_gen_images} images...")
    _, next_latents, _, timesteps = generate_batch(
        model, scheduler, num_gen_images, device
    )  # next_latents: [num_gen_images, timesteps, C, H, W] ; timesteps: [timesteps]

    # Denoise from intermediate timesteps, for each image
    print("✨ Denoising latents for all images and timesteps...")
    all_last_latents = [
        [] for _ in range(num_gen_images)
    ]  # list of lists: [num_gen_images][timesteps][num_samples, C, H, W]
    for t_idx, timestep in enumerate(tqdm(timesteps, desc="Denoising latents for all images")):
        latents_at_t = next_latents[:, t_idx]  # shape: [num_gen_images, C, H, W]
        for img_idx in range(num_gen_images):
            denoised = denoise_corrupted_batched(
                latents_at_t[img_idx],
                model,
                scheduler,
                num_samples=num_denoised_samples,
                batch_size=batch_size,
                t_start=timestep,
                device=device,
            )
            all_last_latents[img_idx].append(denoised)  # denoised shape: [num_samples, C, H, W]

    # Compute original reward, IR, and gender scores for all generated images in a batch
    print(
        "📊 Computing original reward, IR, and gender scores for all "
        f"{num_gen_images} generated images..."
    )
    original_rewards, original_scores = reward_function(
        next_latents[:, -1]
    )  # next_latents[:, -1] shape: [num_gen_images, C, H, W]
    original_ir_person = original_scores["ir_person"]
    original_sex_score_binary = original_scores["sex_score_binary"]

    # For each timestep, aggregate statistics (mean of stds, mean of means,
    # etc.) across all images for that timestep.
    reward_stds = []
    reward_means = []
    reward_mean_diffs = []
    ir_person_means = []
    ir_person_stds = []
    ir_person_mean_diffs = []
    sex_score_binary_means = []
    sex_score_binary_stds = []
    sex_score_binary_mean_diffs = []
    timesteps_list = []

    print("⏳ Evaluating rewards by timestep...")
    for t_idx, timestep in enumerate(tqdm(timesteps, desc="Evaluating rewards by timestep")):
        per_image_reward_stds = []
        per_image_reward_means = []
        per_image_reward_mean_diffs = []
        per_image_ir_means = []
        per_image_ir_stds = []
        per_image_ir_mean_diffs = []
        per_image_gender_means = []
        per_image_gender_stds = []
        per_image_gender_mean_diffs = []

        for img_idx in range(num_gen_images):
            latents = all_last_latents[img_idx][t_idx]  # [num_samples, C, H, W]
            rewards, scores = reward_function(latents)
            per_image_reward_stds.append(rewards.std().item())
            per_image_reward_means.append(rewards.mean().item())
            per_image_reward_mean_diffs.append(
                abs(rewards.mean().item() - original_rewards[img_idx].item())
            )

            ir_person = scores["ir_person"]
            sex_score_binary = scores["sex_score_binary"]

            per_image_ir_means.append(ir_person.mean().item())
            per_image_ir_stds.append(ir_person.std().item())
            per_image_ir_mean_diffs.append(
                abs(ir_person.mean().item() - original_ir_person[img_idx].item())
            )

            per_image_gender_means.append(sex_score_binary.mean().item())
            per_image_gender_stds.append(sex_score_binary.std().item())
            per_image_gender_mean_diffs.append(
                abs(sex_score_binary.mean().item() - original_sex_score_binary[img_idx].item())
            )

        # For this timestep, aggregate across images (mean of stds, mean of means, etc.)
        reward_stds.append(np.mean(per_image_reward_stds))
        reward_means.append(np.mean(per_image_reward_means))
        reward_mean_diffs.append(np.mean(per_image_reward_mean_diffs))

        ir_person_means.append(np.mean(per_image_ir_means))
        ir_person_stds.append(np.mean(per_image_ir_stds))
        ir_person_mean_diffs.append(np.mean(per_image_ir_mean_diffs))

        sex_score_binary_means.append(np.mean(per_image_gender_means))
        sex_score_binary_stds.append(np.mean(per_image_gender_stds))
        sex_score_binary_mean_diffs.append(np.mean(per_image_gender_mean_diffs))

        timesteps_list.append(int(timestep))

    indices = list(range(len(timesteps_list)))

    print("📈 Plotting results...")

    # Plot for reward std
    plt.figure(figsize=(8, 4))
    plt.plot(indices, reward_stds, marker="o")
    plt.xlabel("Timestep")
    plt.ylabel("Std of Rewards")
    plt.title("Timestep vs Std of Rewards")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "timestep_vs_std_of_rewards.png"))
    print(f"💾 Saved: {os.path.join(plot_dir, 'timestep_vs_std_of_rewards.png')}")
    plt.close()

    # Plot for IR person std
    plt.figure(figsize=(8, 4))
    plt.plot(indices, ir_person_stds, marker="o", color="orange")
    plt.xlabel("Timestep")
    plt.ylabel("Std of IR Score")
    plt.title("Timestep vs Std of IR Score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "timestep_vs_std_of_ir_score.png"))
    print(f"💾 Saved: {os.path.join(plot_dir, 'timestep_vs_std_of_ir_score.png')}")
    plt.close()

    # Plot for Gender (sex_score_binary) std
    plt.figure(figsize=(8, 4))
    plt.plot(indices, sex_score_binary_stds, marker="o", color="green")
    plt.xlabel("Timestep")
    plt.ylabel("Std of Gender Score (Binary)")
    plt.title("Timestep vs Std of Gender Score (Binary)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "timestep_vs_std_of_gender_score_binary.png"))
    print(f"💾 Saved: {os.path.join(plot_dir, 'timestep_vs_std_of_gender_score_binary.png')}")
    plt.close()

    # Plot for reward mean difference
    plt.figure(figsize=(8, 4))
    plt.plot(indices, reward_mean_diffs, marker="o")
    plt.xlabel("Timestep")
    plt.ylabel("Mean Reward Difference (vs. Original)")
    plt.title("Timestep vs Mean Reward Difference from Original Images (Absolute Value)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "timestep_vs_mean_reward_diff_from_original.png"))
    print(f"💾 Saved: {os.path.join(plot_dir, 'timestep_vs_mean_reward_diff_from_original.png')}")
    plt.close()

    # Plot for IR person mean difference
    plt.figure(figsize=(8, 4))
    plt.plot(indices, ir_person_mean_diffs, marker="o", color="orange")
    plt.xlabel("Timestep")
    plt.ylabel("Mean IR Score Difference (vs. Original)")
    plt.title("Timestep vs Mean IR Score Difference from Original Images (Absolute Value)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "timestep_vs_mean_ir_score_diff_from_original.png"))
    print(f"💾 Saved: {os.path.join(plot_dir, 'timestep_vs_mean_ir_score_diff_from_original.png')}")
    plt.close()

    # Plot for Gender (sex_score_binary) mean difference
    plt.figure(figsize=(8, 4))
    plt.plot(indices, sex_score_binary_mean_diffs, marker="o", color="green")
    plt.xlabel("Timestep")
    plt.ylabel("Mean Gender Score Difference (vs. Original, Binary)")
    plt.title("Timestep vs Mean Gender Score Difference from Original Images (Absolute Value)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "timestep_vs_mean_gender_score_diff_from_original.png"))
    saved_gender_diff_plot = os.path.join(
        plot_dir, "timestep_vs_mean_gender_score_diff_from_original.png"
    )
    print(f"💾 Saved: {saved_gender_diff_plot}")
    plt.close()

    print("🎉 All plots saved! Analysis complete.")
