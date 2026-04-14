import os

import numpy as np
import pandas as pd
from PIL import Image

# Configuration
num_workers = 2
num_images = 3  # number of generated images per worker
num_timesteps = 5  # number of timesteps
num_samples = 4  # number of denoised samples per image per timestep
score_keys = ["ir_person", "sex_score", "sex_score_binary", "aesthetics_score"]

base_dir = "./dummy_workers"
os.makedirs(base_dir, exist_ok=True)

for worker_id in range(num_workers):
    worker_dir = os.path.join(base_dir, f"worker_{worker_id}")
    os.makedirs(worker_dir, exist_ok=True)
    csv_path = os.path.join(worker_dir, "scores_per_image_timestep.csv")

    rows = []

    # Original images (sample_idx=None, timestep=None)
    for img_idx in range(num_images):
        row = {
            "timestep": None,
            "image_idx": int(img_idx),
            "sample_idx": None,
            "reward": float(np.random.uniform(0, 1)),
        }
        for k in score_keys:
            row[k] = float(np.random.uniform(0, 1))
        rows.append(row)

    # Denoised samples (timestep, image_idx, sample_idx are integers)
    for t in range(num_timesteps):
        for img_idx in range(num_images):
            for sample_idx in range(num_samples):
                row = {
                    "timestep": int(t),
                    "image_idx": int(img_idx),
                    "sample_idx": int(sample_idx),
                    "reward": float(np.random.uniform(0, 1)),
                }
                for k in score_keys:
                    row[k] = float(np.random.uniform(0, 1))
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Created dummy CSV: {csv_path}")


def create_dummy_recon_images(
    plot_dir="dummy_workers",
    num_workers=2,
    num_images=2,
    num_timesteps=10,
    num_samples=5,
    image_size=(64, 64),
):
    """
    Creates dummy reconstruction images for testing GIF creation.

    Args:
        plot_dir (str): Base directory to store worker folders
        num_workers (int): Number of workers
        num_images (int): Number of images per worker
        num_timesteps (int): Number of timesteps
        num_samples (int): Number of denoised samples per image per timestep
        image_size (tuple): (H, W) of dummy images
    """
    os.makedirs(plot_dir, exist_ok=True)

    for w in range(num_workers):
        recon_dir = os.path.join(plot_dir, f"worker_{w}", "recon_plots")
        os.makedirs(recon_dir, exist_ok=True)

        for img_idx in range(num_images):
            for t in range(num_timesteps):
                for sample_idx in range(num_samples):
                    # Generate random RGB image in [0, 255]
                    dummy_img = np.random.randint(
                        0, 256, (image_size[0], image_size[1], 3), dtype=np.uint8
                    )
                    img = Image.fromarray(dummy_img)

                    save_path = os.path.join(recon_dir, f"img{img_idx}_t{t}_sample{sample_idx}.png")
                    img.save(save_path)

    print(f"Dummy images created in {plot_dir}")


# Example usage
create_dummy_recon_images()
