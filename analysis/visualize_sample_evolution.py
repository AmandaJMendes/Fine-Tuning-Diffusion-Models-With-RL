import os
import shutil
from math import ceil, sqrt

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import wandb
from PIL import Image

DEFAULT_PROJECT = "tao-diffusion"
ARTIFACTS_DIR = "artifacts/gifs"


def _resolve_run_path(entity, project, run_name):
    api = wandb.Api()
    for run in api.runs(f"{entity}/{project}"):
        if run.name == run_name:
            return f"{entity}/{project}/{run.id}"
    raise ValueError(f"Run '{run_name}' not found in {entity}/{project}")



def create_grid(pil_imgs, cols=None):
    n = len(pil_imgs)
    cols = cols or ceil(sqrt(n))
    rows = ceil(n / cols)
    w, h = pil_imgs[0].size
    grid = Image.new("RGB", (cols * w, rows * h), "white")
    for i, im in enumerate(pil_imgs):
        grid.paste(im, ((i % cols) * w, (i // cols) * h))
    return grid


def download_frames(run_path, frames_dir, history_key="eval/samples"):
    os.makedirs(frames_dir, exist_ok=True)
    run = wandb.Api().run(run_path)

    print("Fetching eval step list...")
    eval_rows = [
        row for row in run.scan_history(keys=["_step", history_key])
        if (row.get(history_key) or {}).get("filenames")
    ]
    print(f"Found {len(eval_rows)} eval steps.")

    samples_dict = {}
    for row in tqdm.tqdm(eval_rows, desc="downloading", unit="step"):
        step = row["_step"]
        filenames = row[history_key]["filenames"]

        pil_imgs = []
        for idx, rel in enumerate(filenames):
            new_path = os.path.join(frames_dir, f"sample_{idx}_step_{step}.png")
            if not os.path.exists(new_path):
                # wandb.File.download() returns str or file-like depending on SDK version
                local = run.file(rel).download(root=frames_dir, replace=False, exist_ok=True)
                local_path = local if isinstance(local, str) else local.name
                os.rename(local_path, new_path)
            pil_imgs.append(Image.open(new_path).convert("RGB"))

        samples_dict[step] = pil_imgs

    # remove the nested media/ tree W&B creates under frames_dir
    media_dir = os.path.join(frames_dir, "media")
    if os.path.isdir(media_dir):
        shutil.rmtree(media_dir)

    return samples_dict


def save_gif(samples_dict, path, fps=1):
    sorted_steps = sorted(samples_dict)
    frames = [
        np.asarray(create_grid(samples_dict[step]))
        for step in tqdm.tqdm(sorted_steps, desc="building GIF", unit="frame")
    ]
    if not frames:
        print("No frames found – skipping GIF.")
        return
    imageio.mimsave(path, frames, fps=fps)
    print("GIF saved →", path)


def save_evolution_plot(samples_dict, path, n_samples=5, step_interval=100):
    sorted_steps = [s for s in sorted(samples_dict) if s % step_interval == 0]
    if not sorted_steps:
        print(f"No steps at {step_interval}-step intervals – skipping plot.")
        return

    k = min(n_samples, min(len(samples_dict[s]) for s in sorted_steps))
    if k == 0:
        print("No images available – skipping plot.")
        return

    fig, axes = plt.subplots(k, len(sorted_steps), figsize=(2 * len(sorted_steps), 2 * k), squeeze=False)
    fig.suptitle("Sample Evolution During Fine-Tuning", fontsize=20)

    for step_idx, step in enumerate(sorted_steps):
        for img_idx in range(k):
            ax = axes[img_idx][step_idx]
            ax.imshow(samples_dict[step][img_idx])
            ax.axis("off")
            if img_idx == 0:
                ax.set_title(f"Step {step}", fontsize=16)
            if step_idx == 0:
                ax.set_ylabel(f"Image {img_idx + 1}", fontsize=10)

    plt.subplots_adjust(wspace=0.00, hspace=0.01)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Evolution plot saved →", path)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Visualize how generated samples evolve during fine-tuning.")
    ap.add_argument("--run_name", required=True, help="W&B run display name (e.g. morning-wave-19)")
    ap.add_argument("--entity", required=True, help="W&B entity (username or team name)")
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--frames_dir", default=None, help="Frame cache directory (default: artifacts/frames/<run_name>)")
    ap.add_argument("--fps", type=int, default=1)
    ap.add_argument("--plot", action="store_true", help="Also save a static evolution plot")
    ap.add_argument("--n_samples", type=int, default=5, help="Number of sample rows in the evolution plot")
    ap.add_argument("--step_interval", type=int, default=100, help="Show every N-th step in the evolution plot")
    args = ap.parse_args()

    frames_dir = args.frames_dir or os.path.join("artifacts", "frames", args.run_name)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    print(f"Resolving run '{args.run_name}' in {args.entity}/{args.project}...")
    run_path = _resolve_run_path(args.entity, args.project, args.run_name)
    samples = download_frames(run_path, frames_dir)

    if not args.plot and (args.n_samples != 5 or args.step_interval != 100):
        print("Warning: --n_samples and --step_interval have no effect without --plot.")

    save_gif(samples, path=os.path.join(ARTIFACTS_DIR, f"{args.run_name}.gif"), fps=args.fps)

    if args.plot:
        save_evolution_plot(samples, path=os.path.join(ARTIFACTS_DIR, f"{args.run_name}_evolution.png"), n_samples=args.n_samples, step_interval=args.step_interval)
