import os, numpy as np, imageio.v2 as imageio, wandb, tqdm
from PIL import Image
from math import ceil, sqrt
import matplotlib.pyplot as plt

def create_grid(pil_imgs, cols=None):
    n = len(pil_imgs)
    cols = cols or ceil(sqrt(n))
    rows = ceil(n / cols)
    w, h = pil_imgs[0].size
    grid = Image.new("RGB", (cols * w, rows * h), "white")
    for i, im in enumerate(pil_imgs):
        grid.paste(im, ((i % cols) * w, (i // cols) * h))
    return grid

def make_gif(run_path, out_dir="frames", gif_name="samples.gif", fps=4,
             history_key="eval/samples"):
    os.makedirs(out_dir, exist_ok=True)
    run   = wandb.Api().run(run_path)

    # step -> ordered local PNG paths (slot order preserved)
    frames = []

    for row in tqdm.tqdm(
            run.scan_history(keys=["_step", history_key]), desc="steps"):
        step        = row["_step"]
        meta        = row[history_key] or {}             # always a dict
        filenames   = meta.get("filenames", [])          # slot-order list
        if not filenames:
            continue

        local_paths = []
        for rel in filenames:                            # already ordered!
            local = run.file(rel).download(
                        root=out_dir,
                        replace=False,
                        exist_ok=True)                   # cached after 1st run
            local_paths.append(local if isinstance(local, str) else local.name)

        pil_imgs  = [Image.open(p).convert("RGB") for p in local_paths]
        grid      = create_grid(pil_imgs)                # one grid per step
        frames.append(np.asarray(grid))

    if not frames:
        print("Nothing found – abort.")
        return

    gif_path = os.path.join(out_dir, gif_name)
    imageio.mimsave(gif_path, frames, fps=fps)
    print("GIF saved →", gif_path)

def sample_and_plot_grid(run_path, k=5, out_dir="frames", plot_name="evolution_grid.png",
                        history_key="eval/samples"):
    """
    Sample k images and plot them in a grid where each column is a step 
    and each row is one of the k images.
    """
    os.makedirs(out_dir, exist_ok=True)
    run = wandb.Api().run(run_path)

    # Collect all steps and their images
    all_steps = []
    all_images_per_step = []

    for row in tqdm.tqdm(
            run.scan_history(keys=["_step", history_key]), desc="collecting steps"):
        step = row["_step"]
        meta = row[history_key] or {}
        filenames = meta.get("filenames", [])
        if not filenames:
            continue

        local_paths = []
        for rel in filenames:
            local = run.file(rel).download(
                        root=out_dir,
                        replace=False,
                        exist_ok=True)
            local_paths.append(local if isinstance(local, str) else local.name)

        pil_imgs = [Image.open(p).convert("RGB") for p in local_paths]
        all_steps.append(step)
        all_images_per_step.append(pil_imgs)

    if not all_steps:
        print("No steps found – abort.")
        return

    # Sample k images consistently across all steps
    num_images_available = min(len(imgs) for imgs in all_images_per_step)
    k = min(k, num_images_available)
    
    if k == 0:
        print("No images available – abort.")
        return

    # Create the grid plot
    num_steps = len(all_steps)
    fig, axes = plt.subplots(k, num_steps, figsize=(2 * num_steps, 2 * k))
    
    # Handle case where k=1 or num_steps=1
    if k == 1 and num_steps == 1:
        axes = [[axes]]
    elif k == 1:
        axes = [axes]
    elif num_steps == 1:
        axes = [[ax] for ax in axes]

    for step_idx, (step, images) in enumerate(zip(all_steps, all_images_per_step)):
        for img_idx in range(k):
            ax = axes[img_idx][step_idx]
            ax.imshow(np.array(images[img_idx]))
            ax.axis('off')
            
            # Add step label on top row
            if img_idx == 0:
                ax.set_title(f'Step {step}', fontsize=10)
            
            # Add image index label on first column
            if step_idx == 0:
                ax.set_ylabel(f'Image {img_idx + 1}', fontsize=10)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, plot_name)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("Evolution grid saved →", plot_path)


# ─── CLI helper ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_path", required=True,
                    help="ENTITY/PROJECT/RUN_ID (exactly as in the URL)")
    ap.add_argument("--out_dir",  default="frames")
    ap.add_argument("--gif_name", default="samples.gif")
    ap.add_argument("--fps",      type=int, default=1)
    ap.add_argument("--make_grid", action="store_true",
                    help="Also create evolution grid plot")
    ap.add_argument("--k_images", type=int, default=5,
                    help="Number of images to track in evolution grid")
    ap.add_argument("--plot_name", default="evolution_grid.png")
    ns = ap.parse_args(sys.argv[1:])
    
    #make_gif(ns.run_path, ns.out_dir, ns.gif_name, ns.fps)
    
    if ns.make_grid:
        sample_and_plot_grid(ns.run_path, ns.k_images, ns.out_dir, ns.plot_name)
