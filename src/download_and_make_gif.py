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

def get_images_from_wandb(run_path, out_dir="frames", history_key="eval/samples"):
    os.makedirs(out_dir, exist_ok=True)
    run   = wandb.Api().run(run_path)

    samples_dict = {}

    for row in tqdm.tqdm(
            run.scan_history(keys=["_step", history_key]), desc="steps"):
        step        = row["_step"]
        meta        = row[history_key] or {}             # always a dict
        filenames   = meta.get("filenames", [])          # slot-order list
        if not filenames:
            continue

        local_paths = []
        for idx, rel in enumerate(filenames):                            # already ordered!
            local = run.file(rel).download(
                        root=out_dir,
                        replace=False,
                        exist_ok=True)                   # cached after 1st run
            local_path = local if isinstance(local, str) else local.name
            
            # Rename the file to the new format
            new_filename = f"sample_{idx}_step_{step}.png"
            new_path = os.path.join(out_dir, new_filename)
            os.rename(local_path, new_path)
            
            local_paths.append(new_path)

        pil_imgs  = [Image.open(p).convert("RGB") for p in local_paths]

        samples_dict[step] = pil_imgs

    return samples_dict

def get_images_from_local(frames_dir):
    samples_dict = {}  

    for file in os.listdir(frames_dir):
        if file.endswith(".png") and "sample_" in file and "_step_" in file:
            # Parse filename: sample_{idx}_step_{step}.png
            parts = file.split("_")
            if len(parts) >= 4:
                try:
                    idx = int(parts[1])
                    step = int(parts[3].split(".")[0])  # Remove .png extension
                    
                    if step not in samples_dict:
                        samples_dict[step] = []
                    
                    img_path = os.path.join(frames_dir, file)
                    samples_dict[step].append((idx, Image.open(img_path).convert("RGB")))
                except (ValueError, IndexError):
                    continue
    
    for step in samples_dict:
        samples_dict[step] = sorted(samples_dict[step], key=lambda x: x[0])
        samples_dict[step] = [x[1] for x in samples_dict[step]]

    return samples_dict

def make_gif(samples_dict, out_dir="frames", gif_name="samples.gif", fps=4):
    os.makedirs(out_dir, exist_ok=True)

    sorted_steps = sorted(samples_dict.keys())

    frames = []

    for step in sorted_steps:
        pil_imgs = samples_dict[step]
        grid      = create_grid(pil_imgs)                # one grid per step
        frames.append(np.asarray(grid))

    if not frames:
        print("Nothing found – abort.")
        return

    gif_path = os.path.join(out_dir, gif_name)
    imageio.mimsave(gif_path, frames, fps=fps)
    print("GIF saved →", gif_path)

def sample_and_plot_grid(samples_dict, k=5, out_dir="frames", plot_name="evolution_grid.png"):
    """
    Sample k images and plot them in a grid where each column is a step 
    and each row is one of the k images.
    """
    os.makedirs(out_dir, exist_ok=True)

    sorted_steps = sorted(samples_dict.keys())

    # Sample k images consistently across all steps
    num_images_available = min([len(samples_dict[step]) for step in sorted_steps])
    k = min(k, num_images_available)
    
    if k == 0:
        print("No images available – abort.")
        return

    # Create the grid plot
    num_steps = len(samples_dict)
    fig, axes = plt.subplots(k, num_steps, figsize=(2 * num_steps, 2 * k))
    
    # Handle case where k=1 or num_steps=1
    if k == 1 and num_steps == 1:
        axes = [[axes]]
    elif k == 1:
        axes = [axes]
    elif num_steps == 1:
        axes = [[ax] for ax in axes]

    for step_idx, step in enumerate(sorted_steps):
        for img_idx in range(k):
            ax = axes[img_idx][step_idx]
            ax.imshow(np.array(samples_dict[step][img_idx]))
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

if __name__ == "__main__":
    import argparse, sys

    ap = argparse.ArgumentParser()
    ap.add_argument("--run_path",
                    help="ENTITY/PROJECT/RUN_ID (for W&B runs)")
    ap.add_argument("--frames_dir",
                    help="Path to local folder containing renamed frames")
    ap.add_argument("--out_dir",  default="frames")
    ap.add_argument("--gif_name", default="samples.gif")
    ap.add_argument("--fps",      type=int, default=1)
    ap.add_argument("--make_grid", action="store_true",
                    help="Create evolution grid plot")
    ap.add_argument("--make_gif", action="store_true",
                    help="Create GIF animation")
    ap.add_argument("--k_images", type=int, default=5,
                    help="Number of images to track in evolution grid")
    ap.add_argument("--plot_name", default="evolution_grid.png")
    ap.add_argument("--local_only", action="store_true",
                    help="Work with local frames only (no W&B)")
    args = ap.parse_args(sys.argv[1:])

    # 1) Load your dict of step→[PIL images]
    if args.local_only:
        if not args.frames_dir:
            ap.error("--frames_dir is required when using --local_only")
        samples = get_images_from_local(args.frames_dir)
        print(samples)
    else:
        if not args.run_path:
            ap.error("Either --run_path or --frames_dir must be provided")
        samples = get_images_from_wandb(
            args.run_path, out_dir=args.out_dir)

    # 2) Determine what to create (default to GIF if nothing specified)
    if not args.make_grid and not args.make_gif:
        args.make_gif = True

    # 3) Create outputs
    if args.make_grid:
        sample_and_plot_grid(
            samples,
            k=args.k_images,
            out_dir=args.out_dir,
            plot_name=args.plot_name
        )
    
    if args.make_gif:
        make_gif(
            samples,
            out_dir=args.out_dir,
            gif_name=args.gif_name,
            fps=args.fps
        )
