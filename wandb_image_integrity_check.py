import wandb
from PIL import Image
import io
from tqdm import tqdm
import argparse


def check_images(run_path: str):
    api = wandb.Api()
    run = api.run(run_path)

    files = run.files()

    image_exts = (".png", ".jpg", ".jpeg", ".webp")
    image_files = [f for f in files if f.name.lower().endswith(image_exts)]

    bad_images = []
    ok = 0

    for f in tqdm(image_files, desc=f"Checking images in {run_path}"):
        try:
            file_obj = f.download(replace=True)

            with open(file_obj.name, "rb") as fp:
                img_bytes = fp.read()

            img = Image.open(io.BytesIO(img_bytes))
            img.load()  # force decode

            ok += 1

        except Exception as e:
            bad_images.append((f.name, str(e)))

    print("\nDone.")
    print("Run:", run_path)
    print("Valid images:", ok)
    print("Corrupted images:", len(bad_images))

    if bad_images:
        print("\nExamples:")
        for name, err in bad_images[:10]:
            print(name, "->", err)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check corrupted images in a W&B run")
    parser.add_argument(
        "run_path",
        type=str,
        help="W&B run path in format: entity/project/run_id",
    )

    args = parser.parse_args()
    check_images(args.run_path)