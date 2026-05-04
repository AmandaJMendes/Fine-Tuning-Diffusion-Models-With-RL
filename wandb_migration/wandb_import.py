"""
Re-create W&B runs from a local export directory (produced by wandb_export.py).
Run this on the machine logged into the destination account.

Usage:
    python scripts/wandb_import.py \
        --entity new-account \
        --project your-project \
        --input_dir artifacts/wandb_export \
        [--run_ids abc123 def456 ...]   # omit to import all exported runs
"""

import argparse
import json
import sys
from pathlib import Path

import wandb


def _resolve_media(val: dict, run_dir: Path):
    """Convert a W&B media dict from history into a loggable wandb.Image (or None)."""
    media_type = val.get("_type", "")
    if media_type == "images/separated":
        imgs = []
        for fname in val.get("filenames", []):
            fpath = run_dir / fname
            if fpath.exists():
                imgs.append(wandb.Image(str(fpath)))
            else:
                print(f"    Warning: missing media file {fpath}")
        return imgs or None
    if media_type == "image-file":
        fname = val.get("path")
        if fname:
            fpath = run_dir / fname
            if fpath.exists():
                return wandb.Image(str(fpath))
            print(f"    Warning: missing media file {fpath}")
    return None


def import_run(run_dir: Path, entity: str, project: str) -> None:
    metadata = json.loads((run_dir / "metadata.json").read_text())

    run = wandb.init(
        entity=entity,
        project=project,
        name=metadata["name"],
        config=metadata["config"],
        tags=metadata.get("tags") or [],
        notes=metadata.get("notes"),
        group=metadata.get("group"),
        job_type=metadata.get("job_type"),
        reinit=True,
    )

    # Prefer the raw wandb-history.jsonl (has every log() call including timesteps/t=* and
    # grad stats). Fall back to history.jsonl if not present.
    raw_path = run_dir / "wandb-history.jsonl"
    fallback_path = run_dir / "history.jsonl"
    history_file = raw_path if raw_path.exists() else fallback_path
    history_text = history_file.read_text().strip() if history_file.exists() else ""

    if not history_text:
        print("  [history] empty, skipping")
    else:
        rows = [json.loads(line) for line in history_text.splitlines() if line.strip()]
        rows.sort(key=lambda r: (r.get("_step") or 0, r.get("_timestamp") or 0))
        seen_steps: set = set()
        rows = [r for r in rows if (s := r.get("_step")) not in seen_steps and not seen_steps.add(s)]
        for row in rows:
            step = row.pop("_step", None)
            row.pop("_timestamp", None)
            row.pop("_runtime", None)

            log_row = {}
            for key, val in row.items():
                if isinstance(val, dict) and val.get("_type", "").startswith("image"):
                    resolved = _resolve_media(val, run_dir)
                    if resolved is not None:
                        log_row[key] = resolved
                else:
                    log_row[key] = val

            if step is not None:
                wandb.log(log_row, step=int(step))
            else:
                wandb.log(log_row)
        print(f"  [history] {len(rows)} rows logged")

    # Re-upload artifacts
    artifacts_dir = run_dir / "artifacts"
    if artifacts_dir.exists():
        for art_dir in sorted(artifacts_dir.iterdir()):
            if not art_dir.is_dir():
                continue
            meta_file = art_dir / "_artifact_meta.json"
            if meta_file.exists():
                art_meta = json.loads(meta_file.read_text())
                art_name = art_meta["name"].split(":")[0]  # strip version suffix
                art_type = art_meta["type"]
            else:
                art_name = art_dir.name
                art_type = "config"

            if art_type in ("wandb-history", "code"):
                continue
            artifact = wandb.Artifact(art_name, type=art_type)
            for f in sorted(art_dir.rglob("*")):
                if f.is_file() and f.name != "_artifact_meta.json":
                    artifact.add_file(str(f), name=f.relative_to(art_dir).as_posix())
            wandb.log_artifact(artifact)
            print(f"  [artifact] {art_name} uploaded")

    wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", required=True, help="Destination W&B entity (username or team)")
    parser.add_argument("--project", required=True, help="Destination W&B project name")
    parser.add_argument("--input_dir", default="artifacts/wandb_export", help="Local export directory")
    parser.add_argument("--run_ids", nargs="*", help="Specific run IDs to import (default: all exported runs)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: {input_dir} does not exist.", file=sys.stderr)
        sys.exit(1)

    if args.run_ids:
        run_dirs = [input_dir / rid for rid in args.run_ids]
    else:
        run_dirs = sorted(d for d in input_dir.iterdir() if d.is_dir())

    print(f"Importing {len(run_dirs)} run(s) into {args.entity}/{args.project}")
    for i, run_dir in enumerate(run_dirs, 1):
        if not (run_dir / "metadata.json").exists():
            print(f"\n[{i}] Skipping {run_dir.name} — no metadata.json found")
            continue
        metadata = json.loads((run_dir / "metadata.json").read_text())
        print(f"\n[{i}/{len(run_dirs)}] {metadata['name']} (original id: {run_dir.name})")
        try:
            import_run(run_dir, args.entity, args.project)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)

    print("\nDone.")


if __name__ == "__main__":
    main()
