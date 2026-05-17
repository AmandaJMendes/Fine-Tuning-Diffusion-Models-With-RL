"""
Export W&B runs to a local directory.

Usage:
    # By run name (as shown in the W&B UI / notebook):
    python scripts/wandb_export.py \
        --entity amanda-mendes-cloudwalk \
        --project diffusion-finetune \
        --output_dir artifacts/wandb_export \
        --run_names morning-wave-19 atomic-smoke-24 royal-plasma-100 ...

    # By run ID (8-char hex):
    python scripts/wandb_export.py \
        --entity amanda-mendes-cloudwalk \
        --project diffusion-finetune \
        --output_dir artifacts/wandb_export \
        --run_ids abc123 def456 ...

    # Export all runs in the project (omit both flags):
    python scripts/wandb_export.py \
        --entity amanda-mendes-cloudwalk \
        --project diffusion-finetune \
        --output_dir artifacts/wandb_export
"""

import argparse
import json
import sys
from pathlib import Path

import wandb


def export_run(run, output_dir: Path) -> None:
    run_dir = output_dir / run.id
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "id": run.id,
        "name": run.name,
        "config": dict(run.config),
        "summary": {k: v for k, v in run.summary.items() if not k.startswith("_")},
        "tags": list(run.tags),
        "notes": run.notes,
        "group": run.group,
        "job_type": run.job_type,
        "state": run.state,
        "created_at": str(run.created_at),
        "entity": run.entity,
        "project": run.project,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(f"  [metadata] saved")

    # Use scan_history() as the source of truth — it queries the wandb backend's merged
    # per-step history, which is authoritative. The raw wandb-history.jsonl uploaded by
    # the client can be missing the final row when the run shuts down before the last
    # log() call flushes (observed: step=1000 eval missing for some runs even though it
    # ran and updated the summary).
    history = list(run.scan_history(page_size=10000))
    (run_dir / "history.jsonl").write_text(
        "\n".join(json.dumps(row, default=str) for row in history)
    )
    print(f"  [history] {len(history)} rows saved")

    # Download every media file referenced in history (eval images, etc.)
    media_files_downloaded = 0
    for row in history:
        for val in row.values():
            if not isinstance(val, dict):
                continue
            media_type = val.get("_type", "")
            filenames = []
            if media_type == "images/separated":
                filenames = val.get("filenames", [])
            elif media_type == "image-file":
                fname = val.get("path")
                if fname:
                    filenames = [fname]
            for fname in filenames:
                dest = run_dir / fname
                if dest.exists():
                    media_files_downloaded += 1
                    continue
                for attempt in range(3):
                    try:
                        run.file(fname).download(root=str(run_dir), replace=True)
                        media_files_downloaded += 1
                        break
                    except Exception as e:
                        if attempt == 2:
                            print(f"  Warning: could not download media file {fname}: {e}")
    print(f"  [media] {media_files_downloaded} files downloaded")

    # Artifacts logged by this run
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    for artifact in run.logged_artifacts():
        art_label = artifact.name.replace("/", "_")
        art_dir = artifacts_dir / f"{art_label}_{artifact.version}"
        artifact.download(root=str(art_dir))
        # Save artifact metadata so the importer knows the type
        art_meta = {"name": artifact.name, "type": artifact.type, "version": artifact.version}
        (art_dir / "_artifact_meta.json").write_text(json.dumps(art_meta, indent=2))
        print(f"  [artifact] {artifact.name}:{artifact.version} saved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", required=True, help="W&B entity (username or team) to export from")
    parser.add_argument("--project", required=True, help="W&B project name")
    parser.add_argument("--output_dir", default="artifacts/wandb_export", help="Local export directory")
    parser.add_argument("--run_ids", nargs="*", help="Specific run IDs (8-char hex) to export")
    parser.add_argument("--run_names", nargs="*", help="Specific run display names to export (e.g. morning-wave-19)")
    args = parser.parse_args()

    if args.run_ids and args.run_names:
        print("Error: use --run_ids or --run_names, not both.", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api(timeout=60)

    if args.run_ids:
        runs = [api.run(f"{args.entity}/{args.project}/{rid}") for rid in args.run_ids]
    elif args.run_names:
        target_names = set(args.run_names)
        all_runs = api.runs(f"{args.entity}/{args.project}")
        runs = [r for r in all_runs if r.name in target_names]
        found_names = {r.name for r in runs}
        missing = target_names - found_names
        if missing:
            print(f"Warning: run names not found in project: {sorted(missing)}", file=sys.stderr)
    else:
        runs = list(api.runs(f"{args.entity}/{args.project}"))

    print(f"Exporting {len(runs)} run(s) to {output_dir}/")
    for i, run in enumerate(runs, 1):
        print(f"\n[{i}/{len(runs)}] {run.name} ({run.id})")
        try:
            export_run(run, output_dir)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)

    print(f"\nDone. Transfer {output_dir}/ to your other machine, then run wandb_import.py.")


if __name__ == "__main__":
    main()
