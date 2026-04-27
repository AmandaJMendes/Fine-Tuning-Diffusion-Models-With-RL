"""
Compute timestep weight JSONs from perceptual analysis CSVs.

Usage:
    python scripts/compute_weights.py \\
        --data_dir p2_reward/ \\
        --output_dir weights/ \\
        --metric sensitivity \\
        --reward_col reward
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tao_diffusion.timestep_metrics import (
    load_perceptual_analysis_data,
    reward_sensitivity,
    reward_variance,
    snr_weight,
    subsample_data,
)

METRICS = ["sensitivity", "variance", "snr", "snr_x_sensitivity", "snr_x_variance"]


def compute_metric(df: pd.DataFrame, metric: str, reward_col: str) -> pd.Series:
    if metric == "sensitivity":
        return reward_sensitivity(df, reward_col)
    elif metric == "variance":
        return reward_variance(df, reward_col)
    elif metric == "snr":
        return snr_weight(df)
    elif metric == "snr_x_sensitivity":
        return snr_weight(df) * reward_sensitivity(df, reward_col)
    elif metric == "snr_x_variance":
        return snr_weight(df) * reward_variance(df, reward_col)


def save_weights(series: pd.Series, output_path: str) -> None:
    out_dict = {int(k): (None if pd.isna(v) else float(v)) for k, v in series.to_dict().items()}
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out_dict, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute timestep weight JSONs from perceptual analysis CSVs."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing worker_* CSV folders from run_perceptual_analysis.py.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the output JSON file.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        required=True,
        choices=METRICS,
        help="Weighting strategy to compute.",
    )
    parser.add_argument(
        "--reward_col",
        type=str,
        default="reward",
        help="Reward column to use. Ignored for --metric snr. (default: reward)",
    )
    parser.add_argument(
        "--n_images",
        type=int,
        default=None,
        help="Number of images to subsample. Uses all images if not set.",
    )
    parser.add_argument(
        "--m_reconstructions",
        type=int,
        default=None,
        help="Number of reconstructions per image to subsample. Uses all if not set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subsampling (default: 42).",
    )
    args = parser.parse_args()

    print(f"Loading CSVs from {args.data_dir} ...")
    df = load_perceptual_analysis_data(args.data_dir)
    print(f"Loaded {len(df)} rows, {df['image_idx'].nunique()} images.")

    if args.n_images is not None or args.m_reconstructions is not None:
        n = args.n_images or df["image_idx"].nunique()
        m = args.m_reconstructions or int(df["sample_idx"].dropna().nunique())
        df = subsample_data(df, n_images=n, m_reconstructions=m, seed=args.seed)
        print(f"Subsampled to {df['image_idx'].nunique()} images, {df['sample_idx'].dropna().nunique()} reconstructions.")

    print(f"Computing '{args.metric}' weights for reward_col='{args.reward_col}' ...")
    weights = compute_metric(df, args.metric, args.reward_col)

    suffix = f"_n{n}_m{m}" if (args.n_images is not None or args.m_reconstructions is not None) else ""
    output_path = os.path.join(args.output_dir, f"{args.metric}_{args.reward_col}{suffix}.json")
    save_weights(weights, output_path)
    print(f"Saved to {output_path}")
