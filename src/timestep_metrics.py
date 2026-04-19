import glob
import os

import pandas as pd
import torch


def load_perceptual_analysis_data(
    data_dir: str,
    csv_pattern: str = "worker_*/scores_per_image_timestep.csv",
) -> pd.DataFrame:
    """
    Load and concatenate all worker CSVs produced by run_perceptual_analysis.py.

    Each worker writes its own CSV with locally-scoped image_idx values.
    This function re-indexes them to be globally unique across workers.

    Args:
        data_dir: Root directory containing the worker output folders.
        csv_pattern: Glob pattern relative to data_dir to locate CSVs.
                     Override if the analysis script used a different structure.

    Returns:
        A single concatenated DataFrame with globally unique image_idx values.
    """
    csv_files = sorted(glob.glob(os.path.join(data_dir, csv_pattern)))
    if not csv_files:
        raise FileNotFoundError(f"No CSVs found matching {os.path.join(data_dir, csv_pattern)}")

    all_data = []
    offset = 0
    for path in csv_files:
        df = pd.read_csv(path)
        if df["image_idx"].notnull().any():
            df.loc[df["image_idx"].notnull(), "image_idx"] += offset
            offset = int(df["image_idx"].max()) + 1
        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)


def reward_sensitivity(df: pd.DataFrame, reward_col: str = "reward") -> pd.Series:
    """
    Reward Sensitivity ΔR(t): average absolute change in reward when an image
    is corrupted to noise level t and deterministically reconstructed (Eq. 6).

    Returns a pd.Series indexed by timestep.
    """
    orig = df[df["sample_idx"].isnull()].set_index("image_idx")[reward_col]
    recon = df[df["sample_idx"].notnull()][["timestep", "image_idx", reward_col]].copy()
    recon = recon.merge(orig.rename("_orig"), left_on="image_idx", right_index=True)
    recon["_abs_diff"] = (recon[reward_col] - recon["_orig"]).abs()
    return (
        recon.groupby(["timestep", "image_idx"])["_abs_diff"]
        .mean()
        .groupby("timestep")
        .mean()
        .rename("reward_sensitivity")
    )


def reward_variance(df: pd.DataFrame, reward_col: str = "reward") -> pd.Series:
    """
    Reward Variance σ²R(t): average variance of reward across independent
    reconstructions of the same image corrupted at timestep t (Eq. 7).

    Returns a pd.Series indexed by timestep.
    """
    recon = df[df["sample_idx"].notnull()][["timestep", "image_idx", reward_col]]
    return (
        recon.groupby(["timestep", "image_idx"])[reward_col]
        .var(ddof=0)
        .groupby("timestep")
        .mean()
        .rename("reward_variance")
    )


def snr_weight(
    df: pd.DataFrame,
    reward_col: str | None = None,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    n_steps: int = 1000,
    k: float = 1,
    gamma: float = 2,
) -> pd.Series:
    """
    SNR-based timestep weighting w_SNR(t) = 1 / (k + SNR(t))^γ (Eq. 5).
    Reward-agnostic: reward_col is accepted for interface consistency but ignored.

    Downweights high-SNR (low-noise) timesteps, concentrating computation on
    semantically formative regions of the trajectory. Inspired by Choi et al.
    (2022) P2 weighting.

    Default values (beta_start, beta_end, n_steps) match google/ddpm-celebahq-256.
    Override them when using a different noise schedule.

    Returns a pd.Series indexed by timestep.
    """
    betas = torch.linspace(beta_start, beta_end, steps=n_steps)
    alpha_bar = torch.cumprod(1.0 - betas, dim=0)
    snr_vals = (alpha_bar / (1.0 - alpha_bar)).numpy()
    weights = 1 / (k + snr_vals) ** gamma

    timesteps = sorted(df["timestep"].dropna().unique().astype(int))
    return pd.Series(weights[timesteps], index=pd.Index(timesteps, name="timestep"), name="snr_weight")


