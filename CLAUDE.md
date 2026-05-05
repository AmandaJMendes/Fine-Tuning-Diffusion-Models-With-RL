# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Paper

**"Policy Gradient Fine-Tuning of Diffusion Models with Timestep-Aware Optimization"** — ECCV 2026 submission (#14843).

Two reward-dependent importance measures drive the timestep weighting:

- **Reward Sensitivity ΔR(t)**: average absolute change in reward when an image is corrupted to noise level t and deterministically reconstructed — measures how much a timestep "controls" reward-relevant features.
- **Reward Variance σ²R(t)**: variance of rewards across M independent reconstructions at the same noise level t — measures instability/freedom at that noise level.

## What Is and Isn't Part of the Paper

**Core (paper-relevant) code:**
- `scripts/train.py` — main PPO training loop on DDPM-CelebA-HQ
- `tao_diffusion/custom_ddim_scheduler.py` — DDIM scheduler extended with log-probability computation
- `tao_diffusion/rewards.py` — reward pipeline (ImageReward + binary gender score)
- `tao_diffusion/sampling.py` — trajectory generation: pure noise → denoising loop, collecting latents and log-probs
- `tao_diffusion/timestep_metrics.py` — implements reward_sensitivity, reward_variance, snr_weight
- `scripts/profile_timesteps.py` — distributed computation of ΔR(t) and σ²R(t) metrics
- `scripts/compute_weights.py` — produces weight JSON files from profiling CSVs
- `analysis/timestep_profiling_analysis.ipynb` — visualizes ΔR(t) and σ²R(t) per timestep; includes sample size comparison, reconstruction grids, and per-image GIFs
- `analysis/sampling_strategy_comparison.ipynb` — compares all sampling strategies using training run data
- `configs/` — one YAML per sampling strategy
- `artifacts/weights/` — pre-computed timestep weight JSON files

**Not part of the paper (exploratory/peripheral):**
- `scripts/train_sd.py` — Stable Diffusion variant, not used in experiments
- `analysis/visualize_sample_evolution.py` — downloads W&B frames and creates GIFs
- `analysis/initial_training_exploration.ipynb` — early exploratory notebook
- `paper/formulas.py` — renders LaTeX formulas to PNGs

## Architecture

### Training loop (`scripts/train.py`)

```
Sample trajectories
  └─ pure noise xT → DDIM denoising for T=50 steps
     Each step t: collect (x_t, x_{t-1}, log p_θ(x_{t-1}|x_t))

Compute reward on final x_0
  └─ r(x_0) = IR(x_0) + 2 · 𝟙[Gender(x_0) ≥ 0.8]
     Advantage = (r - mean) / std  across all GPUs

PPO inner loop (per sampling batch)
  └─ Draw timesteps t ~ p_φ(t)  [the weighting distribution — the paper's key variable]
     For each sampled t:
       Rescore (x_t → x_{t-1}) with current θ
       ratio = exp(new_log_prob − old_log_prob)
       loss = −min(ratio·adv, clip(ratio, 1±ε)·adv)   ε = 1e-4
       Backprop → optimizer step
```

### Reward-aware metric pipeline

```
scripts/profile_timesteps.py
  └─ For each image x_0^(i) (N=10):
       For each timestep t ∈ {1..50}:
         Sample M=10 corruptions x_t^(i,j) via forward diffusion
         Denoise deterministically → x̂_0^(i,j)
         Evaluate r(x̂_0^(i,j)) for all three reward models
       Output: artifacts/timestep_profiles/worker_*/reconstruction_scores.csv

tao_diffusion/timestep_metrics.py
  └─ load_perceptual_analysis_data() — merges worker CSVs
     reward_sensitivity()  → ΔR(t)
     reward_variance()     → σ²R(t)
     snr_weight()          → w_SNR(t)

scripts/compute_weights.py
  └─ Calls timestep_metrics functions, normalizes
     → artifacts/weights/*.json  (consumed via --timestep_weights at training time)
```

### Key files

| File | Role |
|---|---|
| `tao_diffusion/custom_ddim_scheduler.py` | Extends `DDIMScheduler`; `step()` returns the denoised sample **and** a Gaussian log-probability for the transition. This avoids re-running the forward pass during the PPO rescore. |
| `tao_diffusion/rewards.py` | Loads ImageReward, gender classifier (`rizvandwiki/gender-classification`), LAION aesthetics predictor. Default combined reward: `IR + 2·𝟙[gender ≥ 0.8]`. |
| `tao_diffusion/sampling.py` | `generate_batch()` runs the full denoising loop, collecting `(latents, next_latents, log_probs, timesteps)` for PPO reuse. |
| `scripts/train.py` | Full distributed training loop: trajectory collection, reward normalization, PPO updates, W&B logging, evaluation with fixed seeds. |
| `scripts/profile_timesteps.py` | Multi-worker (torchrun) computation of reward sensitivity and variance across all timesteps. |
| `tao_diffusion/timestep_metrics.py` | Pure-function module implementing reward_sensitivity, reward_variance, snr_weight. Consumed by `scripts/compute_weights.py` and the profiling analysis notebook. |
| `analysis/timestep_profiling_analysis.ipynb` | Visualizes ΔR(t) and σ²R(t) across timesteps; includes sample size comparison, reconstruction quality grids, and per-image GIFs. |

## Lint and format

Ruff, line-length 100, Python 3.10+. Pre-commit hooks run automatically on `git commit`.

```bash
ruff check --fix tao_diffusion/ scripts/ analysis/
ruff format tao_diffusion/ scripts/ analysis/
```

## Non-Obvious Design Details

**PPO clipping ε = 1e-4 (not 0.2).** Diffusion policies have high entropy; standard PPO clipping permits policy shifts large enough to cause catastrophic forgetting. The tight clip keeps each update small.

**Log-probability inside the scheduler.** `custom_ddim_scheduler.step()` computes `log p_θ(x_{t-1}|x_t)` as a Gaussian log-likelihood in-place. The `rescore_batch()` function in `train.py` calls `step()` with current model weights to get the new log-prob for the importance ratio — no second forward pass needed.

**Decoupled evaluation RNG.** Evaluation seeds are `fixed_seed + batch_idx + rank`, completely independent from the training RNG, so evaluation never perturbs the training trajectory.

**Aesthetics predictor is tracked but not in the reward.** The paper reports all three metrics (gender, ImageReward, aesthetics) but the training reward is only `IR + 2·Gender`.

**t=0 always excluded from training.** The final denoising step has near-zero noise variance, producing a degenerate log-probability. `timestep_weights[0]` is forced to 0 regardless of the weight file.
