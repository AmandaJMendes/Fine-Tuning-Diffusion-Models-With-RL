# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Paper

**"Policy Gradient Fine-Tuning of Diffusion Models with Timestep-Aware Optimization"** — ECCV 2026 submission (#14843).

The central claim: standard policy-gradient fine-tuning of diffusion models applies the same terminal reward uniformly across all T denoising timesteps, which is neither computationally nor convergence-efficient. This work proposes a **weighted policy-gradient estimator** that samples only a subset of timesteps per update, guided by novel reward-dependent importance measures:

- **Reward Sensitivity ΔR(t)**: average absolute change in reward when an image is corrupted to noise level t and deterministically reconstructed — measures how much a timestep "controls" reward-relevant features.
- **Reward Variance σ²R(t)**: variance of rewards across M independent reconstructions at the same noise level t — measures instability/freedom at that noise level.

Best result: combining SNR-based weighting (reward-agnostic structural prior) with Reward Sensitivity or Variance achieves **>3× faster convergence using only 40% of timesteps** (20 out of 50) on CelebA-HQ.

## What Is and Isn't Part of the Paper

**Core (paper-relevant) code:**
- `scripts/train.py` — main PPO training loop on DDPM-CelebA-HQ
- `tao_diffusion/custom_ddim_scheduler.py` — DDIM scheduler extended with log-probability computation, enabling importance sampling
- `tao_diffusion/rewards.py` — reward pipeline (ImageReward + binary gender score)
- `tao_diffusion/sampling.py` — trajectory generation: pure noise → denoising loop, collecting latents and log-probs
- `tao_diffusion/timestep_metrics.py` — functions implementing reward_sensitivity, reward_variance, snr_weight; consumed by the notebook and compute_weights script
- `scripts/profile_timesteps.py` — distributed computation of ΔR(t) and σ²R(t) metrics
- `scripts/compute_weights.py` — CLI alternative to the notebook for producing weight JSON files
- `analysis/timestep_profiling_analysis.ipynb` — aggregates CSVs into timestep weight JSONs used during training
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

The **timestep sampling distribution p_φ(t)** is the variable the paper sweeps across:

| Strategy | Config | Weight w_φ(t) |
|---|---|---|
| Full-Trajectory | `full_trajectory.yaml` | uniform over all 50 steps |
| Uniform | `uniform.yaml` | uniform over 20 sampled steps |
| SNR | `snr.yaml` | `1 / (1 + SNR(t))` (k=1, γ=1) |
| Sensitivity | `sensitivity.yaml` | `ΔR(t)` from analysis |
| Variance | `variance.yaml` | `σ²R(t)` from analysis |
| Sensitivity × SNR | `sensitivity_snr.yaml` | product of the two |
| Variance × SNR | `variance_snr.yaml` | product of the two |

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

analysis/timestep_profiling_analysis.ipynb  (or scripts/compute_weights.py)
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
| `tao_diffusion/timestep_metrics.py` | Pure-function module implementing reward_sensitivity, reward_variance, snr_weight. Consumed by both the notebook and `scripts/compute_weights.py`. |
| `analysis/timestep_profiling_analysis.ipynb` | Aggregates CSV outputs and generates the timestep weight JSON files consumed by `--timestep_weights` at training time. |

## Commands

**Install dependencies**
```bash
pip install -e .
```

**Lint and format** (Ruff, line-length 100, Python 3.10+)
```bash
ruff check --fix tao_diffusion/ scripts/ analysis/
ruff format tao_diffusion/ scripts/ analysis/
```
Pre-commit hooks run Ruff automatically on `git commit`.

**Step 1 — Compute reward-aware timestep metrics**
```bash
torchrun --nproc_per_node=4 scripts/profile_timesteps.py \
  --num_images 10 \
  --num_reconstructions 10 \
  --output_dir artifacts/timestep_profiles \
  --batch_size 10
```
Then run `analysis/timestep_profiling_analysis.ipynb` (or `scripts/compute_weights.py`) to aggregate CSVs and produce weight JSON files.

**Step 2 — Train with a timestep sampling strategy**

Using a config file (recommended):
```bash
accelerate launch scripts/train.py --config configs/sensitivity_snr.yaml
```

Full-trajectory baseline (no config):
```bash
accelerate launch scripts/train.py \
  --local_batch_size 5 \
  --num_denoising_steps 50 \
  --num_epochs 100 \
  --learning_rate 1e-6 \
  --eval_every_steps 20 \
  --eval_samples 20
```

With reward-aware / SNR / hybrid weighting (40% timestep budget):
```bash
accelerate launch scripts/train.py \
  --local_batch_size 5 \
  --num_denoising_steps 50 \
  --timesteps_per_update 20 \
  --timestep_weights artifacts/weights/snr_x_sensitivity_reward.json \
  --num_epochs 100 \
  --learning_rate 1e-6
```

## Non-Obvious Design Details

**PPO clipping ε = 1e-4 (not 0.2).** Diffusion policies have high entropy; standard PPO clipping permits policy shifts large enough to cause catastrophic forgetting. The tight clip keeps each update small.

**Log-probability inside the scheduler.** `custom_ddim_scheduler.step()` computes `log p_θ(x_{t-1}|x_t)` as a Gaussian log-likelihood in-place. The `rescore_batch()` function in `train.py` calls `step()` with current model weights to get the new log-prob for the importance ratio — no second forward pass needed.

**Decoupled evaluation RNG.** Evaluation seeds are `fixed_seed + batch_idx + rank`, completely independent from the training RNG, so evaluation never perturbs the training trajectory.

**Aesthetics predictor is tracked but not in the reward.** The paper reports all three metrics (gender, ImageReward, aesthetics) but the training reward is only `IR + 2·Gender`.

**t=0 always excluded from training.** The final denoising step has near-zero noise variance, producing a degenerate log-probability. `timestep_weights[0]` is forced to 0 regardless of the weight file.

## Experiment Tracking

All runs log to **Weights & Biases**. Metrics: total reward mean/std, per-model scores (ir_person, sex_score, aesthetics_score), gradient norms/mean/std per timestep (when `--log_grad_stats` is set), timestep sampling histograms, eval image grids.
