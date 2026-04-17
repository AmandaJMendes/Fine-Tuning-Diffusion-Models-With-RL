# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Paper

**"Policy Gradient Fine-Tuning of Diffusion Models with Timestep-Aware Optimization"** — ECCV 2026 submission (#14843).

The central claim: standard policy-gradient fine-tuning of diffusion models applies the same terminal reward uniformly across all T denoising timesteps, which is neither computationally nor convergence-efficient. This work proposes a **weighted policy-gradient estimator** (Eq. 4 in the paper) that samples only a subset of timesteps per update, guided by novel reward-dependent importance measures:

- **Reward Sensitivity ΔR(t)**: average absolute change in reward when an image is corrupted to noise level t and deterministically reconstructed — measures how much a timestep "controls" reward-relevant features.
- **Reward Variance σ²R(t)**: variance of rewards across M independent reconstructions at the same noise level t — measures instability/freedom at that noise level.

Best result: combining SNR-based weighting (reward-agnostic structural prior) with Reward Sensitivity or Variance achieves **>3× faster convergence using only 40% of timesteps** (20 out of 50) on CelebA-HQ.

## What Is and Isn't Part of the Paper

**Core (paper-relevant) code:**
- `src/main.py` — main PPO training loop on DDPM-CelebA-HQ
- `src/custom_ddim_scheduler.py` — DDIM scheduler extended with log-probability computation, enabling importance sampling
- `src/rewards.py` — reward pipeline (ImageReward + binary gender score)
- `timesteps_analysis/perceptual_analysis_csv.py` — distributed computation of ΔR(t) and σ²R(t) metrics
- `notebooks/reward_metrics.ipynb` — aggregates CSVs into timestep weight JSONs used during training
- Notebooks under `notebooks/` and `analysis/` that compare sampling strategies

**Not part of the paper (exploratory/peripheral):**
- `src/main_sd.py` — Stable Diffusion variant, not used in experiments
- `timesteps_analysis/perceptual_analysis.py` — single-GPU prototype of the analysis
- `timesteps_analysis/dummy.py` — synthetic test data generator
- `utils/` — visualization helpers (GIF creation, LaTeX rendering)

## Architecture

### Training loop (`src/main.py`)

```
Sample trajectories
  └─ pure noise xT → DDIM denoising for T=50 steps
     Each step t: collect (x_t, x_{t-1}, log p_θ(x_{t-1}|x_t))

Compute reward on final x_0
  └─ r(x_0) = IR(x_0) + 2 · 𝟙[Gender(x_0) ≥ 0.8]   [Eq. 8 in paper]
     Advantage = (r - mean) / std  across all GPUs

PPO inner loop (per sampling batch)
  └─ Draw timesteps t ~ p_φ(t)  [the weighting distribution — the paper's key variable]
     For each sampled t:
       Rescore (x_t → x_{t-1}) with current θ
       ratio = exp(new_log_prob − old_log_prob)
       loss = −min(ratio·adv, clip(ratio, 1±ε)·adv)   ε = 1e-4
       Backprop → optimizer step
```

The **timestep sampling distribution p_φ(t)** (Eq. 3) is the variable the paper sweeps across:

| Strategy | Weight w_φ(t) | Source |
|---|---|---|
| Full-Trajectory | uniform over all 50 steps | Baseline |
| Uniform | uniform over 20 sampled steps | Baseline |
| SNR | `1 / (1 + SNR(t))` (k=1, γ=1) | Reward-agnostic |
| Sensitivity | `ΔR(t)` from analysis | Reward-aware |
| Variance | `σ²R(t)` from analysis | Reward-aware |
| Sensitivity × SNR | product of the two | Hybrid (best) |
| Variance × SNR | product of the two | Hybrid (best) |

### Reward-aware metric pipeline

```
perceptual_analysis_csv.py
  └─ For each image x_0^(i) (N=10):
       For each timestep t ∈ {1..50}:
         Sample M=10 corruptions x_t^(i,j) via forward diffusion
         Denoise deterministically → x̂_0^(i,j)
         Evaluate r(x̂_0^(i,j)) for all three reward models
       Output: CSV rows [image_id, timestep, reward_scores...]

notebooks/reward_metrics.ipynb
  └─ Load CSVs from all workers (p2_reward/worker_*/metrics.csv)
     Compute ΔR(t) = mean_i mean_j |r(x_0^i) − r(x̂_0^(i,j))|   [Eq. 6]
     Compute σ²R(t) = mean_i Var_j r(x̂_0^(i,j))                  [Eq. 7]
     Normalize → timestep weight JSON files for --timesteps_weights_json
```

### Key files

| File | Role |
|---|---|
| `src/custom_ddim_scheduler.py` | Extends `DDIMScheduler`; `step()` returns the denoised sample **and** a Gaussian log-probability for the transition. This avoids re-running the forward pass during the PPO rescore. |
| `src/rewards.py` | Loads ImageReward, gender classifier (`rizvandwiki/gender-classification`), LAION aesthetics predictor. Default combined reward: `IR + 2·𝟙[gender ≥ 0.8]`. |
| `src/main.py` | Full distributed training loop: trajectory collection, reward normalization, PPO updates, W&B logging, evaluation with fixed seeds. |
| `timesteps_analysis/perceptual_analysis_csv.py` | Multi-worker (torchrun) computation of reward sensitivity and variance across all timesteps. Outputs CSVs to `p2_reward/`. |
| `notebooks/reward_metrics.ipynb` | Aggregates CSV outputs and generates the timestep weight JSON files consumed by `--timesteps_weights_json` at training time. |

## Commands

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Lint and format** (Ruff, line-length 100, Python 3.10+)
```bash
ruff check --fix src/ timesteps_analysis/ utils/
ruff format src/ timesteps_analysis/ utils/
```
Pre-commit hooks run Ruff automatically on `git commit`.

**Step 1 — Compute reward-aware timestep metrics**
```bash
torchrun --nproc_per_node=4 timesteps_analysis/perceptual_analysis_csv.py \
  --num_gen_images 10 \
  --num_denoised_samples 10 \
  --plot_dir p2_reward \
  --batch_size 10
```
Then run `notebooks/reward_metrics.ipynb` to aggregate CSVs and produce weight JSON files.

**Step 2 — Train with a timestep sampling strategy**

Full-trajectory baseline:
```bash
accelerate launch src/main.py \
  --per_gpu_batch_size 5 \
  --inference_timesteps 50 \
  --full_epochs 100 \
  --learning_rate 1e-6 \
  --eval_every_steps 20 \
  --eval_samples 20
```

With reward-aware / SNR / hybrid weighting (40% timestep budget):
```bash
accelerate launch src/main.py \
  --per_gpu_batch_size 5 \
  --inference_timesteps 50 \
  --num_train_timesteps 20 \
  --timesteps_weights_json path/to/weights.json \
  --full_epochs 100 \
  --learning_rate 1e-6
```

## Non-Obvious Design Details

**PPO clipping ε = 1e-4 (not 0.2).** Diffusion policies have high entropy; standard PPO clipping permits policy shifts large enough to cause catastrophic forgetting. The tight clip keeps each update small.

**Log-probability inside the scheduler.** `custom_ddim_scheduler.step()` computes `log p_θ(x_{t-1}|x_t)` as a Gaussian log-likelihood in-place. The `rescore_batch()` function in `main.py` calls `step()` with current model weights to get the new log-prob for the importance ratio — no second forward pass needed.

**Decoupled evaluation RNG.** Evaluation seeds are `fixed_seed + batch_idx + rank`, completely independent from the training RNG, so evaluation never perturbs the training trajectory.

**Aesthetics predictor is tracked but not in the reward.** The paper reports all three metrics (gender, ImageReward, aesthetics) but the training reward is only `IR + 2·Gender`, consistent with Eq. 8.

## Experiment Tracking

All runs log to **Weights & Biases**. Metrics: total reward mean/std, per-model scores (ir_person, sex_score, aesthetics_score), gradient norms/mean/std per timestep, timestep sampling histograms, eval image grids.
