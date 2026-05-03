# Policy Gradient Fine-Tuning of Diffusion Models with Timestep-Aware Optimization

Code for the ECCV 2026 paper of the same name. The project is also referred to as **TAO-Diffusion** (Timestep-Aware Optimization).

> **[Paper — link coming soon](#citation)**

Standard policy-gradient fine-tuning of diffusion models applies the same terminal reward uniformly across all T denoising timesteps. This is neither computationally nor convergence-efficient. We propose a **weighted policy-gradient estimator** that samples only a subset of timesteps per update, guided by two reward-dependent importance measures:

- **Reward Sensitivity ΔR(t)** — how much a timestep controls reward-relevant features
- **Reward Variance σ²R(t)** — generative instability / freedom at a given noise level

**Key result:** combining SNR-based weighting with Reward Sensitivity or Variance achieves **>3× faster convergence using only 40% of timesteps** (20 out of 50) on CelebA-HQ.

---

## Repository structure

```
tao_diffusion/
  custom_ddim_scheduler.py  # DDIM scheduler extended with log-probability computation
  rewards.py                # ImageReward + binary gender score pipeline
  sampling.py               # trajectory generation (noise → denoising loop)
  timestep_metrics.py       # reward_sensitivity, reward_variance, snr_weight

scripts/
  train.py                  # main PPO training loop (distributed, multi-GPU)
  profile_timesteps.py      # distributed computation of ΔR(t) and σ²R(t)
  compute_weights.py        # converts profile CSVs into timestep weight JSONs

configs/                    # one YAML per sampling strategy (pass to --config)
artifacts/
  weights/                  # pre-computed timestep weight JSON files
  timestep_profiles/        # CSV outputs from profile_timesteps.py

analysis/
  timestep_profiling_analysis.ipynb   # visualization of reward metrics per timestep
  sampling_strategy_comparison.ipynb  # compares strategies using W&B run data
  visualize_sample_evolution.py       # downloads W&B eval frames and renders evolution GIFs
```

---

## Setup

**1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't** (follow the official instructions for your platform).

**2. Install the package and all dependencies:**
```bash
uv sync
```

**Requirements:** Python ≥ 3.10. No GPU is strictly required, but training and profiling benefit significantly from multiple GPUs — the default configs assume 4.

---

## Workflow

There are three steps: profile timesteps → compute weights → train. Pre-computed weight files for the CelebA-HQ experiments are already committed in `artifacts/weights/`, so you can skip steps 1–2 to reproduce the paper results directly.

### Step 1 — Profile timesteps

Measures ΔR(t) and σ²R(t) across the 50 denoising steps by running forward-diffuse-then-reconstruct for N images × M noise samples per timestep.

```bash
uv run torchrun --nproc_per_node=4 scripts/profile_timesteps.py \
  --num_images 10 \
  --num_reconstructions 10 \
  --output_dir artifacts/timestep_profiles \
  --batch_size 10
```

Output: `artifacts/timestep_profiles/worker_*/reconstruction_scores.csv`

### Step 2 — Compute timestep weights

Converts the profile CSVs into a JSON weight file consumed by the training script:

```bash
uv run python scripts/compute_weights.py \
  --data_dir artifacts/timestep_profiles/ \
  --output_dir artifacts/weights/ \
  --metric snr_x_sensitivity \
  --reward_col reward
```

Available `--metric` values: `sensitivity`, `variance`, `snr`, `snr_x_sensitivity`, `snr_x_variance`.

To reproduce all five weight files used in the paper, run the command once per metric.

### Step 3 — Train

Pass a config file to select the timestep sampling strategy:

```bash
uv run accelerate launch scripts/train.py --config configs/sensitivity_snr.yaml
```

Any argument in the config can be overridden on the command line:

```bash
uv run accelerate launch scripts/train.py \
  --config configs/sensitivity_snr.yaml \
  --learning_rate 5e-7 \
  --num_epochs 200
```

**All seven strategies and their configs:**

| Config | Strategy | Timesteps used | Weight w(t) |
|---|---|---|---|
| `full_trajectory.yaml` | Full-Trajectory (baseline) | all 50 | uniform |
| `uniform.yaml` | Uniform | 20 | uniform |
| `snr.yaml` | SNR | 20 | `1 / (1 + SNR(t))` |
| `sensitivity.yaml` | Sensitivity | 20 | ΔR(t) |
| `variance.yaml` | Variance | 20 | σ²R(t) |
| `sensitivity_snr.yaml` | Sensitivity × SNR | 20 | ΔR(t) · SNR weight |
| `variance_snr.yaml` | Variance × SNR | 20 | σ²R(t) · SNR weight |

---

## Experiment tracking

All runs log to [Weights & Biases](https://wandb.ai). Tracked metrics include total reward mean/std, per-model scores (`ir_person`, `sex_score`, `aesthetics_score`), gradient statistics per timestep (with `--log_grad_stats`), timestep sampling histograms, and eval image grids.

Set `wandb_project` in your config or pass `--wandb_project <name>` on the command line.

---

## Citation

```bibtex
@inproceedings{mendes2026tao,
  title     = {Policy Gradient Fine-Tuning of Diffusion Models with Timestep-Aware Optimization},
  author    = {Mendes, Amanda et al.},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026},
}
```

*(Paper link will be added upon publication.)*
