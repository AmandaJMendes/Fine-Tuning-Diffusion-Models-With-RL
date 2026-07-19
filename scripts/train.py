import argparse
import contextlib
import json
import logging
import math
import os
import shutil
import time
from collections.abc import Generator

import torch
import torch.distributed as dist
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import UNet2DModel
from tqdm import tqdm

import wandb
from tao_diffusion.custom_ddim_scheduler import CustomDDIMScheduler
from tao_diffusion.rewards import (
    DEFAULT_REWARD_PROMPT,
    reward_function,
    tensor_batch_to_pil_images,
)
from tao_diffusion.sampling import generate_batch, generate_eval_samples


@contextlib.contextmanager
def capture_grad_moments(
    model: UNet2DModel, accelerator: Accelerator
) -> Generator[dict, None, None]:
    """
    Context manager that captures per-backward grad statistics without extra allocations.
    Yields a dict populated after the backward pass with keys: N, mean, var, std.
    Empty dict on non-main ranks.
    """
    model = accelerator.unwrap_model(model)

    N = S = Q = 0.0

    def _hook(grad):
        nonlocal N, S, Q
        g = grad.detach()
        N += g.numel()
        S += g.sum().item()
        Q += (g * g).sum().item()

    stats = {}
    handles = [p.register_hook(_hook) for p in model.parameters()]
    try:
        yield stats
    finally:
        for h in handles:
            h.remove()

        reduced = {}
        for name, val in zip(("N", "S", "Q"), (N, S, Q), strict=False):
            t = torch.tensor(val, device=accelerator.device)
            accelerator.reduce(t, reduction="sum")
            reduced[name] = t.item()

        if accelerator.is_main_process:
            N, S, Q = reduced["N"], reduced["S"], reduced["Q"]
            if N == 0:
                stats.update(N=0, mean=0.0, var=0.0, std=0.0)
            else:
                mu = S / N
                var = max(Q / N - mu * mu, 0.0)
                stats.update(N=N, mean=mu, var=var, std=math.sqrt(var))


def rescore_batch(
    model: UNet2DModel,
    scheduler: CustomDDIMScheduler,
    latents: torch.Tensor,  # (B, C, H, W)
    next_latents: torch.Tensor,  # (B, C, H, W)
    timestep: torch.Tensor,  # scalar — same t broadcast across the batch
) -> torch.Tensor:
    """
    Compute log p_θ(next_latents | latents, t) for the current model weights.
    Returns a (B,) tensor of per-sample log-probabilities.
    """
    pred_noise = model(latents, timestep).sample
    _, log_prob = scheduler.step(pred_noise, timestep, latents, next_latents, eta=1.0)
    return log_prob


def check_model_sync(model: UNet2DModel, accelerator: Accelerator, tol: float = 1e-6) -> None:
    """Check that model parameters are identical across all ranks; logs a warning if not."""
    if accelerator.num_processes == 1:
        return

    _logger = logging.getLogger(__name__)
    model = accelerator.unwrap_model(model)
    device = next(model.parameters()).device

    max_diff = torch.tensor(0.0, device=device)
    for p in model.parameters():
        global_max = p.data.clone()
        global_min = p.data.clone()

        dist.all_reduce(global_max, op=dist.ReduceOp.MAX)
        dist.all_reduce(global_min, op=dist.ReduceOp.MIN)

        diff = (global_max - global_min).abs().max()
        max_diff = torch.max(max_diff, diff)

    if accelerator.is_main_process:
        if max_diff <= tol:
            _logger.info(f"Model parameters in sync across all ranks (max |Δ| = {max_diff:.3e})")
        else:
            _logger.warning(f"Parameter mismatch across ranks detected (max |Δ| = {max_diff:.3e})")


def save_checkpoint(
    accelerator: Accelerator,
    checkpoint_dir: str,
    global_step: int,
    epoch: int,
) -> None:
    """
    Save a resumable checkpoint (model + optimizer + RNG + scheduler state) using
    accelerator.save_state, plus a small metadata file with global_step and epoch.

    Writes to a temp directory then atomically swaps it into place, so a crash
    mid-save never corrupts the previous checkpoint.  Only one checkpoint is
    ever kept (checkpoint_dir is overwritten).  Must be called by all ranks.
    """
    tmp_dir = checkpoint_dir + ".tmp"
    if accelerator.is_main_process:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    accelerator.wait_for_everyone()

    accelerator.save_state(tmp_dir)

    if accelerator.is_main_process:
        meta = {"global_step": int(global_step), "epoch": int(epoch)}
        with open(os.path.join(tmp_dir, "training_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        # Atomic swap: remove old, rename new into place.
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        os.rename(tmp_dir, checkpoint_dir)
        logging.getLogger(__name__).info(
            f"Checkpoint saved (epoch={epoch}, global_step={global_step}) → {checkpoint_dir}"
        )
    accelerator.wait_for_everyone()


def load_checkpoint(
    accelerator: Accelerator,
    checkpoint_dir: str,
) -> tuple[int, int]:
    """
    Restore model, optimizer, RNG, and scheduler state from checkpoint_dir.
    Returns (global_step, epoch) saved in the checkpoint's metadata.
    Must be called by all ranks, after accelerator.prepare().
    """
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    accelerator.load_state(checkpoint_dir)
    with open(os.path.join(checkpoint_dir, "training_meta.json")) as f:
        meta = json.load(f)
    global_step = int(meta["global_step"])
    epoch = int(meta["epoch"])
    logging.getLogger(__name__).info(
        f"Resumed from {checkpoint_dir} (epoch={epoch}, global_step={global_step})"
    )
    return global_step, epoch


def evaluate_model(
    step: int,
    model: UNet2DModel,
    scheduler: CustomDDIMScheduler,
    num_samples: int,
    batch_size: int,
    accelerator: Accelerator,
    fixed_seed: int = 1234,
    save_dir: str = "artifacts/eval_images",
    reward_prompt: str = DEFAULT_REWARD_PROMPT,
    gender_threshold: float = 0.8,
    gender_weight: float = 2.0,
    reward_device: str = "cuda",
) -> None:
    """
    Deterministic, **portable**, parallelized evaluation.

    Every eval image is identified by a global index g ∈ [0, num_samples).
    Its noise stream (x_T + all per-step DDIM variance noise) comes from a
    CPU generator seeded (fixed_seed + g), so the set of eval images is:

      * independent of the number of GPUs — image identity is determined
        by g, not by which rank generates it.  Work is distributed across
        ranks via stride sharding (rank r generates indices
        {r, r+P, r+2P, …}) and the results are gathered and reordered
        into global-index order;
      * independent of GPU architecture — the RNG runs on CPU;
      * independent of the training RNG — private generators are used.

    Note: the UNet forward pass itself is not bit-exact across GPU
    families (different cuDNN kernels do slightly different floating-point
    reductions), so two runs on different architectures will produce
    near-identical, not pixel-identical, images from the same seed.  The
    dominant randomness (the injected noise) is identical, which is what
    makes the images comparable.
    """
    os.makedirs(save_dir, exist_ok=True)

    was_training = model.training
    model.eval()

    with torch.no_grad():
        n_channels = model.config.in_channels
        image_size = model.config.sample_size
        num_processes = accelerator.num_processes
        rank = accelerator.process_index
        device = accelerator.device

        # ------------------------------------------------------------------
        # Stride sharding: rank r owns global indices {r, r+P, r+2P, …}.
        # This partitions [0, num_samples) exactly once across ranks with
        # counts differing by at most 1.  Each image g is seeded by
        # (fixed_seed + g) on CPU, so the mapping g → image is the same
        # regardless of which rank produces it or how many ranks exist.
        # ------------------------------------------------------------------
        owned_indices = list(range(rank, num_samples, num_processes))
        max_per_rank = math.ceil(num_samples / num_processes)

        metric_keys = [
            "ir_person",
            "sex_score",
            "sex_score_binary",
            "aesthetics_score",
        ]

        # --- Generate this rank's owned images in batches ----------------
        image_parts: list[torch.Tensor] = []
        reward_parts: list[torch.Tensor] = []
        metric_parts: dict[str, list[torch.Tensor]] = {k: [] for k in metric_keys}

        for b_start in range(0, len(owned_indices), batch_size):
            batch_indices = owned_indices[b_start : b_start + batch_size]
            samples = generate_eval_samples(
                model, scheduler, batch_indices, device, base_seed=fixed_seed
            )
            rewards, scores = reward_function(
                samples,
                prompt=reward_prompt,
                male_threshold=gender_threshold,
                gender_weight=gender_weight,
                device=reward_device,
            )
            image_parts.append(samples)
            reward_parts.append(rewards.to(device))
            for k in metric_keys:
                metric_parts[k].append(scores[k].to(device))

        # --- Concatenate (handle ranks that own zero images) -------------
        if image_parts:
            local_images = torch.cat(image_parts)
            local_rewards = torch.cat(reward_parts)
            local_metrics = {k: torch.cat(metric_parts[k]) for k in metric_keys}
        else:
            local_images = torch.empty(0, n_channels, image_size, image_size, device=device)
            local_rewards = torch.empty(0, device=device)
            local_metrics = {k: torch.empty(0, device=device) for k in metric_keys}

        # --- Pad to max_per_rank so every rank has equal-sized tensors ---
        pad = max_per_rank - local_images.size(0)
        if pad > 0:
            local_images = torch.cat(
                [
                    local_images,
                    torch.zeros(pad, n_channels, image_size, image_size, device=device),
                ]
            )
            local_rewards = torch.cat(
                [
                    local_rewards,
                    torch.zeros(pad, device=device),
                ]
            )
            for k in metric_keys:
                local_metrics[k] = torch.cat(
                    [
                        local_metrics[k],
                        torch.zeros(pad, device=device),
                    ]
                )

        # --- Gather across ranks and reorder to global-index order -------
        # After gather the tensor is laid out as
        #   [rank0_block | rank1_block | … | rankP-1_block]
        # where rank r's block has max_per_rank entries corresponding to
        # global indices [r, r+P, r+2P, …] (plus any padding at the end).
        #
        # Global index g was produced by rank (g % P) at local position
        # (g // P), so its gathered position is:
        #   (g % P) * max_per_rank + (g // P)
        gathered_images = accelerator.gather(local_images)
        gathered_rewards = accelerator.gather(local_rewards)
        gathered_metrics = {k: accelerator.gather(local_metrics[k]) for k in metric_keys}

        reorder = [
            (g % num_processes) * max_per_rank + (g // num_processes) for g in range(num_samples)
        ]
        all_images = gathered_images[reorder]
        all_rewards = gathered_rewards[reorder]

        metrics = {
            "eval/reward": all_rewards.mean().item(),
            "eval/reward_std": all_rewards.std(unbiased=False).item(),
        }
        for k in metric_keys:
            vals = gathered_metrics[k][reorder].float()
            metrics[f"eval/{k}"] = vals.mean().item()
            metrics[f"eval/{k}_std"] = vals.std().item()

        if accelerator.is_main_process:
            wandb_imgs = []
            for img_idx, img in enumerate(tensor_batch_to_pil_images(all_images)):
                img.save(
                    os.path.join(
                        save_dir,
                        f"step_{step:08d}_{img_idx:05d}.png",
                    )
                )
                wandb_imgs.append(wandb.Image(img))
            accelerator.log({"eval/samples": wandb_imgs, **metrics}, step=step)

    if was_training:
        model.train()


def parse_timesteps_weights(path: str, scheduler_timesteps: list[int]) -> dict[int, float]:
    with open(path) as f:
        weights_json = json.load(f)

    # Accept sparse mapping; check keys are subset
    weights = {}
    for k, v in weights_json.items():
        timestep = int(k)
        weight = abs(float(v))
        if timestep not in scheduler_timesteps:
            raise ValueError(
                f"Timestep {timestep} in weights file not found in scheduler timesteps."
            )
        weights[timestep] = weight

    for t in scheduler_timesteps:
        weights.setdefault(int(t), 0.0)

    return weights


if __name__ == "__main__":
    # Pre-parse to get --config, then load YAML defaults before full parse
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()

    yaml_defaults = {}
    if pre_args.config:
        with open(pre_args.config) as f:
            yaml_defaults = yaml.safe_load(f) or {}

    parser = argparse.ArgumentParser(description="Fine-tune diffusion model with RL")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    # Model
    parser.add_argument(
        "--model_id",
        type=str,
        default="google/ddpm-celebahq-256",
        help="HuggingFace model ID for the pretrained diffusion model",
    )

    # Trajectory collection
    parser.add_argument(
        "--num_denoising_steps",
        type=int,
        default=50,
        help="Number of DDIM denoising steps used to generate each trajectory",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Total trajectories collected per outer iteration across all GPUs",
    )
    parser.add_argument(
        "--local_batch_size",
        type=int,
        default=5,
        help="Trajectories generated per GPU per forward pass",
    )

    # Timestep sampling strategy
    parser.add_argument(
        "--timesteps_per_update",
        type=int,
        default=None,
        help="Timesteps drawn per optimizer step; None uses all steps (full-trajectory baseline)",
    )
    parser.add_argument(
        "--timestep_weights",
        type=str,
        default=None,
        help="Path to JSON file with per-timestep sampling weights; uniform if omitted",
    )

    # Training
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Total number of outer training iterations"
    )
    parser.add_argument(
        "--inner_epochs",
        type=int,
        default=2,
        help="PPO inner optimization passes over each collected batch",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="AdamW learning rate")
    parser.add_argument(
        "--clip_epsilon",
        type=float,
        default=1e-4,
        help="PPO clipping epsilon; kept tight to prevent catastrophic forgetting",
    )

    # Reward
    parser.add_argument(
        "--reward_prompt",
        type=str,
        default=DEFAULT_REWARD_PROMPT,
        help="Text prompt passed to ImageReward for scoring",
    )
    parser.add_argument(
        "--gender_threshold",
        type=float,
        default=0.8,
        help="Male-probability threshold for the binary gender reward term",
    )
    parser.add_argument(
        "--gender_weight",
        type=float,
        default=2.0,
        help="Weight on the gender term in the combined reward",
    )
    parser.add_argument(
        "--reward_device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for reward models (ImageReward, gender, aesthetics). "
        "Use 'cpu' on low-memory GPUs to avoid OOM.",
    )

    # Evaluation
    parser.add_argument(
        "--eval_every_steps", type=int, default=20, help="Run evaluation every N optimizer steps"
    )
    parser.add_argument(
        "--eval_samples", type=int, default=20, help="Total number of samples drawn per evaluation"
    )

    # Logging
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="tao-diffusion",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--log_grad_stats",
        action="store_true",
        help="Log per-timestep gradient mean/std/norm to W&B (adds hook overhead per backward)",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint_every_epochs",
        type=int,
        default=10,
        help="Save a resumable checkpoint every N epochs (model + optimizer + RNG + "
        "global_step). Only the latest is kept.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="artifacts/checkpoints/latest",
        help="Directory for the single kept checkpoint (overwritten each save).",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint directory to resume from (restores model, optimizer, "
        "RNG, global_step, and epoch). Skips the pre-training eval.",
    )

    parser.set_defaults(**yaml_defaults)
    args = parser.parse_args()

    if args.timesteps_per_update is not None and args.timesteps_per_update < 1:
        parser.error("--timesteps_per_update must be >= 1")
    if args.timestep_weights is not None and args.timesteps_per_update is None:
        parser.error("--timestep_weights requires --timesteps_per_update")

    logger = get_logger(__name__, log_level="INFO")
    logging.basicConfig(level=logging.INFO)

    # One grad-accum step per sampled timestep. The full-trajectory baseline
    # excludes t=0 (near-zero noise variance → degenerate log-prob), hence -1.
    if args.timesteps_per_update is not None:
        gradient_accumulation_steps = args.timesteps_per_update
    else:
        gradient_accumulation_steps = args.num_denoising_steps - 1

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps, log_with="wandb"
    )
    device = accelerator.device
    logger.info(f"Using device: {device}")

    accelerator.init_trackers(project_name=args.wandb_project, config=vars(args))
    if accelerator.is_main_process:
        wandb.run.log_code(
            root=".",
            include_fn=lambda p: (
                p.startswith(os.path.abspath("tao_diffusion") + "/")
                or p == os.path.abspath(__file__)
            ),
        )

    num_batches_per_gpu = math.ceil(
        args.num_samples / (args.local_batch_size * accelerator.num_processes)
    )

    logger.info(f"Loading {args.model_id} ...")
    scheduler = CustomDDIMScheduler.from_pretrained(args.model_id, use_safetensors=True)
    pretrained_model = UNet2DModel.from_pretrained(args.model_id).to(device)

    optimizer = torch.optim.AdamW(pretrained_model.parameters(), lr=args.learning_rate)
    pretrained_model, optimizer = accelerator.prepare(pretrained_model, optimizer)

    scheduler.set_timesteps(args.num_denoising_steps, device=device)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)

    # Timestep sampling weights (p_φ(t) in the paper)
    if args.timesteps_per_update is not None:
        if args.timestep_weights is not None:
            timestep_weights = parse_timesteps_weights(
                args.timestep_weights, scheduler.timesteps.tolist()
            )
            logger.info(
                f"Timestep sampling: {args.timesteps_per_update} steps/update "
                f"from {args.timestep_weights}"
            )
        else:
            timestep_weights = {timestep: 1.0 for timestep in scheduler.timesteps.tolist()}
            logger.info(
                f"Timestep sampling: {args.timesteps_per_update} steps/update, uniform weights"
            )
        if args.timestep_weights is not None and timestep_weights.get(0, 0.0) != 0.0:
            logger.warning(
                f"t=0 sampling weight ({timestep_weights[0]:.4f}) overridden to 0 — "
                "the final denoising step has degenerate log-probability; excluded from training"
            )
        timestep_weights[0] = 0.0  # t=0: final denoising step, log-prob is degenerate

    if accelerator.is_main_process and args.timestep_weights:
        art = wandb.Artifact("timestep-weights", type="config")
        art.add_file(args.timestep_weights, name="timestep_weights.json")
        wandb.log_artifact(art)

    global_step = 0
    start_epoch = 0
    if args.resume_from_checkpoint:
        global_step, start_epoch = load_checkpoint(accelerator, args.resume_from_checkpoint)

    if start_epoch == 0:
        logger.info("Evaluating model before fine-tuning")
        evaluate_model(
            step=global_step,
            model=accelerator.unwrap_model(pretrained_model),
            scheduler=scheduler,
            num_samples=args.eval_samples,
            batch_size=args.local_batch_size,
            accelerator=accelerator,
            reward_prompt=args.reward_prompt,
            gender_threshold=args.gender_threshold,
            gender_weight=args.gender_weight,
            reward_device=args.reward_device,
        )

    for _epoch in tqdm(
        range(start_epoch, args.num_epochs),
        desc="Training Epochs",
        disable=not accelerator.is_main_process,
    ):
        # Collect trajectories: sample x_T → x_0, record latents/log_probs/rewards
        batches = []
        all_rewards = []
        all_metrics = {
            "ir_person": [],
            "sex_score": [],
            "sex_score_binary": [],
            "aesthetics_score": [],
        }

        for _ in tqdm(
            range(num_batches_per_gpu),
            desc=f"Generating batches of size {args.local_batch_size}",
            disable=not accelerator.is_main_process,
        ):
            latents, next_latents, log_probs, timesteps = generate_batch(
                accelerator.unwrap_model(pretrained_model), scheduler, args.local_batch_size, device
            )

            rewards, scores = reward_function(
                next_latents[:, -1],
                prompt=args.reward_prompt,
                male_threshold=args.gender_threshold,
                gender_weight=args.gender_weight,
                device=args.reward_device,
            )
            rewards = rewards.to(device)

            all_rewards.append(accelerator.gather(rewards))
            for k in all_metrics:
                all_metrics[k].append(accelerator.gather(scores[k].to(device)).cpu().flatten())

            batches.append((latents, next_latents, log_probs, timesteps, rewards))

        # Aggregate rewards across batches and log
        all_rewards = torch.cat(all_rewards)
        global_mean = all_rewards.mean()
        global_std = all_rewards.std(unbiased=False).clamp(min=1e-8)

        if accelerator.is_main_process:
            logger.info(f"[step {global_step}] avg reward: {global_mean.item():.4f}")

            metrics_to_log = {
                "train/reward": global_mean.item(),
                "train/reward_std": global_std.item(),
            }
            for k in all_metrics:
                all_values = torch.cat(all_metrics[k])
                metrics_to_log[f"train/{k}"] = all_values.float().mean().item()
                metrics_to_log[f"train/{k}_std"] = all_values.float().std().item()
            accelerator.log(metrics_to_log, step=global_step)

        # PPO inner optimization over the collected trajectories
        for _inner_epoch in range(args.inner_epochs):
            if args.timesteps_per_update is None:  # Full-trajectory baseline: all steps except t=0
                train_timesteps = [t for t in scheduler.timesteps.tolist() if t != 0]
            else:
                # Each device independently draws t ~ p_φ(t) with replacement
                ts, weights = zip(*timestep_weights.items(), strict=False)
                weights_tensor = torch.tensor(weights)
                sampled_indices = torch.multinomial(
                    weights_tensor, args.timesteps_per_update, replacement=True
                )
                train_timesteps = [ts[i] for i in sampled_indices.tolist()]

            # Gather across devices for logging only
            timesteps_tensor = torch.tensor(train_timesteps, device=device)
            all_timesteps = accelerator.gather(timesteps_tensor)
            if accelerator.is_main_process:
                all_ts = all_timesteps.cpu().tolist()
                timestep_counts = {int(t): 0 for t in scheduler.timesteps.cpu().tolist()}
                for t in all_ts:
                    timestep_counts[int(t)] += 1
                accelerator.log(
                    {f"timesteps/t={t}": count for t, count in timestep_counts.items()},
                    step=global_step,
                )

            for batch in batches:
                latents, next_latents, log_probs, timesteps, rewards = batch
                t_to_idx = {
                    int(t.item()): idx for idx, t in enumerate(timesteps)
                }  # timestep value → trajectory index

                # Advantage = normalized reward
                advantages = (rewards - global_mean) / global_std

                for t in train_timesteps:
                    t_idx = t_to_idx[t]
                    with accelerator.accumulate(pretrained_model):
                        lat_gpu = latents[:, t_idx].to(device, non_blocking=True)
                        nxt_gpu = next_latents[:, t_idx].to(device, non_blocking=True)
                        t_gpu = timesteps[t_idx].to(device, non_blocking=True)
                        new_log_probs = rescore_batch(
                            pretrained_model, scheduler, lat_gpu, nxt_gpu, t_gpu
                        )

                        # Importance ratio r_t = π_θ / π_θ_old
                        importance_ratio = torch.exp(new_log_probs - log_probs[:, t_idx].to(device))

                        # PPO-clip objective (ε = clip_epsilon, kept tight for diffusion policies)
                        clipped_ratio = torch.clamp(
                            importance_ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon
                        )
                        loss = -torch.min(
                            importance_ratio * advantages, clipped_ratio * advantages
                        ).mean()

                        if args.log_grad_stats:
                            with capture_grad_moments(pretrained_model, accelerator) as grad_stats:
                                accelerator.backward(loss)
                            if accelerator.is_main_process:
                                grad_norm = grad_stats["std"] * math.sqrt(grad_stats["N"])
                                accelerator.log(
                                    {
                                        f"grad_inc_norm/t={t}": grad_norm,
                                        f"grad_inc_mean/t={t}": grad_stats["mean"],
                                        f"grad_inc_std/t={t}": grad_stats["std"],
                                    },
                                    step=global_step,
                                )
                        else:
                            accelerator.backward(loss)

                        del lat_gpu, nxt_gpu, t_gpu, loss
                        del new_log_probs, importance_ratio, clipped_ratio

                        # Optimizer steps once all timesteps for this batch have been accumulated
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                        if (
                            accelerator.sync_gradients
                        ):  # True only after the optimizer actually stepped
                            global_step += 1
                            if global_step % args.eval_every_steps == 0:
                                logger.info(f"Evaluating model at step {global_step}")
                                evaluate_model(
                                    step=global_step,
                                    model=accelerator.unwrap_model(pretrained_model),
                                    scheduler=scheduler,
                                    num_samples=args.eval_samples,
                                    batch_size=args.local_batch_size,
                                    accelerator=accelerator,
                                    reward_prompt=args.reward_prompt,
                                    gender_threshold=args.gender_threshold,
                                    gender_weight=args.gender_weight,
                                    reward_device=args.reward_device,
                                )

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                accelerator.wait_for_everyone()

        # Periodic resumable checkpoint at end of epoch (keep only the latest).
        # Runs once per outer epoch, after all inner epochs complete.
        if (
            args.checkpoint_every_epochs > 0 and (_epoch + 1) % args.checkpoint_every_epochs == 0
        ) or (_epoch + 1) == args.num_epochs:
            save_checkpoint(
                accelerator,
                args.checkpoint_dir,
                global_step=global_step,
                epoch=_epoch + 1,
            )

    check_model_sync(pretrained_model, accelerator)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if accelerator.is_main_process:
        model_to_save = accelerator.unwrap_model(pretrained_model)
        model_dir = os.path.join("artifacts", "models", timestamp)
        model_to_save.save_pretrained(model_dir)

        args_dict = vars(args)
        with open(f"{model_dir}/training_args.json", "w") as f:
            json.dump(args_dict, f, indent=2)

        logger.info(f"Model and training arguments saved to {model_dir}")
