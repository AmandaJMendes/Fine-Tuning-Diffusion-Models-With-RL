# CelebA-HQ timestep weights

Timestep weight distributions used to train the CelebA-HQ model (reward column:
`reward`), sampled from the profiling data in `artifacts/timestep_profiles/`.
All files below were regenerated with `scripts/compute_weights.py` and verified
against the `timestep_weights.json` artifacts saved by the corresponding W&B
training runs (max |Δ| ≤ 5e-16).

## Files

| File | Metric | Subset | Regeneration command |
|---|---|---|---|
| `sensitivity_reward_n10_m10.json` | `sensitivity` | fixed 10-image / 10-reconstruction subset (the exact images used in training — see note below) | `compute_weights.py` + fixed ids |
| `variance_reward_n10_m10.json` | `variance` | fixed 10-image / 10-reconstruction subset (see note) | `compute_weights.py` + fixed ids |
| `snr_reward.json` | `snr` | full data (reward-agnostic; subset-independent) | `compute_weights.py --metric snr` |
| `snr_x_sensitivity_reward.json` | `snr_x_sensitivity` | full data (N=20, M=20) | `compute_weights.py --metric snr_x_sensitivity` |
| `snr_x_variance_reward.json` | `snr_x_variance` | full data (N=20, M=20) | `compute_weights.py --metric snr_x_variance` |

## Note on the `_n10_m10` subset

The sensitivity and variance weights were computed on the fixed 10-image /
10-reconstruction subset that was actually used in the experiment:

```python
IMAGE_IDS  = [0, 17, 15, 1, 13, 10, 6, 3, 18, 16]
SAMPLE_IDS = [19, 16, 15, 5, 4, 12, 14, 7, 3, 6]
```

(`compute_weights.py` does not currently expose a CLI flag for explicit ids; these
were passed to `subsample_data(image_ids=..., sample_ids=...)` during generation.)

The SNR and SNR×reward (hybrid) weights were computed on the **full** profiling
dataset (N=20 images, M=20 reconstructions); they have no `_n10_m10` suffix.
