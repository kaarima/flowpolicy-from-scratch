# PushT baselines

All numbers below were read on 2026-11-26 from the linked pages. Nothing here is re-measured by us yet.

## Metrics (gym-pusht)
- `coverage` = area(T ∩ goal) / area(goal), in `info["coverage"]`.
- `reward` = clip(coverage / 0.95, 0, 1). `info["is_success"]` = coverage > 0.95; the episode terminates on success.
- Episode limit: 300 steps at 10 Hz (`max_episode_steps=300`, `control_hz=10`).
- LeRobot model cards report, per episode: **max overlap** (`eval/avg_max_reward`, in [0, 1]) and **success** =
  "whether or not the maximum overlap is at least 95%".
- The diffusion policy paper reports **target area coverage** for Push-T, not a success rate.

## LeRobot diffusion policy (Hugging Face model cards)

| Model | Obs | Train data | Success rate | Avg max overlap | Eval episodes | Source |
|---|---|---|---|---|---|---|
| `lerobot/diffusion_pusht` (175k-step checkpoint) | image + agent pos | `lerobot/pusht` | **65.4 %** | 0.955 | 500 | [card](https://huggingface.co/lerobot/diffusion_pusht) ([raw README](https://huggingface.co/lerobot/diffusion_pusht/raw/main/README.md)) |
| same card, "Theirs" (original DP repo weights, evaluated in LeRobot) | image + agent pos | – | 64.2 % | 0.957 | 500 | same |
| `lerobot/diffusion_pusht_keypoints` | keypoints + agent pos | `lerobot/pusht_keypoints` | **71.0 %** | 0.97 | 500 (table header) | [card](https://huggingface.co/lerobot/diffusion_pusht_keypoints) ([raw README](https://huggingface.co/lerobot/diffusion_pusht_keypoints/raw/main/README.md)) |

Notes:
- Neither card gives eval seeds. The keypoints card gives no checkpoint step for its number, and its
  "how to get started" text says 50 episodes while the results table says 500.
- A web search snippet quoted 63.8 % / 0.959 for the image card. The raw README read directly says 65.4 / 0.955,
  so 63.8 is not used here.
- The keypoints card conditions the U-Net on the encoded observation (global conditioning), not inpainting.

**Our target for M2/M3: the keypoints row (71.0 % success, 0.97 max overlap, 500 episodes)**, since our
policy uses the same observations.

## Original Diffusion Policy paper (Chi et al., RSS 2023)
[arXiv 2303.04137](https://arxiv.org/abs/2303.04137), tables read from [the HTML v5](https://arxiv.org/html/2303.04137v5).
Push-T column, format "max performance / average of last 10 checkpoints", each "averaged across 3 training seeds
and 50 different environment initial conditions (150 in total)" (Table 1 caption). Metric for Push-T: target area coverage.

| Policy | State obs (Table 1) | Image obs (Table 2) |
|---|---|---|
| DiffusionPolicy-C (CNN) | 0.95 / 0.91 | 0.91 / 0.84 |
| DiffusionPolicy-T (Transformer) | 0.95 / 0.79 | 0.78 / 0.66 |

Push-T hyperparameters (Table 7, CNN): observation horizon To = 2, action horizon Ta = 8, prediction horizon Tp = 16,
100 diffusion iterations for training and inference. Same horizons as our M2 plan.

These paper numbers use a different protocol (coverage, best/last-10 checkpoints, 150 rollouts) from the LeRobot
cards (success at 95 %, 500 episodes): do not compare them directly.
