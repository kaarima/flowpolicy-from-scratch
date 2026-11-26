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
- Eval seeds: the keypoints repo's `eval_info.json` lists all 500 episodes with their seeds: **100000-100499**
  (pc_success 71.0, avg_max_reward 0.9689, run on 2024-07-05). Their inference settings (config.json): DDIM,
  10 denoising steps, n_obs_steps 2, horizon 16, n_action_steps 8; 248.8 M parameters.
- Overlap with our protocol: our checkpoint-selection seeds 100000-100049 are inside their eval seeds, so our policy
  is NOT compared on 100000-100499 (it would be slightly biased). The fair head-to-head is both policies on our test
  seeds 200000-200499, same env and metric (`python/scripts/lerobot_baseline_eval.py`). Their checkpoint on
  100000-100499 is only used to check that we reproduce their 71.0 % with today's gym-pusht.
- The image card gives no eval seeds. The keypoints card gives no checkpoint step for its number, and its
  "how to get started" text says 50 episodes while the results table says 500.
- A web search snippet quoted 63.8 % / 0.959 for the image card. The raw README read directly says 65.4 / 0.955,
  so 63.8 is not used here.
- The keypoints card conditions the U-Net on the encoded observation (global conditioning), not inpainting.

## Re-measured in our eval (2026-11-26)
`python/scripts/lerobot_baseline_eval.py`: their checkpoint, their inference settings (DDIM, 10 steps), loaded in
LeRobot 0.4.4 with a strict key check, run through our eval loop (gym-pusht 0.1.8, pymunk 6.11.1).

| Seeds | LeRobot DP keypoints | Our FiLM policy (3 training seeds) |
|---|---|---|
| 100000-100499 (their published seeds) | 76.6 % ± 1.9, avg max reward 0.962 (published: 71.0 %, 0.969) | not run: our selection seeds overlap |
| **200000-200499 (our test seeds)** | **75.6 % ± 1.9**, avg max reward 0.964 | **70.3 % ± 2.0** (70.0 / 68.4 / 72.4), 0.942 |

- Same checkpoint, same seeds, but 76.6 % today vs 71.0 % published: per-seed max reward is uncorrelated with their
  eval_info.json (r = 0.01; 92 episodes succeed only there, 120 only here). gym-pusht now resets with the env's own RNG,
  so a seed no longer gives the 2024 start layout. Use only same-env comparisons.
- Compute per action chunk, batch 1 (python/scripts/speed_benchmark.py): ours 2.4 ms (NumPy, CPU, 2.4 M params,
  10 Euler steps); LeRobot 242 ms on an RTX 3050 laptop GPU (24 ms per U-Net pass x 10), 623 ms on CPU (248.8 M params).

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
