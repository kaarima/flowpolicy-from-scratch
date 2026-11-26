# Flow-matching robot policy from scratch

A π0-style flow-matching action policy written from scratch: NumPy prototype first, then a C++ port
(std, maybe Eigen, no ML libraries). Evaluated on PushT (`gym-pusht` + `lerobot/pusht`), compared with
LeRobot's diffusion policy, then deployed on a real arm. PyTorch is used only for gradient checks.

## Milestones
| | Milestone | Status |
|---|---|---|
| M0 | Setup: repo, env, inspect PushT env and dataset | done |
| M1 | 2D flow-matching toy in NumPy (own MLP, backprop, Adam) | done |
| M2 | Conditional flow-matching action policy on PushT state | |
| M3 | Action chunks, evaluation vs LeRobot diffusion policy | |
| M4 | C++ port (forward + training) | |
| M5 | Speed / numerical parity, C++ vs NumPy | |
| M6 | Deployment on a real arm | |

## Results
| Policy | PushT success rate | Mean coverage | Notes |
|---|---|---|---|
| Flow matching (NumPy) | – | – | |
| Flow matching (C++) | – | – | |
| LeRobot diffusion policy | – | – | |

## PushT env notes (checked in `python/scripts/m2_inspect.py`)
- `obs_type="environment_state_agent_pos"` matches `lerobot/pusht_keypoints` exactly: same 16 keypoint order,
  pixel units in [0, 512]; max error 2.7e-5 px over all 25,650 frames (swapping the T's two rectangles gives 81 px),
  and open-loop replay of recorded actions reproduces whole episodes.
- Gotcha 1: `reset_to_state` sets the block position *before* the angle, and pymunk rotates about the centre of
  gravity (0, 45), so the block origin moves by `cog - R(theta) cog`. Pre-compensate to reset to a dataset frame.
- Gotcha 2: the dataset's `next.reward` is not the env's reward and `next.success` is never True, although replays
  reach env success. Score success and coverage only from the env's `info`.

## Target
LeRobot diffusion policy on keypoints: **71.0 % success, 0.97 avg max overlap, 500 episodes**
([model card](https://huggingface.co/lerobot/diffusion_pusht_keypoints)). Caveats: eval seeds and checkpoint step
are not given, and the card says 50 episodes in its usage text but 500 in its results table. Details: `docs/BASELINES.md`.

## Layout
```
python/flowmatch/   NumPy prototype
python/scripts/     runnable scripts (m0_inspect.py, ...)
python/tests/       tests and gradient checks
cpp/                C++ port
notebooks/          Colab/Kaggle notebooks
docs/PROGRESS.md    one line per session
```

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # add -r requirements-dev.txt for grad checks
python python/scripts/m0_inspect.py
PYTHONPATH=python python python/scripts/m1_toy2d.py --data moons
pytest -q                                # gradient checks vs torch + finite differences
```

## M1 toy results (10k steps, batch 512, MLP 3x128 SiLU, 50 Euler steps)
Sliced Wasserstein distance to data (lower is better; floor = data vs fresh data).

| Dataset | SW samples | floor | noise | train time (CPU) |
|---|---|---|---|---|
| moons | 0.054 | 0.046 | 0.421 | 21 s |
| checkerboard | 0.027 | 0.021 | 0.207 | 42 s |
| 8 gaussians | 0.024 | 0.017 | 0.505 | 23 s |

Checkerboard at 30k steps: SW 0.028, squares visibly sharper; SW misses this.

Convention: t=0 is noise, t=1 is data (π0 uses the opposite).

![moons](docs/media/m1_moons.png)
