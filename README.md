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
