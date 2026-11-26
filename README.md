# Flow-matching robot policy from scratch

A π0-style flow-matching action policy written from scratch: NumPy prototype first, then a C++ port
(std, maybe Eigen, no ML libraries). Evaluated on PushT (`gym-pusht` + `lerobot/pusht`), compared with
LeRobot's diffusion policy, then deployed on a real arm. PyTorch is used only for gradient checks.

## Milestones
| | Milestone | Status |
|---|---|---|
| M0 | Setup: repo, env, inspect PushT env and dataset | in progress |
| M1 | 2D flow-matching toy in NumPy (own MLP, backprop, Adam) | |
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
```
