# Flow-matching robot policy from scratch

A π0-style flow-matching action policy, written from scratch in NumPy (own MLP, backprop, Adam, sampler).
A C++ port (std, maybe Eigen, no ML libraries) comes in M3. Evaluated on PushT (`gym-pusht` +
`lerobot/pusht_keypoints`) against LeRobot's diffusion policy, then deployed on a real arm.
PyTorch is used only for gradient checks.

![FiLM policy on PushT](docs/media/m2_film_noise4_late.gif)

*Best M2 policy (FiLM + obs noise + late-t sampling): 69.2 % ± 2.3 success (mean ± std over 3 training seeds
× 3 sampling-noise draws, 500 test episodes each). These 4 episodes (test seeds 200000-200003) all succeed; about 30 % of episodes do not.*

## Milestones
| | Milestone | Status |
|---|---|---|
| M0 | Setup: repo, env, inspect PushT env and dataset | done |
| M1 | 2D flow-matching toy in NumPy (own MLP, backprop, Adam) | done |
| M2 | Flow-matching policy on PushT keypoints (NumPy), compared with LeRobot's diffusion policy | done: 69.2 % vs LeRobot 74.7 % on the same seeds |
| M3 | C++ port (std, maybe Eigen, no ML libraries) | inference done: matches NumPy, 2.4 ms per decision on 1 laptop core; Pi 5 next |
| M4 | Vision: policy from images | |
| M5 | Real robot arm | |
| M6 | Ship: write-up, demo, release | |

## Results
PushT, keypoints + agent position. Success = max coverage > 95 % within 300 steps. **Both policies run in the same
env, same eval code, same 500 test seeds (200000-200499).**

| Policy | Success rate | Avg max reward | Params | Time per action chunk (batch 1) |
|---|---|---|---|---|
| **Flow matching, FiLM residual MLP (ours)** | **69.2 % ± 2.3** (3 training seeds × 3 noise draws) | 0.936 | 2.4 M | **2.4 ms** (NumPy, CPU) / **2.4 ms** (C++, 1 core, -O3 -march=native); 10 Euler steps |
| LeRobot diffusion policy, `lerobot/diffusion_pusht_keypoints` (re-measured) | 74.7 % ± 1.0 (3 noise draws: 75.6 / 75.0 / 73.6) | 0.966 | 248.8 M | 242 ms (RTX 3050 GPU), 623 ms (CPU); 10 DDIM steps |
| Same policy in C++ (ours), closed loop | 65.4 % (1 draw; NumPy draws of the same checkpoint: 70.0 / 67.0 / 67.6 / 66.2) | 0.940 | 2.4 M | see M3 below |

- **Close to LeRobot's diffusion policy (about 5.5 points behind) with 100× fewer parameters.** Per decision, our CPU
  policy is **~100× faster than their GPU inference** (2.4 ms vs 242 ms) and **~260× faster CPU-to-CPU** (2.4 ms vs
  623 ms). From scratch in NumPy, trained on a laptop CPU in ~50 min.
- ± = std over runs (ours: 9 = 3 training seeds × 3 sampling-noise draws; LeRobot: 3 noise draws). A single
  500-episode eval also has a standard error of about 2 points, sqrt(p(1-p)/500).
- LeRobot's card reports 71.0 % on seeds 100000-100499 (2024). Re-run today on those seeds it gives 76.6 % ± 1.9, but
  the same seed no longer gives the same start layout (gym-pusht's reset RNG changed; per-seed results are
  uncorrelated), so the published 71.0 % is not comparable seed by seed. Only the same-env numbers above are.
  Details in `docs/BASELINES.md`.

## M2: flow-matching policy on PushT
**Setup**
- Observation: 2 frames × (16 T-keypoint coordinates + agent x, y) = 36-D, normalised to [-1, 1] with min/max from
  the training episodes only (185 train / 21 val episodes).
- Action: chunk of 16 target positions (16 × 2). Execute the first 8, then replan.
- Flow: linear path from noise (t=0) to the action chunk (t=1), loss = MSE on the velocity, 10 Euler steps to sample.
- Training: Adam, lr 1e-3 with warm-up and cosine decay, batch 256, 50k steps, EMA 0.999 of the weights.
- Checkpoints are chosen on eval seeds 100000–100049; reported numbers use separate test seeds 200000–200499.

**What made the difference: conditioning, not size** (500 test episodes each, ± = 1 s.e.)

| Network | Params | Success | Avg max reward | Best-of-8 action error, train / val |
|---|---|---|---|---|
| Concat MLP 512×3 | 0.59 M | 8.2 % ± 1.2 | 0.744 | 11.6 / 15.0 px |
| Concat MLP 1024×3 | 2.2 M | 21.0 % ± 1.8 | 0.844 | 9.7 / 14.5 px |
| **FiLM residual MLP 512×3** | 2.4 M | **49.6 % ± 2.2** | **0.897** | 6.0 / 16.6 px |

- Concat MLP: the observation and time embedding are just concatenated to the noisy actions at the input.
- FiLM: the observation and time go through a small encoder that gives a scale and shift to every residual block,
  as in Diffusion Policy's U-Net. At the same size it more than doubles the success rate.
- Ruled out first: more Euler steps (30) or replanning more often (execute 4) did not help, so the problem was the
  model, not sampling.
- Best-of-8 error: about half of the one-sample error comes from the model picking a different (valid) pushing mode.
- FiLM overfits: its val loss rises from 0.087 (9k steps) to 0.19 (50k) while train loss keeps falling.

![training curves](docs/media/m2_runs.png)

**From 49.6 % to 70.0 %: regularisation and where the flow trains** (FiLM 512×3, 500 test episodes, ± 2.0-2.2 s.e.)

| Change (one at a time) | Success, best ckpt | Success, 50k ckpt | Avg max reward (best) |
|---|---|---|---|
| FiLM | 49.6 % | 47.4 % | 0.897 |
| + weight decay 0.05 | 40.6 % | 49.0 % | 0.925 |
| + obs noise 2 px | 58.4 % | 58.0 % | 0.900 |
| + obs noise 4 px | 60.0 % | 64.6 % | 0.912 |
| + obs noise 2 px + late t | 65.4 % | 63.0 % | 0.951 |
| same, training seed 1 | 61.6 % | 62.6 % | 0.952 |
| **+ obs noise 4 px + late t** | **70.0 %** | **70.0 %** | 0.946 |
| same, training seed 1 | 68.4 % | 68.4 % | 0.943 |
| same, training seed 2 | 72.4 % | 71.4 % | 0.937 |

- Obs noise: Gaussian noise (in pixels) on the 36 observation values during training. +10 points.
- Late t: train the flow more near t=1, the data end, where fine alignment is decided (t = sqrt(u), density 2t).
  +5-7 points, and the highest coverage.
- Weight decay: no effect.
- Seed-to-seed spread: about 2-4 points (std 2.0 over the 3 seeds of the final config).
- Val loss rises in every run, even when closed-loop success improves: on this task it is a poor checkpoint selector.
- Failures of the 70 % policy: 116 of 150 are near misses (max coverage 0.90-0.95), 27 never pass 0.5.

![training curves, all FiLM runs](docs/media/m2_runs_all.png)

## M3: C++ port (in progress)
Inference port of the FiLM policy, C++17, std only. Weight format: `cpp/weights/FORMAT.md` (weights stored (in, out)).

- Matches NumPy on every intermediate (`test_vectors`, allclose atol 1e-5 + rtol 1e-5): velocity max abs error
  1.5e-5, final actions 7.6e-5 px. Mutation tests (wrong SiLU sign, `scale` instead of `1 + scale`, reversed time grid)
  all fail the checks, so the checks have teeth.
- Closed loop in gym-pusht (C++ does all inference behind a pipe): 65.4 %, within the NumPy policy's
  sampling-noise spread (66.2-70.0 %).
- Batch-1 latency per decision (`bench`, median of 1000, 1 core of an i5-12500H), no heap allocation in the Euler loop:

| Euler steps | Success (NumPy, 500 test eps) | C++ `-O2` | C++ `-O3 -march=native` (AVX2) |
|---|---|---|---|
| 1 | 7.0 % | 0.88 ms | 0.24 ms |
| 2 | 45.0 % | 1.79 ms | 0.47 ms |
| 5 | 63.8 % | 4.39 ms | 1.31 ms |
| 10 | 70.0 % | 8.81 ms | 2.41 ms |

One Euler step predicts the average of the demonstrated modes, which pushes nowhere useful; 5-10 steps are needed.
The weights file (`cpp/weights/film_g.bin`, 9.6 MB) is not in git. It is on the Hugging Face Hub
([`Kimaa6/flowpolicy-pusht-film`](https://huggingface.co/Kimaa6/flowpolicy-pusht-film), private for now):
```bash
huggingface-cli download Kimaa6/flowpolicy-pusht-film film_g.bin --local-dir cpp/weights
```
Or regenerate it from a trained checkpoint:
```bash
PYTHONPATH=python python python/scripts/m3_export.py          # -> cpp/weights/film_g.bin + cpp/tests/vectors/vectors.bin
cmake -S cpp -B cpp/build && cmake --build cpp/build
./cpp/build/test_vectors                                        # C++ vs NumPy, every intermediate
cmake -S cpp -B cpp/build-native -DFP_NATIVE=ON && cmake --build cpp/build-native && ./cpp/build-native/bench
PYTHONPATH=python python python/scripts/m3_eval_cpp.py --episodes 500   # closed loop, C++ policy
PYTHONPATH=python python python/scripts/m3_check_loader.py    # C++ loader vs NumPy, bitwise
```

## PushT env notes (checked in `python/scripts/m2_inspect.py`)
- `obs_type="environment_state_agent_pos"` matches `lerobot/pusht_keypoints` exactly: same 16 keypoint order,
  pixel units in [0, 512]; max error 2.7e-5 px over all 25,650 frames (swapping the T's two rectangles gives 81 px),
  and open-loop replay of recorded actions reproduces whole episodes.
- Gotcha 1: `reset_to_state` sets the block position *before* the angle, and pymunk rotates about the centre of
  gravity (0, 45), so the block origin moves by `cog - R(theta) cog`. Pre-compensate to reset to a dataset frame.
- Gotcha 2: the dataset's `next.reward` is not the env's reward and `next.success` is never True, although replays
  reach env success. Score success and coverage only from the env's `info`.

## Layout
```
python/flowmatch/   NumPy code: mlp.py, film.py, adam.py, flow.py (M1), policy.py, data.py, eval_pusht.py
python/scripts/     m0_inspect, m1_toy2d, m2_inspect, m2_data_check, m2_train, m2_eval, m2_offline_eval, m2_plot_runs
python/tests/       gradient checks vs torch and finite differences, data pipeline tests
cpp/                C++ port (M3)
notebooks/          Colab/Kaggle notebooks
docs/PROGRESS.md    one line per session; docs/BASELINES.md: published numbers with sources
```

## Setup and commands
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt   # dev = torch (grad checks only) + pytest
export PYTHONPATH=python                                   # replaces any ROS PYTHONPATH
pytest -q

python python/scripts/m1_toy2d.py --data moons                                   # M1 toy
python python/scripts/m2_train.py --run film_noise4_late --arch film --steps 50000 \
    --obs-noise 4 --t-sampling late                                                # best M2 config (CPU, ~50 min)
python python/scripts/m2_eval.py --ckpt checkpoints/film_noise4_late/best.npz --episodes 500   # test seeds
python python/scripts/m2_offline_eval.py --ckpt checkpoints/film_noise4_late/best.npz
```

## M1: 2D flow-matching toy (10k steps, batch 512, MLP 3x128 SiLU, 50 Euler steps)
Sliced Wasserstein distance to data (lower is better; floor = data vs fresh data).

| Dataset | SW samples | floor | noise | train time (CPU) |
|---|---|---|---|---|
| moons | 0.054 | 0.046 | 0.421 | 21 s |
| checkerboard | 0.027 | 0.021 | 0.207 | 42 s |
| 8 gaussians | 0.024 | 0.017 | 0.505 | 23 s |

Checkerboard at 30k steps: SW 0.028, squares visibly sharper; SW misses this.

Convention: t=0 is noise, t=1 is data (π0 uses the opposite).

![moons](docs/media/m1_moons.png)
