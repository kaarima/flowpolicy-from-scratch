"""Compute per decision: time to produce one action chunk (batch 1, as on a robot) for our NumPy flow policy
and LeRobot's diffusion policy (lerobot/diffusion_pusht_keypoints). Median over --reps calls, after warm-up.
Run on an otherwise idle machine. Needs the gardener-train env (LeRobot) and gym_pusht links, like
lerobot_baseline_eval.py:
  PYTHONPATH=<links>:python ~/miniconda3/envs/gardener-train/bin/python python/scripts/speed_benchmark.py
"""
import argparse
import time

import numpy as np
import torch

from flowmatch.policy import FlowPolicy


def median_ms(fn, reps, sync=None):
    for _ in range(5):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        if sync:
            sync()
        ts.append(time.perf_counter() - t)
    return 1000 * float(np.median(ts))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/film_noise4_late/best.npz")
    p.add_argument("--reps", type=int, default=100)
    args = p.parse_args()

    ours, _ = FlowPolicy.load(args.ckpt)
    n_ours = sum(v.size for v in ours.net.params.values())
    obs = np.zeros((1, 36), np.float32)
    rng = np.random.default_rng(0)
    t_ours = median_ms(lambda: ours.sample(obs, 10, rng), args.reps)
    print(f"ours   : {n_ours / 1e6:6.2f} M params, 10 Euler steps, NumPy CPU          : {t_ours:7.2f} ms / chunk (16 actions)")

    from lerobot_baseline_eval import LeRobotAdapter  # same folder

    raw = np.full((1, 36), 256.0, np.float32)
    for dev in (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]:
        lr = LeRobotAdapter(dev)
        n_lr = sum(v.numel() for v in lr.policy.parameters())
        sync = torch.cuda.synchronize if dev == "cuda" else None
        t = median_ms(lambda: lr.sample(raw), args.reps if dev == "cuda" else max(10, args.reps // 10), sync)
        print(f"LeRobot: {n_lr / 1e6:6.1f} M params, 10 DDIM steps, torch {dev:4s}           : {t:7.2f} ms / chunk (8 actions)")
        del lr
        if dev == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
