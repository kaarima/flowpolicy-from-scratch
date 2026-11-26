"""M2 diagnostic: offline action error of a checkpoint on train vs val windows.

For N windows: sample a chunk from the policy (from the demo's obs), unnormalise, compare with the demo chunk
in pixels. Reports error of action 0, mean over the 8 executed actions, and over all 16.
  train ~ val, both high -> the model cannot fit (underfitting / conditioning)
  train << val          -> overfitting (only 185 train demos)
Reference: "hold" = predict the current agent position for every step (no learning at all).
Demos are multimodal, so one sample vs one demo inflates the error even for a perfect model. So K samples are
drawn per obs and both the mean over samples and the min over samples (best of K) are reported:
  min << mean      -> the model knows the modes; the error is mode choice (fine)
  min still high   -> the model really underfits

Run from the repo root:
  PYTHONPATH=python python python/scripts/m2_offline_eval.py --ckpt checkpoints/base/best.npz
"""
import argparse
from pathlib import Path

import numpy as np

from flowmatch.data import build_datasets, load_stats
from flowmatch.policy import FlowPolicy


def errors(policy, ds, obs_norm, act_norm, n, rng, k=8):
    idx = rng.choice(len(ds), min(n, len(ds)), replace=False)
    o, a = ds.get(idx)
    demo = act_norm.unnormalize(a)                                      # (n, 16, 2)
    pred = act_norm.unnormalize(policy.sample(np.repeat(o, k, axis=0), 10, rng)).reshape(len(o), k, 16, 2)
    e = np.linalg.norm(pred - demo[:, None], axis=3)                    # (n, k, 16) pixels
    e8 = e[:, :, :8].mean(2)                                            # (n, k): executed part
    agent = obs_norm.unnormalize(o[:, 18:])[:, 16:18]                   # current agent position (frame t)
    h = np.linalg.norm(agent[:, None, :] - demo, axis=2)
    return e[:, :, 0].mean(), e8.mean(), e8.min(1).mean(), e.mean(), h[:, :8].mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--n", type=int, default=1000)
    args = p.parse_args()
    train, val, _, _ = build_datasets(val_frac=0.1, seed=0)
    print(f"{'checkpoint':36s} {'set':5s} {'act0 px':>8s} {'1-8 px':>8s} {'1-8 min8':>9s} {'1-16 px':>8s} {'hold 1-8':>9s}")
    for c in args.ckpt:
        policy, _ = FlowPolicy.load(c)
        obs_norm, act_norm = load_stats(Path(c).parent / "stats.json")
        for name, ds in [("train", train), ("val", val)]:
            r = errors(policy, ds, obs_norm, act_norm, args.n, np.random.default_rng(0))
            print(f"{c:36s} {name:5s} {r[0]:8.1f} {r[1]:8.1f} {r[2]:9.1f} {r[3]:8.1f} {r[4]:9.1f}")


if __name__ == "__main__":
    main()
