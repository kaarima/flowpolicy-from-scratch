"""M3 bookkeeping: success over training seeds x sampling-noise draws, and success vs Euler steps.

Part 1: 3 training seeds (film_noise4_late, _s1, _s2; best.npz) x 3 sampling rng draws -> 9 evals, mean +- std.
Part 2: G seed 0, Euler steps 1 / 2 / 5 / 10 (rng 0), success on the same 500 test seeds.
NumPy policy (the C++ port matches it, see m3_eval_cpp.py). Results -> checkpoints/m3_seed_noise_evals.json.

Run from the repo root:
  PYTHONPATH=python python python/scripts/m3_seed_noise_evals.py
"""
import json
from pathlib import Path

import numpy as np

from flowmatch.data import load_stats
from flowmatch.eval_pusht import evaluate
from flowmatch.policy import FlowPolicy

ROOT = Path(__file__).resolve().parents[2]
RUNS = ["film_noise4_late", "film_noise4_late_s1", "film_noise4_late_s2"]


def run(run, rng_seed, steps=10):
    c = ROOT / "checkpoints" / run / "best.npz"
    pol, _ = FlowPolicy.load(c)
    on, an = load_stats(c.parent / "stats.json")
    r = evaluate(pol, on, an, n_episodes=500, seed0=200000, n_flow_steps=steps, rng=np.random.default_rng(rng_seed))
    print(f"{run:22s} rng {rng_seed}  euler {steps:2d}: success {r['success_rate']:.3f}  "
          f"avg max reward {r['avg_max_reward']:.3f}", flush=True)
    return r["success_rate"], r["avg_max_reward"]


def main():
    out = {"seed_x_noise": {}, "euler_steps": {}}
    for run_ in RUNS:
        for s in range(3):
            out["seed_x_noise"][f"{run_}/rng{s}"] = run(run_, s)
    sr = np.array([v[0] for v in out["seed_x_noise"].values()])
    print(f"9 evals: success {100 * sr.mean():.1f} +- {100 * sr.std(ddof=1):.1f} (std), "
          f"min {100 * sr.min():.1f}, max {100 * sr.max():.1f}")
    for steps in [1, 2, 5, 10]:
        out["euler_steps"][steps] = run(RUNS[0], 0, steps)
    json.dump(out, open(ROOT / "checkpoints" / "m3_seed_noise_evals.json", "w"), indent=1)


if __name__ == "__main__":
    main()
