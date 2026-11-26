"""M2: evaluate a trained flow-matching policy in gym-pusht.

Episode i uses reset seed (seed0 + i), so runs are comparable. Saves results JSON next to the checkpoint and
a GIF of the first episodes to docs/media/m2_<run>.gif.

Run from the repo root:
  PYTHONPATH=python python python/scripts/m2_eval.py --ckpt checkpoints/base/best.npz --episodes 500
"""
import argparse
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from flowmatch.data import load_stats
from flowmatch.eval_pusht import evaluate
from flowmatch.policy import FlowPolicy

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--seed0", type=int, default=100000)
    p.add_argument("--flow-steps", type=int, default=10)
    p.add_argument("--n-exec", type=int, default=8)
    p.add_argument("--gif", type=int, default=4, help="episodes to record (0 = none)")
    args = p.parse_args()

    ckpt = Path(args.ckpt)
    policy, extra = FlowPolicy.load(ckpt)
    obs_norm, act_norm = load_stats(ckpt.parent / "stats.json")
    t0 = time.time()
    r = evaluate(policy, obs_norm, act_norm, n_episodes=args.episodes, seed0=args.seed0, n_exec=args.n_exec,
                 n_flow_steps=args.flow_steps, record=args.gif)
    n = args.episodes
    se = np.sqrt(r["success_rate"] * (1 - r["success_rate"]) / n)
    print(f"{ckpt} (step {extra.get('step')}): {n} episodes, seeds {args.seed0}..{args.seed0 + n - 1}, "
          f"{args.flow_steps} flow steps, exec {args.n_exec}")
    print(f"success rate {r['success_rate']:.3f} +- {se:.3f} (1 s.e.)   avg max reward {r['avg_max_reward']:.3f}"
          f"   avg max coverage {r['avg_max_coverage']:.3f}   mean length {r['length'].mean():.0f}   "
          f"{time.time() - t0:.0f}s")

    res = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items() if k != "frames"}
    res.update({"ckpt": str(ckpt), "ckpt_step": extra.get("step"), "flow_steps": args.flow_steps,
                "n_exec": args.n_exec})
    with open(ckpt.with_name(f"{ckpt.stem}_eval{n}.json"), "w") as f:
        json.dump(res, f)
    if args.gif:
        frames = [fr[::2] for fr in r["frames"]]  # 5 fps
        h = min(len(f) for f in frames)
        frames = [np.concatenate([f[min(i, len(f) - 1)] for f in frames], axis=1) for i in range(max(map(len, frames)))]
        gif = ROOT / "docs" / "media" / f"m2_{ckpt.parent.name}.gif"
        imageio.mimsave(gif, [f[::2, ::2] for f in frames], fps=5, loop=0)
        print("saved", gif, f"(shortest episode {h} frames)")


if __name__ == "__main__":
    main()
