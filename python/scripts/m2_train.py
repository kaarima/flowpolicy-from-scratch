"""M2: train the flow-matching policy on lerobot/pusht_keypoints (NumPy, CPU).

Logs train loss, val loss (fixed noise, so runs are comparable) and, every --eval-every steps, a quick
closed-loop eval in gym-pusht with the EMA weights. Saves checkpoints/<run>/{last,best}.npz + stats.json + log.csv.

Run from the repo root:
  PYTHONPATH=python python python/scripts/m2_train.py --run base --steps 50000
"""
import argparse
import csv
import shutil
import time
from pathlib import Path

import numpy as np

from flowmatch.adam import Adam
from flowmatch.data import build_datasets
from flowmatch.eval_pusht import evaluate
from flowmatch.policy import EMA, FlowPolicy

ROOT = Path(__file__).resolve().parents[2]


def val_loss(policy, val, n=2048, seed=123):
    """Mean flow-matching loss on the first n val windows, same noise and t every time."""
    rng = np.random.default_rng(seed)
    o, a = val.get(np.arange(min(n, len(val))))
    return policy.loss_and_grads(o, a, rng)[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="base")
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--n-freqs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=10000)
    p.add_argument("--eval-episodes", type=int, default=50)
    args = p.parse_args()

    out = ROOT / "checkpoints" / args.run
    out.mkdir(parents=True, exist_ok=True)
    train, val, obs_norm, act_norm = build_datasets(val_frac=0.1, seed=0, stats_path=out / "stats.json")
    print(f"train windows {len(train)}, val windows {len(val)}")

    rng = np.random.default_rng(args.seed)
    policy = FlowPolicy(args.hidden, args.layers, args.n_freqs, rng)
    n_params = sum(v.size for v in policy.mlp.params.values())
    print(f"policy MLP {policy.mlp.n_layers} linear layers, hidden {args.hidden}, {n_params / 1e6:.2f} M params")
    opt = Adam(policy.mlp.params, lr=args.lr)
    ema = EMA(policy.mlp.params, args.ema)
    ema_policy = FlowPolicy(args.hidden, args.layers, args.n_freqs)
    ema_policy.mlp.params = ema.params  # shares the arrays: always the current EMA weights

    log_f = open(out / "log.csv", "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["step", "train_loss", "val_loss", "val_loss_ema", "success_rate", "avg_max_reward", "time_s"])
    best, losses, t0 = -1.0, [], time.time()
    for step in range(1, args.steps + 1):
        warm = min(1.0, step / args.warmup)
        cos = 0.5 * (1 + np.cos(np.pi * step / args.steps))
        opt.lr = args.lr * warm * (0.02 + 0.98 * cos)
        o, a = train.sample(args.batch, rng)
        loss, grads = policy.loss_and_grads(o, a, rng)
        opt.step(grads)
        ema.update(policy.mlp.params)
        losses.append(loss)

        if step % args.log_every == 0 or step == args.steps:
            vl, vle = val_loss(policy, val), val_loss(ema_policy, val)
            sr = amr = ""
            if step % args.eval_every == 0 or step == args.steps:
                r = evaluate(ema_policy, obs_norm, act_norm, n_episodes=args.eval_episodes)
                sr, amr = r["success_rate"], r["avg_max_reward"]
                if sr > best:
                    best = sr
                    ema_policy.save(out / "best.npz", extra={"step": step, "success_rate": sr})
            el = time.time() - t0
            print(f"step {step:6d}  train {np.mean(losses[-args.log_every:]):.4f}  val {vl:.4f}  val_ema {vle:.4f}"
                  + (f"  success {sr:.2f}  avg_max_reward {amr:.3f}" if sr != "" else "") + f"  {el:.0f}s",
                  flush=True)
            log.writerow([step, np.mean(losses[-args.log_every:]), vl, vle, sr, amr, round(el)])
            log_f.flush()
            ema_policy.save(out / "last.npz", extra={"step": step})
    log_f.close()
    shutil.copy(out / "stats.json", ROOT / "data" / "pusht_keypoints_stats.json")
    print("done:", out)


if __name__ == "__main__":
    main()
