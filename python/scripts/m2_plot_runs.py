"""Plot train/val loss and closed-loop eval of training runs from checkpoints/<run>/log.csv.

Run from the repo root:
  python python/scripts/m2_plot_runs.py base long150k wide1024 --out docs/media/m2_runs.png
"""
import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(run):
    with open(f"checkpoints/{run}/log.csv") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", default="docs/media/m2_runs.png")
    args = p.parse_args()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for i, run in enumerate(args.runs):
        rows, c = read(run), f"C{i}"
        s = [int(r["step"]) for r in rows]
        ax[0].plot(s, [float(r["train_loss"]) for r in rows], c=c, ls="--", label=f"{run} train")
        ax[0].plot(s, [float(r["val_loss_ema"]) for r in rows], c=c, label=f"{run} val (EMA)")
        ev = [r for r in rows if r["success_rate"]]
        es = [int(r["step"]) for r in ev]
        ax[1].plot(es, [float(r["avg_max_reward"]) for r in ev], "-o", c=c, label=run)
        ax[2].plot(es, [float(r["success_rate"]) for r in ev], "-o", c=c, label=run)
    ax[0].set_title("flow-matching loss")
    ax[0].set_ylim(0.05, 0.2)
    ax[1].set_title("avg max reward (50 episodes)")
    ax[2].set_title("success rate (50 episodes)")
    ax[1].axhline(0.97, c="gray", ls=":", label="LeRobot DP keypoints")
    ax[2].axhline(0.71, c="gray", ls=":", label="LeRobot DP keypoints")
    for a in ax:
        a.set_xlabel("step")
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out, dpi=100)
    print("saved", args.out)


if __name__ == "__main__":
    main()
