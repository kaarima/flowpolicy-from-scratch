"""M1: 2D flow-matching toy in NumPy (own MLP, backprop, Adam, Euler sampler).

Trains a velocity field on a 2D toy distribution and reports the sliced Wasserstein distance (SW)
between generated samples and fresh data. Two reference numbers put SW in context:
  floor   = SW(data, other data)   -> best possible (finite-sample noise)
  noise   = SW(N(0, I), data)      -> untrained starting point
Outputs: docs/media/m1_<dataset>.png (data | samples | paths | loss), docs/media/m1_<dataset>.gif

Run from the repo root:
  python python/scripts/m1_toy2d.py --data moons
  python python/scripts/m1_toy2d.py --data checkerboard --steps 20000
"""
import argparse
import time
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flowmatch.adam import Adam
from flowmatch.flow import FlowModel
from flowmatch.mlp import MLP
from flowmatch.toy_data import DATASETS, sliced_wasserstein

MEDIA = Path(__file__).resolve().parents[2] / "docs" / "media"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="moons", choices=list(DATASETS))
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=3, help="number of hidden layers")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--n-freqs", type=int, default=8)
    p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    sample_data = DATASETS[args.data]
    model = FlowModel(MLP([2 + 2 * args.n_freqs] + [args.hidden] * args.layers + [2], rng), args.n_freqs)
    opt = Adam(model.mlp.params, lr=args.lr)

    losses, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        # cosine decay to 5% of lr
        opt.lr = args.lr * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * step / args.steps)))
        loss, grads = model.loss_and_grads(sample_data(args.batch, rng), rng)
        opt.step(grads)
        losses.append(loss)
        if step % 1000 == 0 or step == 1:
            print(f"step {step:6d}  loss(avg 1k) {np.mean(losses[-1000:]):.4f}  {time.time() - t0:.1f}s")

    # evaluate
    n = 5000
    eval_rng = np.random.default_rng(1000)
    data = sample_data(n, eval_rng)
    samples, path = model.sample(n, 2, args.sample_steps, eval_rng, return_path=True)
    sw = sliced_wasserstein(samples, data)
    floor = sliced_wasserstein(sample_data(n, eval_rng), data)
    noise = sliced_wasserstein(eval_rng.standard_normal((n, 2)).astype(np.float32), data)
    print(f"\n{args.data}: SW(samples, data) = {sw:.4f}   floor {floor:.4f}   noise {noise:.4f}"
          f"   train {time.time() - t0:.0f}s")

    # figure
    MEDIA.mkdir(parents=True, exist_ok=True)
    lim = np.abs(data).max() * 1.2
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].scatter(*data.T, s=1, alpha=0.4)
    ax[0].set_title("data")
    ax[1].scatter(*samples.T, s=1, alpha=0.4, c="C1")
    ax[1].set_title(f"samples ({args.sample_steps} Euler steps), SW={sw:.3f}")
    for i in range(200):
        ax[2].plot(path[:, i, 0], path[:, i, 1], lw=0.5, alpha=0.5)
    ax[2].scatter(*path[-1, :200].T, s=3, c="k")
    ax[2].set_title("200 sample paths t: 0 -> 1")
    for a in ax[:3]:
        a.set_xlim(-lim, lim)
        a.set_ylim(-lim, lim)
        a.set_aspect("equal")
    k = 100
    ax[3].plot(np.convolve(losses, np.ones(k) / k, mode="valid"))
    ax[3].set_title("training loss (moving avg 100)")
    ax[3].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(MEDIA / f"m1_{args.data}.png", dpi=100)

    frames = []
    for j in range(0, args.sample_steps + 1, max(1, args.sample_steps // 25)):
        f, a = plt.subplots(figsize=(4, 4))
        a.scatter(*path[j, :2000].T, s=1, alpha=0.5, c="C1")
        a.set_xlim(-lim, lim)
        a.set_ylim(-lim, lim)
        a.set_aspect("equal")
        a.set_title(f"t = {j / args.sample_steps:.2f}")
        f.canvas.draw()
        frames.append(np.asarray(f.canvas.buffer_rgba())[..., :3].copy())
        plt.close(f)
    imageio.mimsave(MEDIA / f"m1_{args.data}.gif", frames + [frames[-1]] * 8, fps=8, loop=0)
    print("saved", MEDIA / f"m1_{args.data}.png", "and .gif")


if __name__ == "__main__":
    main()
