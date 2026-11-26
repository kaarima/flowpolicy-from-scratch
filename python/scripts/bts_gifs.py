"""Behind-the-scenes GIFs: record chosen PushT episodes of one or more checkpoints, side by side, with a label.

Each panel shows: run name, step, live coverage and the outcome. The same reset seed is used for every panel,
so you see different policies from the same start. Sampling noise differs from the 500-episode eval, so an
episode can end differently from its row in best_eval500.json; the label always shows what happened here.

Run from the repo root:
  PYTHONPATH=python python python/scripts/bts_gifs.py --seed 200000 --runs base film512 --name base_vs_film
"""
import argparse
from pathlib import Path

import gymnasium as gym
import gym_pusht  # noqa: F401
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from flowmatch.data import env_obs_to_vector, load_stats
from flowmatch.policy import FlowPolicy

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "media" / "bts"


def run_episode(run, seed, n_exec=8, flow_steps=10, max_steps=300, sample_seed=0):
    ckpt = ROOT / "checkpoints" / run / "best.npz"
    policy, extra = FlowPolicy.load(ckpt)
    obs_norm, act_norm = load_stats(ckpt.parent / "stats.json")
    rng = np.random.default_rng(sample_seed)
    env = gym.make("gym_pusht/PushT-v0", obs_type="environment_state_agent_pos", render_mode="rgb_array",
                   max_episode_steps=max_steps)
    o, _ = env.reset(seed=seed)
    cur = prev = env_obs_to_vector(o)
    frames, covs, success, step, done = [env.render()], [env.unwrapped._get_coverage()], False, 0, False
    while not done:
        obs_in = np.concatenate([obs_norm.normalize(prev), obs_norm.normalize(cur)])[None]
        chunk = act_norm.unnormalize(policy.sample(obs_in, flow_steps, rng))[0]
        for k in range(n_exec):
            o, r, term, trunc, info = env.step(np.clip(chunk[k], 0, 512))
            prev, cur = cur, env_obs_to_vector(o)
            frames.append(env.render())
            covs.append(info["coverage"])
            success |= bool(info["is_success"])
            step += 1
            if term or trunc:
                done = True
                break
    env.close()
    return frames, covs, success, extra.get("step")


def label(frame, lines, color):
    im = Image.fromarray(frame).resize((340, 340))
    d = ImageDraw.Draw(im)
    for i, (txt, col) in enumerate(lines):
        d.text((8, 6 + 14 * i), txt, fill=col or (0, 0, 0))
    d.rectangle([0, 0, 339, 339], outline=color, width=3)
    return np.asarray(im)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--title", default="")
    p.add_argument("--every", type=int, default=2, help="keep every n-th frame (env is 10 Hz)")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    panels, summary = [], []
    for run in args.runs:
        frames, covs, success, ckpt_step = run_episode(run, args.seed)
        n = len(frames)
        outcome = "SUCCESS" if success else f"FAIL (max cov {max(covs):.2f})"
        color = (40, 160, 40) if success else (200, 40, 40)
        lab = []
        for i, f in enumerate(frames):
            lines = [(f"{run} (ckpt {ckpt_step // 1000}k)", None), (f"seed {args.seed}  step {i}", None),
                     (f"coverage {covs[i]:.2f}  max {max(covs[:i + 1]):.2f}", None)]
            if i == n - 1:
                lines.append((outcome, color))
            lab.append(label(f, lines, color if i == n - 1 else (180, 180, 180)))
        panels.append(lab)
        summary.append(f"{run}: {outcome}, {n - 1} steps")

    T = max(len(p_) for p_ in panels)
    frames = [np.concatenate([p_[min(t, len(p_) - 1)] for p_ in panels], axis=1) for t in range(0, T, args.every)]
    frames += [frames[-1]] * 10  # hold the last frame
    path = OUT / f"{args.name}.gif"
    imageio.mimsave(path, frames, fps=5, loop=0)
    print(f"{path}  |  " + "  |  ".join(summary))


if __name__ == "__main__":
    main()
