"""M0: inspect the PushT env and the lerobot/pusht dataset.

What it does:
  1. Env: prints obs/action spaces, runs 150 random steps -> docs/media/random.gif
  2. Dataset: columns, shapes, per-dim min/max of observation.state and action,
     episode count and length min/mean/max (lerobot/pusht and lerobot/pusht_keypoints)
  3. Plots episode 0 agent trajectory vs actions -> docs/media/episode0.png

Run from the repo root (unset ROS PYTHONPATH if you have one):
  python python/scripts/m0_inspect.py
"""
from pathlib import Path

import gymnasium as gym
import gym_pusht  # noqa: F401  (registers gym_pusht/PushT-v0)
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

MEDIA = Path(__file__).resolve().parents[2] / "docs" / "media"
MEDIA.mkdir(parents=True, exist_ok=True)


def inspect_env(n_steps=150, seed=0):
    print("=== Env: gym_pusht/PushT-v0, obs_type='state' ===")
    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")
    print("observation_space:", env.observation_space)
    print("action_space:     ", env.action_space)
    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)
    print("obs[0] (agent_x, agent_y, block_x, block_y, block_theta):", np.round(obs, 2))
    print("info keys:", list(info.keys()))
    frames = [env.render()]
    for _ in range(n_steps):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        frames.append(env.render())
        if terminated or truncated:
            break
    env.close()
    print(f"frame shape: {frames[0].shape}, last reward: {reward:.3f}, success: {info['is_success']}")
    imageio.mimsave(MEDIA / "random.gif", frames, fps=10, loop=0)
    print("saved", MEDIA / "random.gif")


def episode_lengths(ds):
    ep = np.asarray(ds["episode_index"])
    _, counts = np.unique(ep, return_counts=True)
    return counts


def inspect_dataset(name):
    print(f"\n=== Dataset: {name} ===")
    ds = load_dataset(name, split="train")
    print("columns:", ds.column_names)
    print("frames:", len(ds))
    ds_np = ds.with_format("numpy")
    for col in ["observation.state", "observation.environment_state", "action"]:
        if col not in ds.column_names:
            continue
        x = np.stack(ds_np[col])
        print(f"{col}: shape {x.shape}")
        print("  min:", np.round(x.min(0), 2))
        print("  max:", np.round(x.max(0), 2))
    lens = episode_lengths(ds)
    print(f"episodes: {len(lens)}, length min/mean/max: {lens.min()} / {lens.mean():.1f} / {lens.max()}")
    return ds_np


def plot_episode0(ds_np):
    ep = np.asarray(ds_np["episode_index"])
    mask = ep == 0
    state = np.stack(ds_np["observation.state"])[mask]
    action = np.stack(ds_np["action"])[mask]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(state[:, 0], state[:, 1], "-o", ms=2, label="agent position (observation.state)")
    ax.plot(action[:, 0], action[:, 1], "-x", ms=3, alpha=0.7, label="action (target position)")
    ax.plot(*state[0], "gs", ms=8, label="start")
    ax.set_xlim(0, 512)
    ax.set_ylim(512, 0)  # image coordinates: y down
    ax.set_aspect("equal")
    ax.set_title(f"lerobot/pusht episode 0 ({mask.sum()} frames)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(MEDIA / "episode0.png", dpi=120)
    print("saved", MEDIA / "episode0.png")


if __name__ == "__main__":
    inspect_env()
    ds_np = inspect_dataset("lerobot/pusht")
    plot_episode0(ds_np)
    inspect_dataset("lerobot/pusht_keypoints")
