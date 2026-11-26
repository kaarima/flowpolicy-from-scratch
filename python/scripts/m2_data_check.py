"""M2: build the PushT data pipeline on the real dataset and print what it produces.

Prints split sizes, normalisation stats, batch shapes and ranges (val data can fall slightly outside [-1, 1]
because stats come from train only), and checks that an env observation gives the same 18-D vector as the
dataset frame it was reset to. Saves the stats to data/pusht_keypoints_stats.json (gitignored; deterministic).

Run from the repo root:
  PYTHONPATH=python python python/scripts/m2_data_check.py
"""
from pathlib import Path

import gymnasium as gym
import gym_pusht  # noqa: F401
import numpy as np

from flowmatch.data import build_datasets, env_obs_to_vector, load_pusht_keypoints, load_stats

ROOT = Path(__file__).resolve().parents[2]
STATS = ROOT / "data" / "pusht_keypoints_stats.json"


def block_reset_state(kp16, agent_xy, local, cog):
    """Same rigid fit + CoG compensation as m2_inspect.py: dataset frame -> env reset_to_state."""
    w = kp16.reshape(8, 2).astype(np.float64)
    lc, wc = local.mean(0), w.mean(0)
    a, b = local - lc, w - wc
    th = np.arctan2((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum(), (a * b).sum())
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    p = wc - R @ lc
    return [*agent_xy, *(p - (cog - R @ cog)), th]


def main():
    STATS.parent.mkdir(exist_ok=True)
    train, val, obs_norm, act_norm = build_datasets(val_frac=0.1, seed=0, stats_path=STATS)
    print(f"windows: train {len(train)}, val {len(val)}")
    print("obs min :", np.round(obs_norm.min, 1))
    print("obs max :", np.round(obs_norm.max, 1))
    print("act min/max:", act_norm.min, act_norm.max, " constant dims:", obs_norm.const.sum() + act_norm.const.sum())

    o, a = train.sample(256, np.random.default_rng(0))
    print(f"train batch: obs {o.shape} {o.dtype} [{o.min():.3f}, {o.max():.3f}],"
          f" actions {a.shape} [{a.min():.3f}, {a.max():.3f}]")
    vo, va = val.get(np.arange(len(val)))
    print(f"val all  : obs [{vo.min():.3f}, {vo.max():.3f}], actions [{va.min():.3f}, {va.max():.3f}]"
          f"  (outside [-1,1]: obs {np.mean(np.abs(vo) > 1):.2%}, act {np.mean(np.abs(va) > 1):.2%})")
    o2, a2 = load_stats(STATS)
    assert np.array_equal(o2.min, obs_norm.min) and np.array_equal(a2.max, act_norm.max)
    print("saved + reloaded", STATS)

    # env obs -> same 18-D vector as the dataset frame
    obs, _, _ = load_pusht_keypoints()
    env = gym.make("gym_pusht/PushT-v0", obs_type="environment_state_agent_pos")
    e, _ = env.reset(options={"reset_to_state": [100, 100, 256, 256, 0.0]})
    local = e["environment_state"].reshape(8, 2) - [256, 256]
    cog = np.array(env.unwrapped.block.center_of_gravity)
    err = 0.0
    for i in np.random.default_rng(1).choice(len(obs), 200, replace=False):
        e, _ = env.reset(options={"reset_to_state": block_reset_state(obs[i, :16], obs[i, 16:], local, cog)})
        err = max(err, np.abs(env_obs_to_vector(e) - obs[i]).max())
    print(f"env_obs_to_vector vs dataset, 200 frames: max |diff| = {err:.2e} px")


if __name__ == "__main__":
    main()
