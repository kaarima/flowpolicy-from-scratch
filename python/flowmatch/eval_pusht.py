"""Closed-loop evaluation of a FlowPolicy in gym-pusht (keypoints + agent pos).

n_envs environments run in lockstep, so one batched sampling call serves all of them.
Each replan: obs window [t-1, t] (first obs repeated at reset) -> 16 actions -> execute the first n_exec.
Metrics per episode, from the env's info only (never the dataset columns):
  success   = info["is_success"] at any step (coverage > 0.95; the env terminates there)
  max_cov   = max coverage over the episode
  max_rew   = max reward = clip(coverage / 0.95, 0, 1)  (LeRobot's "avg max overlap" is the mean of this)
Episode i is reset with seed = seed0 + i.
"""
import gymnasium as gym
import gym_pusht  # noqa: F401
import numpy as np

from .data import N_EXEC_STEPS, env_obs_to_vector


def evaluate(policy, obs_norm, act_norm, n_episodes=50, seed0=100000, n_exec=N_EXEC_STEPS,
             n_flow_steps=10, max_steps=300, record=0, rng=None):
    """Returns dict with per-episode arrays and means; frames of the first `record` episodes if record > 0."""
    rng = np.random.default_rng(0) if rng is None else rng
    render_mode = "rgb_array" if record else None
    envs = [gym.make("gym_pusht/PushT-v0", obs_type="environment_state_agent_pos", render_mode=render_mode,
                     max_episode_steps=max_steps) for _ in range(n_episodes)]
    cur = np.stack([env_obs_to_vector(e.reset(seed=seed0 + i)[0]) for i, e in enumerate(envs)])
    prev = cur.copy()
    done = np.zeros(n_episodes, bool)
    success = np.zeros(n_episodes, bool)
    max_cov = np.zeros(n_episodes)
    max_rew = np.zeros(n_episodes)
    length = np.zeros(n_episodes, int)
    frames = [[envs[i].render()] for i in range(record)]

    step = 0
    while not done.all() and step < max_steps:
        active = np.flatnonzero(~done)
        obs_in = np.concatenate([obs_norm.normalize(prev[active]), obs_norm.normalize(cur[active])], axis=1)
        chunk = act_norm.unnormalize(policy.sample(obs_in, n_flow_steps, rng))  # (A, 16, 2) pixels
        for k in range(n_exec):
            for j, i in enumerate(active):
                if done[i]:
                    continue
                o, r, term, trunc, info = envs[i].step(np.clip(chunk[j, k], 0, 512))
                prev[i], cur[i] = cur[i], env_obs_to_vector(o)
                max_cov[i] = max(max_cov[i], info["coverage"])
                max_rew[i] = max(max_rew[i], r)
                success[i] |= bool(info["is_success"])
                length[i] += 1
                if i < record:
                    frames[i].append(envs[i].render())
                if term or trunc:
                    done[i] = True
            step += 1
            if step >= max_steps:
                break
    for e in envs:
        e.close()
    out = {"success": success, "max_coverage": max_cov, "max_reward": max_rew, "length": length,
           "success_rate": float(success.mean()), "avg_max_reward": float(max_rew.mean()),
           "avg_max_coverage": float(max_cov.mean()), "n_episodes": n_episodes, "seed0": seed0}
    if record:
        out["frames"] = frames
    return out
