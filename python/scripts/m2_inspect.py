"""M2 investigation: does gym-pusht's keypoint observation match lerobot/pusht_keypoints?

What it does:
  1. Dataset: columns, shapes, per-dim min/max, episodes and lengths, fps.
  2. Env: obs_type="environment_state_agent_pos" (keypoints + agent pos, from the gym_pusht source),
     reset with a fixed seed, print the obs and info.
  3. Order / scale check, frame by frame: recover the block pose (x, y, theta) from each dataset frame's
     16 keypoints with a rigid 2D fit to the env's T shape, reset the env to that state, and compare the
     env's 16 keypoints with the dataset's. A wrong keypoint order shows up as a large fit residual; a
     wrong scale or unit as a large env-vs-dataset difference.
  4. Replay: reset the env to frame 0 of some episodes and step the recorded actions open loop. Compares
     keypoints along the way, and the env's success/reward with the dataset's next.success/next.reward.
  5. Action: confirms it is a target position (agent moves towards it) and prints its range.

Run from the repo root:
  python python/scripts/m2_inspect.py
"""
import gymnasium as gym
import gym_pusht  # noqa: F401
import numpy as np
from datasets import load_dataset

OBS_TYPE = "environment_state_agent_pos"


def episode_slices(ep_index):
    starts = np.flatnonzero(np.r_[True, ep_index[1:] != ep_index[:-1]])
    ends = np.r_[starts[1:], len(ep_index)]
    return list(zip(starts, ends))


def fit_pose(local, world):
    """Rigid 2D fit world ~= R(theta) @ local + p. Returns (p, theta, rms residual)."""
    lc, wc = local.mean(0), world.mean(0)
    a, b = local - lc, world - wc
    theta = np.arctan2((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum(), (a * b).sum())
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    p = wc - R @ lc
    res = world - (local @ R.T + p)
    return p, theta, float(np.sqrt((res ** 2).sum(1).mean()))


def reset_state(agent_xy, p, theta, cog):
    """State for env.reset(options={"reset_to_state": ...}) that puts the block origin at p.

    gym_pusht's _set_state sets the block position first, then the angle, and pymunk rotates about the
    centre of gravity (cog, body frame). That moves the origin by cog - R(theta) cog, so pre-compensate.
    """
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return [*agent_xy, *(p - (cog - R @ cog)), theta]


def main():
    # ---------------------------------------------------------------- 1. dataset
    print("=== 1. Dataset lerobot/pusht_keypoints ===")
    ds = load_dataset("lerobot/pusht_keypoints", split="train").with_format("numpy")
    print("columns:", ds.column_names)
    kp = np.stack(ds["observation.environment_state"]).astype(np.float64)
    agent = np.stack(ds["observation.state"]).astype(np.float64)
    act = np.stack(ds["action"]).astype(np.float64)
    ep = np.asarray(ds["episode_index"])
    ts = np.asarray(ds["timestamp"])
    success = np.asarray(ds["next.success"])
    reward = np.asarray(ds["next.reward"])
    for name, x in [("observation.environment_state", kp), ("observation.state", agent), ("action", act)]:
        print(f"{name}: shape {x.shape}\n  min {np.round(x.min(0), 1)}\n  max {np.round(x.max(0), 1)}")
    sl = episode_slices(ep)
    lens = np.array([e - s for s, e in sl])
    print(f"episodes {len(sl)}, length min/mean/max {lens.min()}/{lens.mean():.1f}/{lens.max()}, "
          f"frame dt {np.median(np.diff(ts[:lens[0]])):.3f} s")
    ep_success = np.array([success[s:e].any() for s, e in sl])
    print(f"episodes with next.success at some frame: {ep_success.sum()}/{len(sl)}")

    # ---------------------------------------------------------------- 2. env
    print(f"\n=== 2. Env gym_pusht/PushT-v0, obs_type='{OBS_TYPE}' ===")
    env = gym.make("gym_pusht/PushT-v0", obs_type=OBS_TYPE)
    print("observation_space:", env.observation_space)
    print("action_space:", env.action_space, " max_episode_steps:", env.spec.max_episode_steps)
    print("control_hz:", env.unwrapped.control_hz, " success_threshold:", env.unwrapped.success_threshold)
    obs, info = env.reset(seed=42)
    print("reset(seed=42) obs:", {k: np.round(v, 1) for k, v in obs.items()})
    print("reset info:", {k: np.round(v, 3) if isinstance(v, np.ndarray) else v for k, v in info.items()})
    obs, r, term, trunc, info = env.step(np.array([256.0, 256.0]))
    print("step info keys:", list(info.keys()), f" coverage {info['coverage']:.4f}  reward {r:.4f}  "
          f"is_success {info['is_success']}  terminated {term}")

    # T shape in the block frame: reset with block at (256, 256), angle 0
    obs, _ = env.reset(options={"reset_to_state": [100, 100, 256, 256, 0.0]})
    local = obs["environment_state"].reshape(8, 2) - [256, 256]
    cog = np.array(env.unwrapped.block.center_of_gravity)
    print("block centre of gravity (body frame):", cog)

    # ---------------------------------------------------------------- 3. order / scale, every frame
    print("\n=== 3. Order and scale: env keypoints at the dataset's fitted pose vs dataset keypoints ===")
    fit_res, kp_err, ag_err = [], [], []
    for i in range(len(kp)):
        p, th, res = fit_pose(local, kp[i].reshape(8, 2))
        o, _ = env.reset(options={"reset_to_state": reset_state(agent[i], p, th, cog)})
        fit_res.append(res)
        kp_err.append(np.abs(o["environment_state"] - kp[i]).max())
        ag_err.append(np.abs(o["agent_pos"] - agent[i]).max())
    fit_res, kp_err, ag_err = map(np.array, (fit_res, kp_err, ag_err))
    print(f"frames checked: {len(kp)}")
    print(f"rigid-fit residual (px): median {np.median(fit_res):.2e}  max {fit_res.max():.2e}")
    print(f"|env keypoints - dataset| max over 16 dims (px): median {np.median(kp_err):.2e}  max {kp_err.max():.2e}")
    print(f"|env agent_pos - observation.state| (px): max {ag_err.max():.2e}")
    # what a wrong order would look like: swap the two rectangles (keypoints 0-3 <-> 4-7)
    swapped = np.r_[local[4:], local[:4]]
    print(f"sanity: residual if the two rectangles were swapped: {fit_pose(swapped, kp[0].reshape(8, 2))[2]:.1f} px")
    match = fit_res.max() < 1e-2 and kp_err.max() < 1e-2 and ag_err.max() < 1e-2
    print("RESULT:", "MATCH (same order, same units: pixels in [0, 512])" if match else "MISMATCH")

    # ---------------------------------------------------------------- 4. open-loop replay
    print("\n=== 4. Open-loop replay of recorded actions ===")
    rows = []
    for e in range(0, len(sl), 20):
        s, t = sl[e]
        p, th, _ = fit_pose(local, kp[s].reshape(8, 2))
        env.reset(options={"reset_to_state": reset_state(agent[s], p, th, cog)})
        errs, env_succ, env_rew, env_cov = [], False, [], []
        for i in range(s, t):
            o, r, term, _, info = env.step(act[i])
            env_rew.append(r)
            env_cov.append(info["coverage"])
            env_succ |= bool(info["is_success"])
            if i + 1 < t:
                errs.append(np.abs(o["environment_state"] - kp[i + 1]).max())
        n5 = min(5, len(errs))
        rows.append((e, t - s, max(errs[:n5]), max(errs), np.abs(np.array(env_rew) - reward[s:t]).max(),
                     np.abs(np.array(env_cov) - reward[s:t]).max(), reward[s:t].max(), max(env_cov), env_succ))
    print("kp_err = max |env keypoints - dataset| (px); reward column compared with env reward and env coverage")
    print(" ep  len  kp_err 5 steps  kp_err all  |rew-env_rew|  |rew-env_cov|  data max rew  env max cov  env success")
    for e, n, e5, eall, rd, rc, dmax, cmax, en_s in rows:
        print(f"{e:3d}  {n:3d}  {e5:13.3f}  {eall:10.1f}  {rd:12.3f}  {rc:13.3f}  {dmax:12.3f}  {cmax:11.3f}  {str(en_s):>11}")

    # ---------------------------------------------------------------- 5. action
    print("\n=== 5. Action = target position for the agent (PD controller in the env) ===")
    same = ep[:-1] == ep[1:]
    d_act = np.linalg.norm(act[:-1] - agent[1:], axis=1)[same]
    d_still = np.linalg.norm(agent[:-1] - agent[1:], axis=1)[same]
    gap_before = np.linalg.norm(act[:-1] - agent[:-1], axis=1)[same]
    print(f"action range: min {act.min(0)}  max {act.max(0)}  (env action_space [0, 512])")
    print(f"median |action_t - agent_t|   = {np.median(gap_before):.1f} px")
    print(f"median |action_t - agent_t+1| = {np.median(d_act):.1f} px  (agent moves towards the action)")
    print(f"median |agent_t+1 - agent_t|  = {np.median(d_still):.1f} px per step")
    env.close()


if __name__ == "__main__":
    main()
