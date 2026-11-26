"""PushT data pipeline for M2: split, min/max normalisation, obs/action windows, batches.

Per-frame observation (18-D), the same order everywhere (dataset, env, C++):
    [16 T keypoints (x0, y0, ..., x7, y7), agent_x, agent_y]      pixels in [0, 512]
Window at frame t (convention (a)):
    obs     = frames t-1, t            -> 36-D  (normalised [obs_{t-1}, obs_t])
    actions = actions t, ..., t+15     -> (16, 2)
    At eval: observe t-1 and t, predict 16 actions, execute the first 8 (t .. t+7), then replan.
Padding (never crosses an episode boundary):
    before the episode start, repeat the first obs; after the end, repeat the last action.
Normalisation: x_n = 2 (x - min) / (max - min) - 1, per dimension, min/max from TRAIN episodes only.
"""
import json

import numpy as np

N_OBS_STEPS = 2
N_ACTION_STEPS = 16  # prediction horizon
N_EXEC_STEPS = 8     # executed before replanning
OBS_DIM = 18
ACT_DIM = 2


def load_pusht_keypoints(name="lerobot/pusht_keypoints"):
    """Returns obs (N, 18) float32, actions (N, 2) float32, episode_index (N,) int64. Frames are in episode order."""
    from datasets import load_dataset

    ds = load_dataset(name, split="train").with_format("numpy")
    kp = np.stack(ds["observation.environment_state"])
    agent = np.stack(ds["observation.state"])
    obs = np.concatenate([kp, agent], axis=1).astype(np.float32)
    act = np.stack(ds["action"]).astype(np.float32)
    ep = np.asarray(ds["episode_index"]).astype(np.int64)
    assert np.all(np.diff(ep) >= 0), "frames must be grouped by episode"
    return obs, act, ep


def env_obs_to_vector(env_obs):
    """gym-pusht obs (obs_type='environment_state_agent_pos') -> 18-D vector, same order as the dataset."""
    return np.concatenate([env_obs["environment_state"], env_obs["agent_pos"]]).astype(np.float32)


def split_episodes(episode_index, val_frac=0.1, seed=0):
    """Random split by episode (never by frame). Returns (train_episode_ids, val_episode_ids), sorted."""
    eps = np.unique(episode_index)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(eps)
    n_val = max(1, int(round(val_frac * len(eps))))
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


class MinMaxNormalizer:
    """Per-dimension map [min, max] -> [-1, 1]. A constant dimension (max - min < eps) maps to 0."""

    def __init__(self, x_min, x_max, eps=1e-6):
        self.min = np.asarray(x_min, dtype=np.float32)
        self.max = np.asarray(x_max, dtype=np.float32)
        rng = self.max - self.min
        self.const = rng < eps
        self.range = np.where(self.const, 1.0, rng).astype(np.float32)

    @classmethod
    def fit(cls, x):
        return cls(x.min(axis=0), x.max(axis=0))

    def normalize(self, x):
        y = 2.0 * (x - self.min) / self.range - 1.0
        return np.where(self.const, 0.0, y).astype(np.float32)

    def unnormalize(self, y):
        x = (y + 1.0) * 0.5 * self.range + self.min
        return np.where(self.const, self.min, x).astype(np.float32)

    def to_dict(self):
        return {"min": self.min.tolist(), "max": self.max.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["min"], d["max"])


def save_stats(path, obs_norm, act_norm, meta=None):
    """JSON so that C++ can read it too: {"obs": {"min": [18], "max": [18]}, "action": {...}, "meta": {...}}."""
    with open(path, "w") as f:
        json.dump({"obs": obs_norm.to_dict(), "action": act_norm.to_dict(), "meta": meta or {}}, f, indent=1)


def load_stats(path):
    with open(path) as f:
        d = json.load(f)
    return MinMaxNormalizer.from_dict(d["obs"]), MinMaxNormalizer.from_dict(d["action"])


def window_indices(episode_index, n_obs=N_OBS_STEPS, n_act=N_ACTION_STEPS):
    """For every frame t: obs frame indices (N, n_obs) = t-n_obs+1 .. t and action indices (N, n_act) = t .. t+n_act-1,
    clamped to the frame's own episode (this is the padding)."""
    n = len(episode_index)
    idx = np.arange(n)
    starts = np.searchsorted(episode_index, episode_index, side="left")
    ends = np.searchsorted(episode_index, episode_index, side="right") - 1  # last frame of the episode
    obs_idx = np.clip(idx[:, None] + np.arange(-n_obs + 1, 1)[None, :], starts[:, None], None)
    act_idx = np.clip(idx[:, None] + np.arange(n_act)[None, :], None, ends[:, None])
    return obs_idx, act_idx


class WindowDataset:
    """Normalised windows for a set of episodes. Sample random batches or take all windows in order."""

    def __init__(self, obs, act, episode_index, episode_ids, obs_norm, act_norm):
        self.obs_n = obs_norm.normalize(obs)
        self.act_n = act_norm.normalize(act)
        obs_idx, act_idx = window_indices(episode_index)
        keep = np.isin(episode_index, episode_ids)
        self.obs_idx, self.act_idx = obs_idx[keep], act_idx[keep]
        self.frames = np.flatnonzero(keep)

    def __len__(self):
        return len(self.frames)

    def get(self, i):
        """Window(s) by position i (int or array) in this dataset -> obs (.., 36), actions (.., 16, 2)."""
        o = self.obs_n[self.obs_idx[i]]  # (.., 2, 18)
        a = self.act_n[self.act_idx[i]]  # (.., 16, 2)
        return o.reshape(*o.shape[:-2], N_OBS_STEPS * OBS_DIM), a

    def sample(self, batch_size, rng):
        return self.get(rng.integers(0, len(self), batch_size))

    def iterate(self, batch_size):
        for s in range(0, len(self), batch_size):
            yield self.get(np.arange(s, min(s + batch_size, len(self))))


def build_datasets(val_frac=0.1, seed=0, stats_path=None):
    """Load lerobot/pusht_keypoints, split by episode, fit stats on train. Returns (train, val, obs_norm, act_norm)."""
    obs, act, ep = load_pusht_keypoints()
    train_ids, val_ids = split_episodes(ep, val_frac, seed)
    tr = np.isin(ep, train_ids)
    obs_norm, act_norm = MinMaxNormalizer.fit(obs[tr]), MinMaxNormalizer.fit(act[tr])
    if stats_path is not None:
        save_stats(stats_path, obs_norm, act_norm,
                   {"val_frac": val_frac, "seed": seed, "train_episodes": train_ids.tolist(),
                    "val_episodes": val_ids.tolist(), "obs_order": "16 keypoints (x0,y0..x7,y7), agent_x, agent_y"})
    train = WindowDataset(obs, act, ep, train_ids, obs_norm, act_norm)
    val = WindowDataset(obs, act, ep, val_ids, obs_norm, act_norm)
    return train, val, obs_norm, act_norm
