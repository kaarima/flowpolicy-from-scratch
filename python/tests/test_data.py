"""Tests for the M2 data pipeline on small synthetic episodes (no download needed)."""
import numpy as np

from flowmatch.data import (MinMaxNormalizer, WindowDataset, load_stats, save_stats, split_episodes,
                            window_indices)


def toy_episodes(lengths=(5, 3, 20)):
    ep = np.concatenate([np.full(n, i) for i, n in enumerate(lengths)])
    t = np.arange(len(ep), dtype=np.float32)
    obs = np.stack([t + k for k in range(18)], 1)   # obs value encodes the global frame index
    act = np.stack([t, -t], 1)
    return obs, act, ep


def test_split_disjoint_and_size():
    ep = np.repeat(np.arange(206), 3)
    tr, va = split_episodes(ep, 0.1, seed=0)
    assert len(va) == 21 and len(tr) == 185
    assert set(tr).isdisjoint(va) and set(tr) | set(va) == set(range(206))


def test_normalizer_roundtrip_range_and_constant_dim():
    rng = np.random.default_rng(0)
    x = rng.uniform(-3, 7, (100, 3)).astype(np.float32)
    x[:, 2] = 4.0  # constant dimension
    n = MinMaxNormalizer.fit(x)
    y = n.normalize(x)
    assert np.all(np.isfinite(y))
    assert np.allclose(y[:, :2].min(0), -1) and np.allclose(y[:, :2].max(0), 1)
    assert np.all(y[:, 2] == 0)
    np.testing.assert_allclose(n.unnormalize(y), x, atol=1e-5)


def test_stats_json_roundtrip(tmp_path):
    a = MinMaxNormalizer([0, 1], [2, 5])
    b = MinMaxNormalizer([10], [20])
    save_stats(tmp_path / "s.json", a, b)
    a2, b2 = load_stats(tmp_path / "s.json")
    assert np.array_equal(a.min, a2.min) and np.array_equal(b.max, b2.max)


def test_windows_padding_and_episode_boundaries():
    _, _, ep = toy_episodes()
    obs_idx, act_idx = window_indices(ep, n_obs=2, n_act=16)
    # every index stays in the frame's own episode
    assert np.all(ep[obs_idx] == ep[:, None]) and np.all(ep[act_idx] == ep[:, None])
    # episode 0 = frames 0..4: first obs repeated at t=0, last action repeated after frame 4
    assert obs_idx[0].tolist() == [0, 0]
    assert obs_idx[3].tolist() == [2, 3]
    assert act_idx[0].tolist() == [0, 1, 2, 3, 4] + [4] * 11
    # episode 1 = frames 5..7
    assert obs_idx[5].tolist() == [5, 5]
    assert act_idx[6].tolist() == [6, 7] + [7] * 14
    # long episode 2 starts at 8: no padding in the middle
    assert act_idx[10].tolist() == list(range(10, 26))


def test_window_dataset_shapes_and_content():
    obs, act, ep = toy_episodes()
    on, an = MinMaxNormalizer.fit(obs), MinMaxNormalizer.fit(act)
    ds = WindowDataset(obs, act, ep, np.array([0, 2]), on, an)
    assert len(ds) == 5 + 20
    o, a = ds.sample(7, np.random.default_rng(0))
    assert o.shape == (7, 36) and a.shape == (7, 16, 2)
    # first window of episode 2 (global frame 8): obs = [frame 8, frame 8], actions = frames 8..23
    o, a = ds.get(5)
    np.testing.assert_allclose(on.unnormalize(o[:18]), obs[8], atol=1e-4)
    np.testing.assert_allclose(on.unnormalize(o[18:]), obs[8], atol=1e-4)
    np.testing.assert_allclose(an.unnormalize(a), act[8:24], atol=1e-4)
    n_batches = sum(1 for _ in ds.iterate(10))
    assert n_batches == 3


def test_obs_noise_in_pixels():
    obs, act, ep = toy_episodes()
    on, an = MinMaxNormalizer.fit(obs), MinMaxNormalizer.fit(act)
    ds = WindowDataset(obs, act, ep, np.array([0, 1, 2]), on, an)
    o0, _ = ds.sample(20000, np.random.default_rng(3))
    o1, _ = ds.sample(20000, np.random.default_rng(3), obs_noise_px=2.0)
    diff_px = (o1 - o0) * 0.5 * np.tile(on.range, 2)   # back to pixels
    np.testing.assert_allclose(diff_px.std(0), 2.0, rtol=0.05)
