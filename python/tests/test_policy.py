"""FlowPolicy: loss gradients vs finite differences (float64), shapes, save/load."""
import numpy as np
import pytest

from flowmatch.policy import EMA, FlowPolicy


def test_policy_loss_grads_finite_difference():
    rng = np.random.default_rng(0)
    pol = FlowPolicy(hidden=16, layers=2, n_freqs=4, rng=rng, dtype=np.float64)
    obs = rng.standard_normal((6, 36))
    act = rng.uniform(-1, 1, (6, 16, 2))
    _, grads = pol.loss_and_grads(obs, act, np.random.default_rng(7))
    eps = 1e-6
    for k, p in pol.net.params.items():
        idx = list(np.ndindex(p.shape))
        for j in np.random.default_rng(1).choice(len(idx), size=min(8, len(idx)), replace=False):
            i = idx[j]
            old = p[i]
            p[i] = old + eps
            lp = pol.loss_and_grads(obs, act, np.random.default_rng(7))[0]
            p[i] = old - eps
            lm = pol.loss_and_grads(obs, act, np.random.default_rng(7))[0]
            p[i] = old
            assert (lp - lm) / (2 * eps) == pytest.approx(grads[k][i], rel=1e-5, abs=1e-8), (k, i)


def test_sample_shape_and_save_load(tmp_path):
    pol = FlowPolicy(hidden=32, layers=2, rng=np.random.default_rng(0))
    obs = np.zeros((3, 36), np.float32)
    a = pol.sample(obs, n_steps=4, rng=np.random.default_rng(1))
    assert a.shape == (3, 16, 2) and a.dtype == np.float32
    ema = EMA(pol.net.params, decay=0.5)
    pol.save(tmp_path / "p.npz", params=ema.params, extra={"step": 3})
    pol2, extra = FlowPolicy.load(tmp_path / "p.npz")
    assert extra == {"step": 3}
    b = pol2.sample(obs, n_steps=4, rng=np.random.default_rng(1))
    np.testing.assert_allclose(a, b)
