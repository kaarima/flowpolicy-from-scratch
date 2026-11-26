"""FiLMResMLP forward/backward vs torch (float64), and the policy loss with arch='film' vs finite differences."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from flowmatch.film import FiLMResMLP
from flowmatch.policy import FlowPolicy


def test_film_matches_torch():
    rng = np.random.default_rng(0)
    net = FiLMResMLP(5, 7, 3, hidden=8, n_blocks=2, cond_hidden=6, rng=rng, dtype=np.float64)
    for k in net.params:  # zero-init W2/b2 would hide bugs in the block gradients
        net.params[k] = rng.standard_normal(net.params[k].shape) * 0.5
    x, c = rng.standard_normal((4, 5)), rng.standard_normal((4, 7))
    dout = rng.standard_normal((4, 3))
    y = net.forward(x, c)
    G, dx, dc = net.backward(dout)

    P = {k: torch.tensor(v, requires_grad=True) for k, v in net.params.items()}
    xt, ct = torch.tensor(x, requires_grad=True), torch.tensor(c, requires_grad=True)
    e = F.silu(ct @ P["Wc"] + P["bc"])
    h = xt @ P["Win"] + P["bin"]
    for k in range(2):
        u1 = F.silu(h) @ P[f"W1_{k}"] + P[f"b1_{k}"]
        g = e @ P[f"Wg_{k}"] + P[f"bg_{k}"]
        u2 = u1 * (1 + g[:, :8]) + g[:, 8:]
        h = h + F.silu(u2) @ P[f"W2_{k}"] + P[f"b2_{k}"]
    yt = F.silu(h) @ P["Wout"] + P["bout"]
    (yt * torch.tensor(dout)).sum().backward()

    np.testing.assert_allclose(y, yt.detach().numpy(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dx, xt.grad.numpy(), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(dc, ct.grad.numpy(), rtol=1e-10, atol=1e-12)
    for k in net.params:
        np.testing.assert_allclose(G[k], P[k].grad.numpy(), rtol=1e-10, atol=1e-12, err_msg=k)


def test_film_policy_loss_finite_difference():
    rng = np.random.default_rng(1)
    pol = FlowPolicy(hidden=8, layers=2, n_freqs=4, rng=rng, dtype=np.float64, arch="film", cond_hidden=6)
    for k in pol.net.params:
        pol.net.params[k] = rng.standard_normal(pol.net.params[k].shape) * 0.3
    obs, act = rng.standard_normal((5, 36)), rng.uniform(-1, 1, (5, 16, 2))
    _, grads = pol.loss_and_grads(obs, act, np.random.default_rng(7))
    eps = 1e-6
    for k, p in pol.net.params.items():
        i = tuple(rng.integers(0, s) for s in p.shape)
        old = p[i]
        p[i] = old + eps
        lp = pol.loss_and_grads(obs, act, np.random.default_rng(7))[0]
        p[i] = old - eps
        lm = pol.loss_and_grads(obs, act, np.random.default_rng(7))[0]
        p[i] = old
        assert (lp - lm) / (2 * eps) == pytest.approx(grads[k][i], rel=1e-5, abs=1e-8), (k, i)
