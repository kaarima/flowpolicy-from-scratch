"""Gradient checks. torch is used here only, as a reference.

Run:  pytest -q
"""
import numpy as np
import pytest
import torch

from flowmatch.adam import Adam
from flowmatch.flow import FlowModel
from flowmatch.mlp import MLP

SIZES = [5, 16, 16, 3]


def torch_mlp(mlp):
    """Same network in torch, sharing the NumPy weights (float64)."""
    ps = {k: torch.tensor(v, dtype=torch.float64, requires_grad=True) for k, v in mlp.params.items()}

    def f(x):
        h = x
        for i in range(mlp.n_layers):
            h = h @ ps[f"W{i}"] + ps[f"b{i}"]
            if i < mlp.n_layers - 1:
                h = torch.nn.functional.silu(h)
        return h

    return f, ps


def test_mlp_forward_backward_matches_torch():
    rng = np.random.default_rng(1)
    mlp = MLP(SIZES, rng, dtype=np.float64)
    x = rng.standard_normal((7, SIZES[0]))
    dout = rng.standard_normal((7, SIZES[-1]))

    out = mlp.forward(x)
    grads, dx = mlp.backward(dout)

    f, ps = torch_mlp(mlp)
    xt = torch.tensor(x, requires_grad=True)
    out_t = f(xt)
    (out_t * torch.tensor(dout)).sum().backward()

    np.testing.assert_allclose(out, out_t.detach().numpy(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dx, xt.grad.numpy(), rtol=1e-10, atol=1e-12)
    for k in mlp.params:
        np.testing.assert_allclose(grads[k], ps[k].grad.numpy(), rtol=1e-10, atol=1e-12, err_msg=k)


def test_flow_loss_grads_finite_difference():
    """Full loss (time embedding, interpolation, MSE) vs central differences, float64."""
    rng = np.random.default_rng(2)
    model = FlowModel(MLP([2 + 2 * 4, 16, 16, 2], rng, dtype=np.float64), n_freqs=4)
    x1 = rng.standard_normal((9, 2))

    def loss_only():
        return model.loss_and_grads(x1, np.random.default_rng(123))[0]

    _, grads = model.loss_and_grads(x1, np.random.default_rng(123))
    eps = 1e-6
    for k, p in model.mlp.params.items():
        idx = list(np.ndindex(p.shape))
        for j in np.random.default_rng(3).choice(len(idx), size=min(10, len(idx)), replace=False):
            i = idx[j]
            old = p[i]
            p[i] = old + eps
            lp = loss_only()
            p[i] = old - eps
            lm = loss_only()
            p[i] = old
            num = (lp - lm) / (2 * eps)
            assert num == pytest.approx(grads[k][i], rel=1e-5, abs=1e-8), (k, i)


def test_adam_matches_torch():
    rng = np.random.default_rng(4)
    params = {"a": rng.standard_normal((3, 4)), "b": rng.standard_normal(4)}
    tparams = [torch.tensor(params["a"], requires_grad=True), torch.tensor(params["b"], requires_grad=True)]
    opt = Adam(params, lr=1e-2)
    topt = torch.optim.Adam(tparams, lr=1e-2)
    for _ in range(20):
        ga, gb = rng.standard_normal((3, 4)), rng.standard_normal(4)
        opt.step({"a": ga, "b": gb})
        tparams[0].grad, tparams[1].grad = torch.tensor(ga), torch.tensor(gb)
        topt.step()
    np.testing.assert_allclose(params["a"], tparams[0].detach().numpy(), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(params["b"], tparams[1].detach().numpy(), rtol=1e-10, atol=1e-12)


def test_adamw_matches_torch():
    rng = np.random.default_rng(5)
    params = {"a": rng.standard_normal((3, 4))}
    tp = torch.tensor(params["a"], requires_grad=True)
    opt = Adam(params, lr=1e-2, weight_decay=0.1)
    topt = torch.optim.AdamW([tp], lr=1e-2, weight_decay=0.1)
    for _ in range(20):
        g = rng.standard_normal((3, 4))
        opt.step({"a": g})
        tp.grad = torch.tensor(g)
        topt.step()
    np.testing.assert_allclose(params["a"], tp.detach().numpy(), rtol=1e-10, atol=1e-12)
