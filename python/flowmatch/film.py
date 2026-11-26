"""Residual MLP with FiLM conditioning, NumPy, manual backprop.

    e   = SiLU(cond @ Wc + bc)                          cond = [obs (36), time embedding]   -> (B, C)
    h0  = x @ Win + bin                                 x = noisy action chunk (32)          -> (B, H)
    block k (k = 0..n_blocks-1):
        u1     = SiLU(h) @ W1 + b1
        scale, shift = split(e @ Wg + bg)               per-block FiLM from the condition
        u2     = u1 * (1 + scale) + shift
        h      = h + SiLU(u2) @ W2 + b2                 W2, b2 start at zero: each block starts as identity
    y   = SiLU(h_L) @ Wout + bout                                                            -> (B, out)

Same idea as Diffusion Policy's conditional U-Net (FiLM in every residual block), without the convolutions.
"""
import numpy as np

from .mlp import silu, silu_grad


class FiLMResMLP:
    def __init__(self, x_dim, cond_dim, out_dim, hidden=512, n_blocks=3, cond_hidden=256, rng=None,
                 dtype=np.float32):
        rng = np.random.default_rng(0) if rng is None else rng

        def lin(i, o, zero=False):
            b = 1.0 / np.sqrt(i)
            W = np.zeros((i, o)) if zero else rng.uniform(-b, b, (i, o))
            bias = np.zeros(o) if zero else rng.uniform(-b, b, o)
            return W.astype(dtype), bias.astype(dtype)

        P = {}
        P["Wc"], P["bc"] = lin(cond_dim, cond_hidden)
        P["Win"], P["bin"] = lin(x_dim, hidden)
        for k in range(n_blocks):
            P[f"W1_{k}"], P[f"b1_{k}"] = lin(hidden, hidden)
            P[f"Wg_{k}"], P[f"bg_{k}"] = lin(cond_hidden, 2 * hidden)
            P[f"W2_{k}"], P[f"b2_{k}"] = lin(hidden, hidden, zero=True)
        P["Wout"], P["bout"] = lin(hidden, out_dim)
        self.params, self.n_blocks, self.hidden = P, n_blocks, hidden
        self._cache = None

    def forward(self, x, cond):
        P, H = self.params, self.hidden
        e_pre = cond @ P["Wc"] + P["bc"]
        e = silu(e_pre)
        h = x @ P["Win"] + P["bin"]
        blocks = []
        for k in range(self.n_blocks):
            a1 = silu(h)
            u1 = a1 @ P[f"W1_{k}"] + P[f"b1_{k}"]
            g = e @ P[f"Wg_{k}"] + P[f"bg_{k}"]
            scale, shift = g[:, :H], g[:, H:]
            u2 = u1 * (1.0 + scale) + shift
            a2 = silu(u2)
            blocks.append((h, a1, u1, scale, u2, a2))
            h = h + a2 @ P[f"W2_{k}"] + P[f"b2_{k}"]
        a_out = silu(h)
        self._cache = (x, cond, e_pre, e, blocks, h, a_out)
        return a_out @ P["Wout"] + P["bout"]

    def backward(self, dout):
        """dout = dL/dy. Returns (grads, dL/dx, dL/dcond)."""
        P = self.params
        x, cond, e_pre, e, blocks, h_last, a_out = self._cache
        G = {}
        G["Wout"] = a_out.T @ dout
        G["bout"] = dout.sum(0)
        dh = (dout @ P["Wout"].T) * silu_grad(h_last)
        de = np.zeros_like(e)
        for k in reversed(range(self.n_blocks)):
            h, a1, u1, scale, u2, a2 = blocks[k]
            G[f"W2_{k}"] = a2.T @ dh
            G[f"b2_{k}"] = dh.sum(0)
            du2 = (dh @ P[f"W2_{k}"].T) * silu_grad(u2)
            du1 = du2 * (1.0 + scale)
            dg = np.concatenate([du2 * u1, du2], axis=1)  # d scale, d shift
            G[f"Wg_{k}"] = e.T @ dg
            G[f"bg_{k}"] = dg.sum(0)
            de += dg @ P[f"Wg_{k}"].T
            G[f"W1_{k}"] = a1.T @ du1
            G[f"b1_{k}"] = du1.sum(0)
            dh = dh + (du1 @ P[f"W1_{k}"].T) * silu_grad(h)
        G["Win"] = x.T @ dh
        G["bin"] = dh.sum(0)
        dx = dh @ P["Win"].T
        de_pre = de * silu_grad(e_pre)
        G["Wc"] = cond.T @ de_pre
        G["bc"] = de_pre.sum(0)
        dcond = de_pre @ P["Wc"].T
        return G, dx, dcond
