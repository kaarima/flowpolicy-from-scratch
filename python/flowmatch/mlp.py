"""Plain NumPy MLP with manual backprop.

Layers: Linear -> SiLU -> ... -> Linear (no activation on the output).
Weights are stored as W with shape (in, out), so a layer is  h = x @ W + b.
"""
import numpy as np


def silu(z):
    return z / (1.0 + np.exp(-z))


def silu_grad(z):
    s = 1.0 / (1.0 + np.exp(-z))
    return s * (1.0 + z * (1.0 - s))


class MLP:
    def __init__(self, sizes, rng=None, dtype=np.float32):
        """sizes: e.g. [in, 128, 128, out]. Init like torch.nn.Linear: U(-1/sqrt(fan_in), 1/sqrt(fan_in))."""
        rng = np.random.default_rng(0) if rng is None else rng
        self.params = {}
        for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            bound = 1.0 / np.sqrt(fan_in)
            self.params[f"W{i}"] = rng.uniform(-bound, bound, (fan_in, fan_out)).astype(dtype)
            self.params[f"b{i}"] = rng.uniform(-bound, bound, (fan_out,)).astype(dtype)
        self.n_layers = len(sizes) - 1
        self._cache = None

    def forward(self, x):
        """x: (B, in) -> (B, out). Caches what backward needs."""
        inputs, pre = [], []
        h = x
        for i in range(self.n_layers):
            inputs.append(h)
            z = h @ self.params[f"W{i}"] + self.params[f"b{i}"]
            if i < self.n_layers - 1:
                pre.append(z)
                h = silu(z)
            else:
                h = z
        self._cache = (inputs, pre)
        return h

    def backward(self, dout):
        """dout = dL/d(output), shape (B, out). Returns (grads dict, dL/dx)."""
        inputs, pre = self._cache
        grads = {}
        d = dout
        for i in reversed(range(self.n_layers)):
            if i < self.n_layers - 1:
                d = d * silu_grad(pre[i])
            grads[f"W{i}"] = inputs[i].T @ d
            grads[f"b{i}"] = d.sum(axis=0)
            d = d @ self.params[f"W{i}"].T
        return grads, d
