"""Conditional flow matching with the linear (rectified-flow / OT) path.

  x0 ~ N(0, I) (noise),  x1 ~ data,  t ~ U(0, 1)
  x_t = (1 - t) x0 + t x1            target velocity  u = x1 - x0
  loss = mean || v_theta(x_t, t) - u ||^2
Sampling integrates dx/dt = v_theta(x, t) from t=0 (noise) to t=1 (data) with Euler steps.
Convention: t=0 is noise, t=1 is data (π0 uses the opposite).
"""
import numpy as np


def time_embedding(t, n_freqs=8):
    """t: (B,) in [0, 1] -> (B, 2*n_freqs) sin/cos features at geometric frequencies 1..100."""
    freqs = np.geomspace(1.0, 100.0, n_freqs).astype(t.dtype)
    a = t[:, None] * freqs[None, :]
    return np.concatenate([np.sin(a), np.cos(a)], axis=1)


class FlowModel:
    """Velocity field v_theta(x, t) = MLP([x, emb(t)])."""

    def __init__(self, mlp, n_freqs=8):
        self.mlp, self.n_freqs = mlp, n_freqs

    def velocity(self, x, t):
        return self.mlp.forward(np.concatenate([x, time_embedding(t, self.n_freqs)], axis=1))

    def loss_and_grads(self, x1, rng):
        """One flow-matching step on a data batch x1 (B, D). Returns (loss, grads)."""
        B = x1.shape[0]
        x0 = rng.standard_normal(x1.shape).astype(x1.dtype)
        t = rng.uniform(0.0, 1.0, B).astype(x1.dtype)
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        target = x1 - x0
        v = self.velocity(xt, t)
        diff = v - target
        loss = float(np.mean(diff ** 2))
        grads, _ = self.mlp.backward(2.0 * diff / diff.size)
        return loss, grads

    def sample(self, n, dim, n_steps=50, rng=None, return_path=False, dtype=np.float32):
        rng = np.random.default_rng() if rng is None else rng
        x = rng.standard_normal((n, dim)).astype(dtype)
        dt = 1.0 / n_steps
        path = [x.copy()]
        for k in range(n_steps):
            t = np.full(n, k * dt, dtype=dtype)
            x = x + dt * self.velocity(x, t)
            if return_path:
                path.append(x.copy())
        return (x, np.stack(path)) if return_path else x
