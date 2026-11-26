"""Flow-matching action policy (pi0-style action expert, MLP version) for PushT.

Velocity field v_theta(a_t, obs, t): action chunk (16 x 2, flattened to 32) conditioned on the 36-D
observation window. Same flow as M1:
  a0 ~ N(0, I) (noise),  a1 = normalised action chunk,  t ~ U(0, 1)
  a_t = (1 - t) a0 + t a1,  target u = a1 - a0,  loss = mean ||v - u||^2
Convention: t=0 is noise, t=1 is data (pi0 uses the opposite).
Networks (arch):
  "mlp":  MLP([a_t (32), obs (36), time embedding (2 * n_freqs)]) -> 32
  "film": FiLMResMLP(x=a_t, cond=[obs, time embedding]) -> 32 (see film.py)
"""
import json

import numpy as np

from .data import ACT_DIM, N_ACTION_STEPS, N_OBS_STEPS, OBS_DIM
from .film import FiLMResMLP
from .flow import time_embedding
from .mlp import MLP

CHUNK_DIM = N_ACTION_STEPS * ACT_DIM   # 32
COND_DIM = N_OBS_STEPS * OBS_DIM       # 36


class FlowPolicy:
    def __init__(self, hidden=512, layers=3, n_freqs=8, rng=None, dtype=np.float32, arch="mlp", cond_hidden=256):
        """layers = hidden layers ("mlp") or residual blocks ("film")."""
        self.config = {"hidden": hidden, "layers": layers, "n_freqs": n_freqs, "arch": arch,
                       "cond_hidden": cond_hidden}
        self.n_freqs, self.arch = n_freqs, arch
        if arch == "mlp":
            sizes = [CHUNK_DIM + COND_DIM + 2 * n_freqs] + [hidden] * layers + [CHUNK_DIM]
            self.net = MLP(sizes, rng, dtype=dtype)
        elif arch == "film":
            self.net = FiLMResMLP(CHUNK_DIM, COND_DIM + 2 * n_freqs, CHUNK_DIM, hidden, layers, cond_hidden, rng,
                                  dtype=dtype)
        else:
            raise ValueError(arch)

    def velocity(self, a_t, obs, t):
        """a_t (B, 32), obs (B, 36), t (B,) -> (B, 32)."""
        temb = time_embedding(t.astype(a_t.dtype), self.n_freqs)
        if self.arch == "mlp":
            return self.net.forward(np.concatenate([a_t, obs, temb], axis=1))
        return self.net.forward(a_t, np.concatenate([obs, temb], axis=1))

    def loss_and_grads(self, obs, actions, rng):
        """obs (B, 36), actions (B, 16, 2) normalised. Returns (loss, grads)."""
        a1 = actions.reshape(len(actions), CHUNK_DIM)
        a0 = rng.standard_normal(a1.shape).astype(a1.dtype)
        t = rng.uniform(0.0, 1.0, len(a1)).astype(a1.dtype)
        a_t = (1 - t[:, None]) * a0 + t[:, None] * a1
        diff = self.velocity(a_t, obs, t) - (a1 - a0)
        loss = float(np.mean(diff ** 2))
        grads = self.net.backward(2.0 * diff / diff.size)[0]
        return loss, grads

    def sample(self, obs, n_steps=10, rng=None):
        """obs (B, 36) normalised -> normalised action chunk (B, 16, 2), Euler from t=0 to t=1."""
        rng = np.random.default_rng() if rng is None else rng
        B = len(obs)
        a = rng.standard_normal((B, CHUNK_DIM)).astype(np.float32)
        dt = 1.0 / n_steps
        for k in range(n_steps):
            a = a + dt * self.velocity(a, obs, np.full(B, k * dt, dtype=np.float32))
        return a.reshape(B, N_ACTION_STEPS, ACT_DIM)

    # ------------------------------------------------------------------ save / load
    def save(self, path, params=None, extra=None):
        """params: e.g. EMA weights (default: current weights). Saved as .npz + config JSON inside."""
        params = self.net.params if params is None else params
        meta = json.dumps({"config": self.config, "extra": extra or {}})
        np.savez(path, __meta__=np.array(meta), **params)

    @classmethod
    def load(cls, path):
        f = np.load(path)
        meta = json.loads(str(f["__meta__"]))
        pol = cls(**meta["config"])
        for k in pol.net.params:
            pol.net.params[k] = f[k].astype(np.float32)
        return pol, meta["extra"]


class EMA:
    """Exponential moving average of the weights (as in Diffusion Policy). Evaluate with the EMA weights."""

    def __init__(self, params, decay=0.999):
        self.decay = decay
        self.params = {k: v.copy() for k, v in params.items()}

    def update(self, params):
        for k, v in params.items():
            self.params[k] = self.decay * self.params[k] + (1.0 - self.decay) * v
