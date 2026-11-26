"""Adam optimizer (Kingma & Ba 2015). With weight_decay > 0 it is AdamW (Loshchilov & Hutter 2019): decoupled
decay p -= lr * wd * p, same as torch.optim.AdamW. `decay` limits it to some params (e.g. weights, not biases)."""
import numpy as np


class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, decay=None):
        self.params = params  # dict name -> array, updated in place
        self.weight_decay = weight_decay
        self.decay = set(params) if decay is None else set(decay)
        self.lr, (self.b1, self.b2), self.eps = lr, betas, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads):
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for k, g in grads.items():
            if self.weight_decay and k in self.decay:
                self.params[k] *= (1.0 - self.lr * self.weight_decay)
            self.m[k] = self.b1 * self.m[k] + (1.0 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1.0 - self.b2) * g * g
            m_hat = self.m[k] / bc1
            v_hat = self.v[k] / bc2
            self.params[k] -= (self.lr * m_hat / (np.sqrt(v_hat) + self.eps)).astype(self.params[k].dtype)
