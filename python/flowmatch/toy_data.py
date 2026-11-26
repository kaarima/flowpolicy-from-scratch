"""2D toy distributions for M1, plus a distance to compare sample sets."""
import numpy as np


def two_moons(n, rng, noise=0.05):
    k = rng.integers(0, 2, n)
    a = rng.uniform(0, np.pi, n)
    x = np.where(k == 0, np.cos(a), 1 - np.cos(a))
    y = np.where(k == 0, np.sin(a), 0.5 - np.sin(a))
    pts = np.stack([x, y], 1) + noise * rng.standard_normal((n, 2))
    return ((pts - [0.5, 0.25]) * 2.0).astype(np.float32)  # roughly centred, unit scale


def checkerboard(n, rng):
    """8 squares on a 4x4 board, in [-2, 2]^2."""
    x = rng.uniform(-2, 2, n)
    y = rng.uniform(0, 1, n) + rng.integers(0, 2, n) * 2 - 2
    y = y + (np.floor(x) % 2)
    return np.stack([x, y], 1).astype(np.float32)


def eight_gaussians(n, rng, radius=2.0, std=0.1):
    ang = rng.integers(0, 8, n) * (np.pi / 4)
    c = radius * np.stack([np.cos(ang), np.sin(ang)], 1)
    return (c + std * rng.standard_normal((n, 2))).astype(np.float32)


DATASETS = {"moons": two_moons, "checkerboard": checkerboard, "8gaussians": eight_gaussians}


def sliced_wasserstein(a, b, n_proj=256, rng=None):
    """Sliced 1-Wasserstein distance between equal-size point sets a, b (N, D)."""
    rng = np.random.default_rng(0) if rng is None else rng
    d = rng.standard_normal((a.shape[1], n_proj))
    d /= np.linalg.norm(d, axis=0, keepdims=True)
    pa, pb = np.sort(a @ d, axis=0), np.sort(b @ d, axis=0)
    return float(np.mean(np.abs(pa - pb)))
