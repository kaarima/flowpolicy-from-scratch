"""M3: export the best M2 policy (G, training seed 0, EMA weights) and test vectors for the C++ port.

Writes:
  cpp/weights/film_g.bin          network weights + normalisation stats + time-embedding frequencies
  cpp/weights/film_g.json         the same config, readable (hidden, blocks, n_freqs, ...)
  cpp/tests/vectors/vectors.bin   fixed inputs and NumPy outputs at several points of the forward pass
Format: python/flowmatch/fpbin.py and cpp/weights/FORMAT.md. Weights are stored (in, out): y = x @ W + b.

Intermediates are read from the real NumPy forward pass (FiLMResMLP._cache), not recomputed.

Run from the repo root:
  PYTHONPATH=python python python/scripts/m3_export.py
"""
import json
from pathlib import Path

import numpy as np

from flowmatch import fpbin
from flowmatch.data import build_datasets, load_stats
from flowmatch.flow import time_embedding
from flowmatch.mlp import silu
from flowmatch.policy import FlowPolicy

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "checkpoints" / "film_noise4_late" / "best.npz"
K = 4                                         # number of test inputs
T_VALUES = np.array([0.0, 0.3, 0.7, 0.95], np.float32)


def main():
    policy, extra = FlowPolicy.load(CKPT)
    obs_norm, act_norm = load_stats(CKPT.parent / "stats.json")
    net, cfg = policy.net, policy.config
    freqs = np.geomspace(1.0, 100.0, policy.n_freqs).astype(np.float32)  # exactly as flow.time_embedding (float32 t)

    # ---------------- weights
    (ROOT / "cpp" / "weights").mkdir(parents=True, exist_ok=True)
    order = ["Wc", "bc", "Win", "bin"]
    for k in range(net.n_blocks):
        order += [f"W1_{k}", f"b1_{k}", f"Wg_{k}", f"bg_{k}", f"W2_{k}", f"b2_{k}"]
    order += ["Wout", "bout"]
    assert set(order) == set(net.params)
    tensors = [(n, net.params[n]) for n in order]
    tensors += [("obs_min", obs_norm.min), ("obs_max", obs_norm.max),
                ("act_min", act_norm.min), ("act_max", act_norm.max), ("time_freqs", freqs)]
    fpbin.write(ROOT / "cpp" / "weights" / "film_g.bin", tensors)
    meta = {"source": str(CKPT.relative_to(ROOT)), "step": extra.get("step"), "hidden": cfg["hidden"],
            "n_blocks": cfg["layers"], "n_freqs": cfg["n_freqs"], "cond_hidden": cfg["cond_hidden"],
            "obs_dim_per_frame": 18, "n_obs_steps": 2, "chunk": [16, 2], "euler_steps": 10,
            "weight_layout": "(in, out), y = x @ W + b", "training": "obs noise 4 px, late t (t = sqrt(u))"}
    json.dump(meta, open(ROOT / "cpp" / "weights" / "film_g.json", "w"), indent=1)

    # bitwise check: read back == source arrays
    back = fpbin.read(ROOT / "cpp" / "weights" / "film_g.bin")
    for n, a in tensors:
        assert back[n].shape == a.shape and back[n].tobytes() == np.asarray(a, "<f4").tobytes(), n
    print(f"film_g.bin: {len(tensors)} tensors, {sum(a.size for _, a in tensors):,} floats, bitwise read-back OK")

    # ---------------- test vectors
    _, val, _, _ = build_datasets(val_frac=0.1, seed=0)
    rng = np.random.default_rng(0)
    obs_n, _ = val.get(np.array([0, 100, 1000, 2000]))                     # 4 real val windows, normalised
    obs_raw = np.concatenate([obs_norm.unnormalize(obs_n[:, :18]), obs_norm.unnormalize(obs_n[:, 18:])], 1)
    x_t = rng.standard_normal((K, 32)).astype(np.float32)
    temb = time_embedding(T_VALUES, policy.n_freqs)
    vel = policy.velocity(x_t, obs_n, T_VALUES)                             # fills net._cache
    x, cond, e_pre, e, blocks, h_last, a_out = net._cache
    h0 = blocks[0][0]                                                       # input projection x @ Win + bin
    block_out = [blocks[k + 1][0] for k in range(net.n_blocks - 1)] + [h_last]
    noise0 = rng.standard_normal((K, 32)).astype(np.float32)
    # full sampler with fixed initial noise: same Euler loop as FlowPolicy.sample
    a = noise0.copy()
    for s in range(10):
        a = a + 0.1 * policy.velocity(a, obs_n, np.full(K, s * 0.1, dtype=np.float32))
    chunk_n = a.reshape(K, 16, 2)
    chunk_px = act_norm.unnormalize(chunk_n)

    vec = [("obs_raw", obs_raw), ("obs_n", obs_n), ("x_t", x_t), ("t", T_VALUES),
           ("temb", temb), ("cond", cond), ("cond_e", e), ("h0", h0),
           ("film0_scale", blocks[0][3]), ("block0_out", block_out[0]),
           ("block1_out", block_out[1]), ("block2_out", block_out[2]), ("velocity", vel),
           ("noise0", noise0), ("sample_chunk_n", chunk_n), ("sample_actions_px", chunk_px)]
    (ROOT / "cpp" / "tests" / "vectors").mkdir(parents=True, exist_ok=True)
    fpbin.write(ROOT / "cpp" / "tests" / "vectors" / "vectors.bin", vec)
    # sanity: e and h0 recomputed directly
    assert np.array_equal(e, silu(cond @ net.params["Wc"] + net.params["bc"]))
    assert np.array_equal(h0, x_t @ net.params["Win"] + net.params["bin"])
    print("vectors.bin:", ", ".join(f"{n}{tuple(np.shape(v))}" for n, v in vec))


if __name__ == "__main__":
    main()
