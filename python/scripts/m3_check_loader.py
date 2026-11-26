"""M3: check that the C++ loader reads every tensor bitwise-identical to NumPy.

Runs cpp/build/test_vectors, parses "label name shape hash" lines, and compares the FNV-1a hash of each tensor with
the hash of the ORIGINAL NumPy array: weights straight from the checkpoint .npz + stats.json (not from the .bin),
vectors from vectors.bin read by the Python reader. Also checks that a truncated file is rejected.

Run from the repo root (after building cpp/):
  PYTHONPATH=python python python/scripts/m3_check_loader.py
"""
import subprocess
from pathlib import Path

import numpy as np

from flowmatch import fpbin
from flowmatch.data import load_stats

ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / "cpp" / "build" / "test_vectors"
CKPT = ROOT / "checkpoints" / "film_noise4_late" / "best.npz"


def fnv1a64(a):
    h = 1469598103934665603
    for b in np.ascontiguousarray(a, "<f4").tobytes():
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def main():
    out = subprocess.run([str(EXE)], capture_output=True, text=True, check=True).stdout
    cpp = {}
    for line in out.splitlines():
        if line.startswith("#"):
            continue
        label, name, rest = line.split(" ", 2)
        shape, h = rest.rsplit(" ", 1)
        cpp[(label, name)] = (shape, int(h, 16))

    npz = np.load(CKPT)
    obs_norm, act_norm = load_stats(CKPT.parent / "stats.json")
    ref = {("weights", k): npz[k].astype(np.float32) for k in npz.files if k != "__meta__"}
    ref.update({("weights", "obs_min"): obs_norm.min, ("weights", "obs_max"): obs_norm.max,
                ("weights", "act_min"): act_norm.min, ("weights", "act_max"): act_norm.max,
                ("weights", "time_freqs"): np.geomspace(1.0, 100.0, 8).astype(np.float32)})
    ref.update({("vectors", k): v for k, v in fpbin.read(ROOT / "cpp" / "tests" / "vectors" / "vectors.bin").items()})

    assert set(ref) == set(cpp), set(ref) ^ set(cpp)
    for key, a in ref.items():
        shape, h = cpp[key]
        assert h == fnv1a64(a), f"hash mismatch: {key}"
    print(f"OK: {len(ref)} tensors bitwise identical (C++ loader vs NumPy checkpoint/stats/vectors)")

    # negative test: truncated file must be rejected
    bad = ROOT / "cpp" / "build" / "truncated.bin"
    bad.write_bytes((ROOT / "cpp" / "weights" / "film_g.bin").read_bytes()[:-4])
    r = subprocess.run([str(EXE), str(bad)], capture_output=True, text=True)
    assert r.returncode != 0 and "truncated" in r.stderr, r.stderr
    print("OK: truncated file rejected:", r.stderr.strip())


if __name__ == "__main__":
    main()
