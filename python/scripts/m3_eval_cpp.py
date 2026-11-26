"""M3: closed-loop eval of the C++ policy in gym-pusht, same eval loop and seeds as the NumPy policy.

The C++ binary (cpp/build/policy_server) does the whole inference: obs normalisation, 10 Euler steps, action
unnormalisation. Python only steps the env and pipes raw observations / pixel actions. Its noise comes from C++'s own
RNG, so episodes differ from the NumPy run; the success rate should match within the error bars (~70 %).

Run from the repo root (after building cpp/):
  PYTHONPATH=python python python/scripts/m3_eval_cpp.py --episodes 500
"""
import argparse
import json
import struct
import subprocess
import time
from pathlib import Path

import numpy as np

from flowmatch.data import MinMaxNormalizer
from flowmatch.eval_pusht import evaluate

ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / "cpp" / "build" / "policy_server"


class CppPolicy:
    """Looks like FlowPolicy to evaluate(): sample(obs (B, 36) pixels) -> (B, 16, 2) pixels, via the C++ server."""

    def __init__(self, seed=0):
        self.proc = subprocess.Popen([str(EXE), str(ROOT / "cpp" / "weights" / "film_g.bin"), str(seed)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.time = 0.0
        self.calls = 0

    def sample(self, obs, n_steps=None, rng=None):
        obs = np.ascontiguousarray(obs, dtype="<f4")
        B = obs.shape[0]
        t = time.perf_counter()
        self.proc.stdin.write(struct.pack("<I", B) + obs.tobytes())
        self.proc.stdin.flush()
        out = self.proc.stdout.read(B * 32 * 4)
        self.time += time.perf_counter() - t
        self.calls += 1
        return np.frombuffer(out, dtype="<f4").reshape(B, 16, 2)

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--seed0", type=int, default=200000)
    args = p.parse_args()
    pol = CppPolicy()
    ident18, ident2 = MinMaxNormalizer(-np.ones(18), np.ones(18)), MinMaxNormalizer(-np.ones(2), np.ones(2))
    t0 = time.time()
    r = evaluate(pol, ident18, ident2, n_episodes=args.episodes, seed0=args.seed0, n_exec=8)
    pol.close()
    n = args.episodes
    se = np.sqrt(r["success_rate"] * (1 - r["success_rate"]) / n)
    print(f"C++ policy, seeds {args.seed0}..{args.seed0 + n - 1}: success {r['success_rate']:.3f} +- {se:.3f} (1 s.e.)"
          f"  avg max reward {r['avg_max_reward']:.3f}  avg max coverage {r['avg_max_coverage']:.3f}"
          f"  mean length {r['length'].mean():.0f}  {time.time() - t0:.0f}s"
          f"  ({pol.calls} batched calls, {1000 * pol.time / pol.calls:.0f} ms each incl. pipe)")
    out = ROOT / "checkpoints" / "film_noise4_late" / f"cpp_eval{n}_seed{args.seed0}.json"
    json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}, open(out, "w"))


if __name__ == "__main__":
    main()
