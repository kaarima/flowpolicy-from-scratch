"""Baseline: evaluate LeRobot's own checkpoint lerobot/diffusion_pusht_keypoints in OUR eval loop.

Same env (gym-pusht, obs_type environment_state_agent_pos), same metric, same seeds as our policy
(flowmatch.eval_pusht.evaluate). Their inference settings are kept as in their config.json: DDIM,
num_inference_steps = 10, n_obs_steps = 2, horizon 16, n_action_steps = 8.

The checkpoint is in LeRobot's old (v1) format: normalisation min/max are buffers inside model.safetensors.
This script rebuilds the model with LeRobot 0.4.4's DiffusionPolicy, loads the weights (strict key check), and
applies their min/max normalisation itself.

PyTorch + LeRobot are used here ONLY for this baseline, never in our model. Runs in the conda env
`gardener-train` (LeRobot 0.4.4), with gym_pusht and the repo's python/ folder on PYTHONPATH:
  PYTHONPATH=<dir with gym_pusht, pymunk, pygame, shapely links>:python \
    ~/miniconda3/envs/gardener-train/bin/python python/scripts/lerobot_baseline_eval.py --seed0 200000 --episodes 500
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from flowmatch.data import MinMaxNormalizer
from flowmatch.eval_pusht import evaluate

REPO = "lerobot/diffusion_pusht_keypoints"
ROOT = Path(__file__).resolve().parents[2]


def build_policy(device):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    old = json.load(open(hf_hub_download(REPO, "config.json")))
    keep = ["n_obs_steps", "horizon", "n_action_steps", "down_dims", "kernel_size", "n_groups",
            "diffusion_step_embed_dim", "use_film_scale_modulation", "noise_scheduler_type", "num_train_timesteps",
            "beta_schedule", "beta_start", "beta_end", "prediction_type", "clip_sample", "clip_sample_range",
            "num_inference_steps", "do_mask_loss_for_padding"]
    cfg = DiffusionConfig(
        **{k: (tuple(old[k]) if isinstance(old[k], list) else old[k]) for k in keep if k in old},
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (2,)),
                        "observation.environment_state": PolicyFeature(FeatureType.ENV, (16,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (2,))},
        device=device,
    )
    policy = DiffusionPolicy(cfg)

    sd = load_file(hf_hub_download(REPO, "model.safetensors"))
    norm = {k: v for k, v in sd.items() if k.startswith(("normalize", "unnormalize"))}
    model_sd = {k: v for k, v in sd.items() if k not in norm}
    missing, unexpected = policy.load_state_dict(model_sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)

    def stats(name):
        return {s: norm[k].numpy().reshape(-1) for k in norm
                if k.startswith("normalize_inputs") and f"buffer_{name}." in k for s in [k.split(".")[-1]]}

    act = {s: norm[k].numpy().reshape(-1) for k in norm
           if k.startswith("unnormalize_outputs") and "buffer_action." in k for s in [k.split(".")[-1]]}
    return policy.to(device).eval(), cfg, stats("observation_state"), stats("observation_environment_state"), act


class LeRobotAdapter:
    """Looks like our FlowPolicy to evaluate(): sample(obs (B, 36) in PIXELS) -> actions (B, 8, 2) in PIXELS.
    Pass identity normalizers to evaluate(), so it hands over raw pixels."""

    def __init__(self, device="cuda"):
        self.device = device
        self.policy, self.cfg, self.s, self.e, self.a = build_policy(device)
        self.calls = 0

    @staticmethod
    def _n(x, st):
        return 2.0 * (x - st["min"]) / (st["max"] - st["min"]) - 1.0

    @torch.inference_mode()
    def sample(self, obs, n_steps=None, rng=None):
        B = len(obs)
        frames = obs.reshape(B, 2, 18)                        # [t-1, t] x [16 keypoints, agent x, y]
        batch = {
            "observation.state": torch.tensor(self._n(frames[:, :, 16:], self.s), dtype=torch.float32),
            "observation.environment_state": torch.tensor(self._n(frames[:, :, :16], self.e), dtype=torch.float32),
        }
        batch = {k: v.to(self.device) for k, v in batch.items()}
        act = self.policy.diffusion.generate_actions(batch).float().cpu().numpy()  # (B, 8, 2) in [-1, 1]
        self.calls += 1
        return (act + 1.0) * 0.5 * (self.a["max"] - self.a["min"]) + self.a["min"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed0", type=int, default=200000)
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--torch-seed", type=int, default=0, help="sampling noise draw (diffusion noise)")
    args = p.parse_args()
    torch.manual_seed(args.torch_seed)

    pol = LeRobotAdapter(args.device)
    n_params = sum(v.numel() for v in pol.policy.parameters())
    print(f"{REPO}: {n_params / 1e6:.1f} M params, {pol.cfg.noise_scheduler_type} "
          f"{pol.cfg.num_inference_steps} denoising steps, device {args.device}")
    ident18, ident2 = MinMaxNormalizer(-np.ones(18), np.ones(18)), MinMaxNormalizer(-np.ones(2), np.ones(2))
    t0 = time.time()
    r = evaluate(pol, ident18, ident2, n_episodes=args.episodes, seed0=args.seed0, n_exec=8)
    n = args.episodes
    se = np.sqrt(r["success_rate"] * (1 - r["success_rate"]) / n)
    print(f"seeds {args.seed0}..{args.seed0 + n - 1}: success {r['success_rate']:.3f} +- {se:.3f} (1 s.e.)  "
          f"avg max reward {r['avg_max_reward']:.3f}  avg max coverage {r['avg_max_coverage']:.3f}  "
          f"mean length {r['length'].mean():.0f}  {time.time() - t0:.0f}s")
    out = ROOT / "checkpoints" / "lerobot_dp_keypoints"
    out.mkdir(parents=True, exist_ok=True)
    res = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
    res.update({"repo": REPO, "n_params": n_params, "denoise_steps": pol.cfg.num_inference_steps})
    suffix = f"_torch{args.torch_seed}" if args.torch_seed else ""
    json.dump(res, open(out / f"eval{n}_seed{args.seed0}{suffix}.json", "w"))


if __name__ == "__main__":
    main()
