from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.habitat_env.env_wrapper import HabitatWrapper
from src.habitat_env.mock_env import MockPointNavConfig, MockPointNavEnv, make_mock_env
from src.models.encoder import VisualEncoder
from src.models.policy import Policy
from src.train.frequency_adapt import freq_adapt
from src.utils.metrics import compute_spl


def _to_torch_obs(obs: np.ndarray, device: torch.device, image_size: int) -> torch.Tensor:
    obs_t = torch.from_numpy(obs).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    obs_t = torch.nn.functional.interpolate(
        obs_t,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    return obs_t.to(device)


def _as_int(cfg: Dict[str, Any], key: str, default: int) -> int:
    return int(cfg.get(key, default))


def _make_env(args: argparse.Namespace, image_size: int, max_episode_steps: int):
    if args.mock_env:
        cfg = MockPointNavConfig(
            image_size=image_size,
            max_episode_steps=max_episode_steps,
            success_distance=args.success_distance,
            seed=args.seed,
        )
        return make_mock_env(cfg), "mock"

    try:
        return HabitatWrapper(args.habitat_config), "habitat"
    except Exception as exc:
        print(f"[eval_nav] Habitat unavailable ({exc}). Falling back to mock env.")
        cfg = MockPointNavConfig(
            image_size=image_size,
            max_episode_steps=max_episode_steps,
            success_distance=args.success_distance,
            seed=args.seed,
        )
        return make_mock_env(cfg), "mock"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPO navigation checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--habitat-config", default="configs/habitat_pointnav.yaml")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mock-env", action="store_true")
    parser.add_argument("--success-distance", type=float, default=0.2)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    image_size = _as_int(cfg, "image_size", 224)
    max_episode_steps = _as_int(cfg, "max_episode_steps", 500)

    freq_cfg = cfg.get("frequency_adapt", {})
    freq_enabled = bool(freq_cfg.get("enabled", False))
    freq_radius = int(freq_cfg.get("radius", 16))
    freq_noise_std = float(freq_cfg.get("noise_std", 1.0))

    env, env_kind = _make_env(args, image_size=image_size, max_episode_steps=max_episode_steps)
    policy = Policy(VisualEncoder()).to(device)
    policy.eval()

    ckpt = torch.load(args.checkpoint, map_location=device)
    policy.load_state_dict(ckpt["model_state_dict"])

    success_flags = []
    spl_scores = []
    returns = []
    lengths = []

    for ep in range(args.episodes):
        obs_np = env.reset()
        done = False
        ep_return = 0.0
        ep_len = 0
        final_info: Dict[str, Any] = {}

        while not done and ep_len < max_episode_steps:
            obs_t = _to_torch_obs(obs_np, device=device, image_size=image_size)
            if freq_enabled:
                obs_t = freq_adapt(obs_t, radius=freq_radius, noise_std=freq_noise_std)

            with torch.no_grad():
                logits, _ = policy(obs_t)
                action = torch.argmax(logits, dim=-1)

            obs_np, reward, done, info = env.step(int(action.item()))
            ep_return += float(reward)
            ep_len += 1
            final_info = info

        success = bool(final_info.get("success", ep_return > 0.0))
        path_len = float(final_info.get("path_length", ep_len))
        shortest_path = float(final_info.get("shortest_path", path_len))
        spl = compute_spl(success=success, path_len=path_len, shortest_path=shortest_path)

        success_flags.append(float(success))
        spl_scores.append(float(spl))
        returns.append(float(ep_return))
        lengths.append(ep_len)
        print(
            f"[eval_nav] episode={ep + 1}/{args.episodes} "
            f"return={ep_return:.3f} success={int(success)} spl={spl:.3f} length={ep_len}"
        )

    results = {
        "checkpoint": args.checkpoint,
        "env_kind": env_kind,
        "episodes": args.episodes,
        "success_rate": float(np.mean(success_flags)) if success_flags else 0.0,
        "mean_spl": float(np.mean(spl_scores)) if spl_scores else 0.0,
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
    }

    print(json.dumps(results, indent=2))
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"[eval_nav] Saved results to {out_path}")


if __name__ == "__main__":
    main()
