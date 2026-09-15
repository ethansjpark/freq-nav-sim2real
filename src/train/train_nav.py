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
from src.train.ppo_utils import RolloutBuffer, compute_gae, ppo_update
from src.utils.metrics import compute_spl


def _to_torch_obs(obs: np.ndarray, device: torch.device, image_size: int) -> torch.Tensor:
    if obs.ndim != 3:
        raise ValueError(f"Expected HWC observation, got {obs.shape}")

    obs_t = torch.from_numpy(obs).float()
    obs_t = obs_t.permute(2, 0, 1).unsqueeze(0) / 255.0
    obs_t = torch.nn.functional.interpolate(
        obs_t,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    return obs_t.to(device)


def _as_float(cfg: Dict[str, Any], key: str, default: float) -> float:
    return float(cfg.get(key, default))


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
        print(f"[train_nav] Habitat unavailable ({exc}). Falling back to mock env.")
        cfg = MockPointNavConfig(
            image_size=image_size,
            max_episode_steps=max_episode_steps,
            success_distance=args.success_distance,
            seed=args.seed,
        )
        return make_mock_env(cfg), "mock"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO navigation policy.")
    parser.add_argument("--config", required=True, help="Path to training YAML config.")
    parser.add_argument(
        "--habitat-config",
        default="configs/habitat_pointnav.yaml",
        help="Habitat task config path.",
    )
    parser.add_argument("--output-dir", default="experiments/checkpoints")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mock-env", action="store_true", help="Force mock env.")
    parser.add_argument("--success-distance", type=float, default=0.2)
    parser.add_argument("--num-steps", type=int, default=-1, help="Override YAML num_steps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    num_steps = _as_int(cfg, "num_steps", 500_000)
    if args.num_steps > 0:
        num_steps = args.num_steps
    lr = _as_float(cfg, "lr", 3e-4)
    gamma = _as_float(cfg, "gamma", 0.99)
    gae_lambda = _as_float(cfg, "gae_lambda", 0.95)
    rollout_steps = _as_int(cfg, "rollout_steps", 256)
    minibatch_size = _as_int(cfg, "batch_size", 64)
    ppo_epochs = _as_int(cfg, "ppo_epochs", 4)
    clip_coef = _as_float(cfg, "clip_coef", 0.2)
    value_coef = _as_float(cfg, "value_coef", 0.5)
    entropy_coef = _as_float(cfg, "entropy_coef", 0.01)
    max_grad_norm = _as_float(cfg, "max_grad_norm", 0.5)
    log_interval = _as_int(cfg, "log_interval", 5)
    ckpt_interval = _as_int(cfg, "checkpoint_interval", 10_000)
    image_size = _as_int(cfg, "image_size", 224)
    max_episode_steps = _as_int(cfg, "max_episode_steps", 500)

    freq_cfg = cfg.get("frequency_adapt", {})
    freq_enabled = bool(freq_cfg.get("enabled", False))
    freq_radius = int(freq_cfg.get("radius", 16))
    freq_noise_std = float(freq_cfg.get("noise_std", 1.0))

    env, env_kind = _make_env(args, image_size=image_size, max_episode_steps=max_episode_steps)
    print(f"[train_nav] Using {env_kind} environment on device={device}.")

    policy = Policy(VisualEncoder()).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    buffer = RolloutBuffer()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    obs_np = env.reset()
    ep_return = 0.0
    ep_steps = 0
    episode_returns = []
    episode_success = []
    episode_spl = []
    update_idx = 0

    while global_step < num_steps:
        buffer.clear()
        for _ in range(rollout_steps):
            obs_t = _to_torch_obs(obs_np, device=device, image_size=image_size)
            if freq_enabled:
                obs_t = freq_adapt(obs_t, radius=freq_radius, noise_std=freq_noise_std)

            with torch.no_grad():
                logits, value = policy(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logprob = dist.log_prob(action)

            action_int = int(action.item())
            next_obs, reward, done, info = env.step(action_int)
            reward_t = torch.tensor([reward], dtype=torch.float32, device=device)
            done_t = torch.tensor([float(done)], dtype=torch.float32, device=device)

            buffer.add(
                obs=obs_t,
                action=action.view(1),
                logprob=logprob.view(1),
                reward=reward_t,
                done=done_t,
                value=value.view(1),
            )

            global_step += 1
            ep_return += float(reward)
            ep_steps += 1
            obs_np = next_obs

            if done:
                success = bool(info.get("success", reward > 0.0))
                path_len = float(info.get("path_length", ep_steps))
                shortest_path = float(info.get("shortest_path", path_len))
                spl = compute_spl(success=success, path_len=path_len, shortest_path=shortest_path)
                episode_returns.append(ep_return)
                episode_success.append(float(success))
                episode_spl.append(float(spl))

                obs_np = env.reset()
                ep_return = 0.0
                ep_steps = 0

            if global_step >= num_steps:
                break

        with torch.no_grad():
            next_obs_t = _to_torch_obs(obs_np, device=device, image_size=image_size)
            if freq_enabled:
                next_obs_t = freq_adapt(next_obs_t, radius=freq_radius, noise_std=freq_noise_std)
            _, next_value = policy(next_obs_t)

        batch = buffer.as_tensors()
        gae = compute_gae(
            rewards=batch["rewards"],
            dones=batch["dones"],
            values=batch["values"],
            next_value=next_value.view(1),
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        batch["advantages"] = gae["advantages"]
        batch["returns"] = gae["returns"]

        metrics = ppo_update(
            policy=policy,
            optimizer=optimizer,
            batch=batch,
            clip_coef=clip_coef,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            update_epochs=ppo_epochs,
            minibatch_size=minibatch_size,
            max_grad_norm=max_grad_norm,
        )
        update_idx += 1

        if update_idx % log_interval == 0:
            avg_return = float(np.mean(episode_returns[-20:])) if episode_returns else 0.0
            avg_success = float(np.mean(episode_success[-20:])) if episode_success else 0.0
            avg_spl = float(np.mean(episode_spl[-20:])) if episode_spl else 0.0
            print(
                f"[train_nav] step={global_step} update={update_idx} "
                f"loss={metrics['loss']:.4f} "
                f"policy={metrics['policy_loss']:.4f} value={metrics['value_loss']:.4f} "
                f"ent={metrics['entropy']:.4f} return20={avg_return:.3f} "
                f"success20={avg_success:.3f} spl20={avg_spl:.3f}"
            )

        if global_step % ckpt_interval == 0 or global_step >= num_steps:
            ckpt_path = out_dir / f"ppo_step_{global_step}.pt"
            torch.save(
                {
                    "global_step": global_step,
                    "model_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_config": cfg,
                    "run_args": vars(args),
                },
                ckpt_path,
            )
            print(f"[train_nav] Saved checkpoint: {ckpt_path}")

    summary = {
        "steps": global_step,
        "episodes": len(episode_returns),
        "mean_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "mean_success": float(np.mean(episode_success)) if episode_success else 0.0,
        "mean_spl": float(np.mean(episode_spl)) if episode_spl else 0.0,
        "env_kind": env_kind,
    }
    summary_path = out_dir / "train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[train_nav] Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
