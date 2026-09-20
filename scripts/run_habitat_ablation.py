"""
Run frequency ablation on Habitat-Sim with real 3D scenes.

Usage (from conda 'habitat' env):
    python scripts/run_habitat_ablation.py \
        --num-steps 50000 --eval-episodes 50 --seed 42

Trains a fresh PPO policy per condition on PointNav with the habitat test
scenes, then evaluates each. Results go to experiments/habitat_ablation/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.encoder import VisualEncoder
from src.models.policy import Policy
from src.train.frequency_adapt import freq_adapt
from src.train.ppo_utils import RolloutBuffer, compute_gae, ppo_update
from src.utils.metrics import compute_spl


@dataclass
class Condition:
    name: str
    freq_enabled: bool = False
    freq_radius: int = 16
    freq_noise_std: float = 1.0


CONDITIONS = [
    Condition("baseline"),
    Condition("freq_r8", freq_enabled=True, freq_radius=8),
    Condition("freq_r16", freq_enabled=True, freq_radius=16),
    Condition("freq_r32", freq_enabled=True, freq_radius=32),
    Condition("freq_r16_noise05", freq_enabled=True, freq_radius=16, freq_noise_std=0.5),
    Condition("freq_r16_noise20", freq_enabled=True, freq_radius=16, freq_noise_std=2.0),
]


def make_habitat_env(image_size: int = 256):
    """Create a Habitat PointNav env with test scenes."""
    import habitat
    from omegaconf import OmegaConf

    config = habitat.get_config(
        config_path="benchmark/nav/pointnav/pointnav_habitat_test.yaml",
    )
    with habitat.config.read_write(config):
        config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = image_size
        config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = image_size
        if hasattr(config.habitat.simulator.agents.main_agent.sim_sensors, "depth_sensor"):
            config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height = image_size
            config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width = image_size
        config.habitat.environment.max_episode_steps = 500

    env = habitat.Env(config=config)
    return env


class HabitatEnvAdapter:
    """Wraps habitat.Env to expose a simple reset/step interface.

    Habitat PointNav actions: 0=STOP, 1=MOVE_FORWARD, 2=TURN_LEFT, 3=TURN_RIGHT
    Our policy actions:       0=FORWARD, 1=TURN_LEFT, 2=TURN_RIGHT
    We map policy -> habitat and never send STOP (episode ends on success/timeout).
    """
    ACTION_MAP = {0: 1, 1: 2, 2: 3}  # policy action -> habitat action

    def __init__(self, env, image_size: int = 256):
        self.env = env
        self.image_size = image_size
        self._episode_steps = 0
        self._start_pos = None
        self._prev_dtg = None

    @staticmethod
    def _to_rgb(img: np.ndarray) -> np.ndarray:
        return img[:, :, :3] if img.shape[-1] == 4 else img

    def reset(self) -> np.ndarray:
        obs = self.env.reset()
        self._episode_steps = 0
        agent_state = self.env.sim.get_agent_state()
        self._start_pos = np.array(agent_state.position)
        metrics = self.env.get_metrics()
        self._prev_dtg = metrics.get("distance_to_goal", 0.0)
        return self._to_rgb(obs["rgb"])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        hab_action = self.ACTION_MAP.get(action, 1)
        obs = self.env.step(hab_action)
        self._episode_steps += 1

        metrics = self.env.get_metrics()
        done = self.env.episode_over

        dtg = metrics.get("distance_to_goal", float("inf"))
        success = metrics.get("success", 0.0)

        # Shaped reward: delta distance-to-goal + success bonus + slack penalty
        reward = (self._prev_dtg - dtg) - 0.01
        if success > 0.5:
            reward += 2.5
        self._prev_dtg = dtg

        agent_state = self.env.sim.get_agent_state()
        pos = np.array(agent_state.position)
        path_length = float(np.linalg.norm(pos - self._start_pos)) if self._start_pos is not None else 0.0

        info = {
            "success": float(success),
            "distance_to_goal": dtg,
            "spl": metrics.get("spl", 0.0),
            "path_length": path_length,
            "shortest_path": metrics.get("distance_to_goal", path_length),
        }
        return self._to_rgb(obs["rgb"]), reward, done, info


def to_torch_obs(obs: np.ndarray, device: torch.device, image_size: int = 224) -> torch.Tensor:
    obs_t = torch.from_numpy(obs).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    obs_t = nn.functional.interpolate(obs_t, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return obs_t.to(device)


def train_condition(
    env,
    condition: Condition,
    num_steps: int,
    device: torch.device,
    seed: int,
    out_dir: Path,
) -> Path:
    """Train PPO for one condition and return the checkpoint path."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    policy = Policy(VisualEncoder()).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    buffer = RolloutBuffer()

    rollout_steps = 256
    ppo_epochs = 4
    minibatch_size = 64
    gamma = 0.99
    gae_lambda = 0.95
    image_size = 224

    out_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    obs_np = env.reset()
    ep_return = 0.0
    ep_steps = 0
    episode_returns = []
    episode_success = []
    update_idx = 0

    while global_step < num_steps:
        buffer.clear()
        for _ in range(rollout_steps):
            obs_t = to_torch_obs(obs_np, device, image_size)
            if condition.freq_enabled:
                obs_t = freq_adapt(obs_t, radius=condition.freq_radius, noise_std=condition.freq_noise_std)

            with torch.no_grad():
                logits, value = policy(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logprob = dist.log_prob(action)

            action_int = int(action.item())
            next_obs, reward, done, info = env.step(action_int)

            buffer.add(
                obs=obs_t,
                action=action.view(1),
                logprob=logprob.view(1),
                reward=torch.tensor([reward], dtype=torch.float32, device=device),
                done=torch.tensor([float(done)], dtype=torch.float32, device=device),
                value=value.view(1),
            )

            global_step += 1
            ep_return += float(reward)
            ep_steps += 1
            obs_np = next_obs

            if done:
                episode_returns.append(ep_return)
                episode_success.append(float(info.get("success", 0)))
                obs_np = env.reset()
                ep_return = 0.0
                ep_steps = 0

            if global_step >= num_steps:
                break

        with torch.no_grad():
            next_obs_t = to_torch_obs(obs_np, device, image_size)
            if condition.freq_enabled:
                next_obs_t = freq_adapt(next_obs_t, radius=condition.freq_radius, noise_std=condition.freq_noise_std)
            _, next_value = policy(next_obs_t)

        batch = buffer.as_tensors()
        gae_out = compute_gae(
            rewards=batch["rewards"], dones=batch["dones"], values=batch["values"],
            next_value=next_value.view(1), gamma=gamma, gae_lambda=gae_lambda,
        )
        batch["advantages"] = gae_out["advantages"]
        batch["returns"] = gae_out["returns"]

        metrics = ppo_update(
            policy=policy, optimizer=optimizer, batch=batch,
            clip_coef=0.2, value_coef=0.5, entropy_coef=0.01,
            update_epochs=ppo_epochs, minibatch_size=minibatch_size, max_grad_norm=0.5,
        )
        update_idx += 1

        if update_idx % 5 == 0:
            avg_ret = float(np.mean(episode_returns[-20:])) if episode_returns else 0.0
            avg_sr = float(np.mean(episode_success[-20:])) if episode_success else 0.0
            print(
                f"  [{condition.name}] step={global_step}/{num_steps} "
                f"loss={metrics['loss']:.4f} return20={avg_ret:.3f} sr20={avg_sr:.3f}"
            )

    ckpt_path = out_dir / f"ppo_{condition.name}.pt"
    torch.save({"model_state_dict": policy.state_dict(), "global_step": global_step}, ckpt_path)
    print(f"  [{condition.name}] Saved checkpoint: {ckpt_path}")
    return ckpt_path


def eval_condition(
    env,
    condition: Condition,
    ckpt_path: Path,
    episodes: int,
    device: torch.device,
    seed: int,
) -> Dict[str, float]:
    """Evaluate a trained checkpoint and return results dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    image_size = 224

    policy = Policy(VisualEncoder()).to(device)
    policy.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
    policy.eval()

    successes, spls, returns, lengths = [], [], [], []

    for ep in range(episodes):
        obs_np = env.reset()
        done = False
        ep_return = 0.0
        ep_len = 0
        info = {}

        while not done and ep_len < 500:
            obs_t = to_torch_obs(obs_np, device, image_size)
            if condition.freq_enabled:
                obs_t = freq_adapt(obs_t, radius=condition.freq_radius, noise_std=condition.freq_noise_std)
            with torch.no_grad():
                logits, _ = policy(obs_t)
                action = torch.argmax(logits, dim=-1)
            obs_np, reward, done, info = env.step(int(action.item()))
            ep_return += float(reward)
            ep_len += 1

        success = float(info.get("success", 0))
        spl = float(info.get("spl", 0))
        successes.append(success)
        spls.append(spl)
        returns.append(ep_return)
        lengths.append(ep_len)

    results = {
        "success_rate": float(np.mean(successes)),
        "mean_spl": float(np.mean(spls)),
        "mean_return": float(np.mean(returns)),
        "mean_length": float(np.mean(lengths)),
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="Habitat-Sim frequency ablation")
    parser.add_argument("--num-steps", type=int, default=50000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="experiments/habitat_ablation")
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu")
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    conditions = CONDITIONS
    if args.conditions:
        allowed = set(args.conditions)
        conditions = [c for c in CONDITIONS if c.name in allowed]

    print(f"=== Habitat-Sim Frequency Ablation ===")
    print(f"Conditions: {[c.name for c in conditions]}")
    print(f"Steps/condition: {args.num_steps}, Eval episodes: {args.eval_episodes}")

    hab_env_raw = make_habitat_env(image_size=256)
    env = HabitatEnvAdapter(hab_env_raw, image_size=256)

    all_results = {}

    for cond in conditions:
        print(f"\n{'='*60}")
        print(f"Condition: {cond.name} (freq={cond.freq_enabled}, r={cond.freq_radius}, noise={cond.freq_noise_std})")
        print(f"{'='*60}")

        cond_dir = out_root / cond.name
        ckpt_path = cond_dir / f"ppo_{cond.name}.pt"

        if not args.skip_train:
            t0 = time.time()
            ckpt_path = train_condition(env, cond, args.num_steps, device, args.seed, cond_dir)
            print(f"  [{cond.name}] Training took {time.time() - t0:.1f}s")

        if not args.skip_eval and ckpt_path.exists():
            t0 = time.time()
            results = eval_condition(env, cond, ckpt_path, args.eval_episodes, device, args.seed)
            print(f"  [{cond.name}] Eval: SR={results['success_rate']:.3f} SPL={results['mean_spl']:.3f} "
                  f"Return={results['mean_return']:.3f} Length={results['mean_length']:.1f}")
            print(f"  [{cond.name}] Eval took {time.time() - t0:.1f}s")
            all_results[cond.name] = results

            (cond_dir / "eval_results.json").write_text(json.dumps(results, indent=2))

    summary = {
        "num_steps": args.num_steps,
        "eval_episodes": args.eval_episodes,
        "seed": args.seed,
        "env": "habitat_test_scenes",
        "conditions": all_results,
    }
    summary_path = out_root / "ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n=== Summary written to {summary_path} ===")

    if all_results:
        print(f"\n{'Condition':<25} {'SR':>8} {'SPL':>8} {'Return':>8} {'Length':>8}")
        print("-" * 60)
        for name, res in all_results.items():
            print(f"{name:<25} {res['success_rate']:.3f}   {res['mean_spl']:.3f}   "
                  f"{res['mean_return']:.3f}   {res['mean_length']:.1f}")


if __name__ == "__main__":
    main()
