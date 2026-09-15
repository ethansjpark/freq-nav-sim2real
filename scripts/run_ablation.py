"""
Ablation experiment runner.

Trains and evaluates PPO policies across frequency conditions:
  - baseline (no frequency adaptation)
  - freq_adapt with varying radii
  - FDA-preprocessed images at varying beta values

Each condition gets its own output directory with checkpoints, eval
results, and TensorBoard logs. A combined summary is written at the end.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Condition:
    name: str
    freq_enabled: bool = False
    freq_radius: int = 16
    freq_noise_std: float = 1.0
    extra_train_args: List[str] = field(default_factory=list)
    extra_eval_args: List[str] = field(default_factory=list)


DEFAULT_CONDITIONS = [
    Condition(name="baseline"),
    Condition(name="freq_r8", freq_enabled=True, freq_radius=8),
    Condition(name="freq_r16", freq_enabled=True, freq_radius=16),
    Condition(name="freq_r32", freq_enabled=True, freq_radius=32),
    Condition(name="freq_r16_noise05", freq_enabled=True, freq_radius=16, freq_noise_std=0.5),
    Condition(name="freq_r16_noise20", freq_enabled=True, freq_radius=16, freq_noise_std=2.0),
]


def _write_condition_config(base_cfg_path: Path, condition: Condition, out_path: Path) -> None:
    """Write a per-condition training config with frequency_adapt overrides."""
    import yaml

    cfg = yaml.safe_load(base_cfg_path.read_text())
    cfg["frequency_adapt"] = {
        "enabled": condition.freq_enabled,
        "radius": condition.freq_radius,
        "noise_std": condition.freq_noise_std,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.dump(cfg, default_flow_style=False))


def _run_command(cmd: List[str], label: str) -> int:
    print(f"\n{'='*60}")
    print(f"[ablation] Running: {label}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"[ablation] WARNING: {label} exited with code {result.returncode}")
    return result.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frequency ablation experiments.")
    parser.add_argument("--config", default="configs/training.yaml", help="Base training config.")
    parser.add_argument("--output-root", default="experiments/ablation")
    parser.add_argument("--num-steps", type=int, default=50000, help="Training steps per condition.")
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--mock-env", action="store_true", help="Force mock env.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--conditions", nargs="*", default=None,
                        help="Subset of condition names to run (default: all).")
    parser.add_argument("--skip-train", action="store_true", help="Skip training, eval only.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip eval, train only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = Path(args.config)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    conditions = DEFAULT_CONDITIONS
    if args.conditions:
        allowed = set(args.conditions)
        conditions = [c for c in conditions if c.name in allowed]
        if not conditions:
            print(f"[ablation] No matching conditions for: {args.conditions}")
            sys.exit(1)

    all_results: Dict[str, dict] = {}

    for cond in conditions:
        cond_dir = output_root / cond.name
        cond_dir.mkdir(parents=True, exist_ok=True)

        cond_cfg_path = cond_dir / "training.yaml"
        _write_condition_config(base_cfg, cond, cond_cfg_path)

        # --- Train ---
        if not args.skip_train:
            train_cmd = [
                sys.executable, "src/train/train_nav.py",
                "--config", str(cond_cfg_path),
                "--output-dir", str(cond_dir),
                "--num-steps", str(args.num_steps),
                "--seed", str(args.seed),
            ]
            if args.mock_env:
                train_cmd.append("--mock-env")
            if args.tensorboard:
                train_cmd.extend(["--tensorboard", "--tb-dir", str(cond_dir / "tb")])
            train_cmd.extend(cond.extra_train_args)

            _run_command(train_cmd, f"train/{cond.name}")

        # --- Find latest checkpoint ---
        ckpts = sorted(cond_dir.glob("ppo_step_*.pt"))
        if not ckpts:
            print(f"[ablation] No checkpoint found for {cond.name}, skipping eval.")
            continue
        latest_ckpt = ckpts[-1]

        # --- Eval ---
        if not args.skip_eval:
            eval_json = cond_dir / "eval_results.json"
            eval_cmd = [
                sys.executable, "src/eval/eval_nav.py",
                "--checkpoint", str(latest_ckpt),
                "--config", str(cond_cfg_path),
                "--episodes", str(args.eval_episodes),
                "--seed", str(args.seed),
                "--output-json", str(eval_json),
            ]
            if args.mock_env:
                eval_cmd.append("--mock-env")
            eval_cmd.extend(cond.extra_eval_args)

            _run_command(eval_cmd, f"eval/{cond.name}")

            if eval_json.exists():
                all_results[cond.name] = json.loads(eval_json.read_text())

    # --- Combined summary ---
    summary_path = output_root / "ablation_summary.json"
    summary = {
        "num_steps": args.num_steps,
        "eval_episodes": args.eval_episodes,
        "seed": args.seed,
        "conditions": {name: res for name, res in all_results.items()},
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[ablation] Summary written to {summary_path}")

    # --- Print comparison table ---
    if all_results:
        print(f"\n{'Condition':<25} {'SR':>8} {'SPL':>8} {'Return':>8} {'Length':>8}")
        print("-" * 60)
        for name, res in all_results.items():
            print(
                f"{name:<25} "
                f"{res.get('success_rate', 0):.3f}   "
                f"{res.get('mean_spl', 0):.3f}   "
                f"{res.get('mean_return', 0):.3f}   "
                f"{res.get('mean_length', 0):.1f}"
            )


if __name__ == "__main__":
    main()
