"""
Aggregate and plot ablation results.

Reads ablation_summary.json (from run_ablation.py) and generates:
  1. Bar chart comparing SR and SPL across conditions
  2. CSV summary table
  3. Optionally, a line plot of SR/SPL vs freq_adapt radius
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_summary(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def load_individual_results(ablation_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load eval_results.json from each condition sub-directory."""
    results = {}
    for cond_dir in sorted(ablation_dir.iterdir()):
        if not cond_dir.is_dir():
            continue
        eval_json = cond_dir / "eval_results.json"
        if eval_json.exists():
            results[cond_dir.name] = json.loads(eval_json.read_text())
    return results


def write_csv(results: Dict[str, Dict[str, Any]], out_path: Path) -> None:
    fields = ["condition", "success_rate", "mean_spl", "mean_return", "mean_length", "episodes"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, res in results.items():
            writer.writerow({
                "condition": name,
                "success_rate": f"{res.get('success_rate', 0):.4f}",
                "mean_spl": f"{res.get('mean_spl', 0):.4f}",
                "mean_return": f"{res.get('mean_return', 0):.4f}",
                "mean_length": f"{res.get('mean_length', 0):.1f}",
                "episodes": res.get("episodes", 0),
            })
    print(f"[plot_results] CSV written to {out_path}")


def plot_bar_comparison(results: Dict[str, Dict[str, Any]], out_path: Path) -> None:
    if not HAS_MPL:
        print("[plot_results] matplotlib not available, skipping bar chart.")
        return

    names = list(results.keys())
    sr = [results[n].get("success_rate", 0) for n in names]
    spl = [results[n].get("mean_spl", 0) for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 5))
    bars1 = ax.bar(x - width / 2, sr, width, label="Success Rate")
    bars2 = ax.bar(x + width / 2, spl, width, label="SPL")

    ax.set_ylabel("Score")
    ax.set_title("Frequency Ablation: SR and SPL by Condition")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        h = bar.get_height()
        if h > 0.01:
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        if h > 0.01:
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"[plot_results] Bar chart saved to {out_path}")


def plot_radius_sweep(results: Dict[str, Dict[str, Any]], out_path: Path) -> None:
    """Plot SR/SPL vs radius for freq_r* conditions."""
    if not HAS_MPL:
        return

    radius_data: List[tuple] = []
    for name, res in results.items():
        if name.startswith("freq_r") and "noise" not in name:
            try:
                r = int(name.split("_r")[1])
                radius_data.append((r, res.get("success_rate", 0), res.get("mean_spl", 0)))
            except (ValueError, IndexError):
                continue

    if len(radius_data) < 2:
        return

    radius_data.sort(key=lambda t: t[0])
    radii, sr_vals, spl_vals = zip(*radius_data)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(radii, sr_vals, "o-", label="Success Rate")
    ax.plot(radii, spl_vals, "s--", label="SPL")
    ax.set_xlabel("Frequency Adapt Radius")
    ax.set_ylabel("Score")
    ax.set_title("SR / SPL vs. Frequency Cutoff Radius")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"[plot_results] Radius sweep chart saved to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate and plot ablation results.")
    parser.add_argument("--ablation-dir", default="experiments/ablation",
                        help="Root directory of ablation experiment outputs.")
    parser.add_argument("--output-dir", default="", help="Output dir for plots/CSV (default: ablation-dir).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ablation_dir = Path(args.ablation_dir)
    output_dir = Path(args.output_dir) if args.output_dir else ablation_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try summary first, fall back to scanning directories.
    summary_path = ablation_dir / "ablation_summary.json"
    if summary_path.exists():
        summary = load_summary(summary_path)
        results = summary.get("conditions", {})
    else:
        results = load_individual_results(ablation_dir)

    if not results:
        print(f"[plot_results] No results found in {ablation_dir}")
        sys.exit(1)

    print(f"[plot_results] Found {len(results)} conditions: {list(results.keys())}")

    # Print table.
    print(f"\n{'Condition':<25} {'SR':>8} {'SPL':>8} {'Return':>8} {'Length':>8}")
    print("-" * 60)
    for name, res in results.items():
        print(
            f"{name:<25} "
            f"{res.get('success_rate', 0):.3f}   "
            f"{res.get('mean_spl', 0):.3f}   "
            f"{res.get('mean_return', 0):.3f}   "
            f"{res.get('mean_length', 0):.1f}"
        )

    write_csv(results, output_dir / "ablation_results.csv")
    plot_bar_comparison(results, output_dir / "ablation_bar.png")
    plot_radius_sweep(results, output_dir / "ablation_radius_sweep.png")


if __name__ == "__main__":
    main()
