"""Generate FDA-adapted images at multiple beta values for ablation studies."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.fda import fourier_swap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FDA at multiple beta values.")
    parser.add_argument("--config", default="configs/fda.yaml")
    parser.add_argument(
        "--betas",
        nargs="+",
        type=float,
        default=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
        help="Beta values to sweep.",
    )
    parser.add_argument("--output-root", default="data/fda_ablation")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    src_dir = Path(cfg["source_dir"])
    tgt_dir = Path(cfg["target_dir"])
    image_size = int(cfg.get("image_size", 224))

    src_paths = sorted(src_dir.glob("*.jpg"))
    tgt_paths = sorted(tgt_dir.glob("*.jpg"))
    assert len(src_paths) > 0, f"No source images in {src_dir}"
    assert len(tgt_paths) > 0, f"No target images in {tgt_dir}"

    rng = np.random.default_rng(args.seed)
    output_root = Path(args.output_root)

    for beta in args.betas:
        beta_dir = output_root / f"beta_{beta:.4f}".rstrip("0").rstrip(".")
        beta_dir.mkdir(parents=True, exist_ok=True)

        for src_path in src_paths:
            src = cv2.imread(str(src_path))
            if src is None:
                continue
            src = cv2.resize(src, (image_size, image_size))

            tgt_path = tgt_paths[rng.integers(len(tgt_paths))]
            tgt = cv2.imread(str(tgt_path))
            if tgt is None:
                continue
            tgt = cv2.resize(tgt, (image_size, image_size))

            adapted = fourier_swap(src, tgt, beta=beta)
            cv2.imwrite(str(beta_dir / src_path.name), adapted)

        print(f"[run_fda_multi] beta={beta:.4f} -> {beta_dir} ({len(src_paths)} images)")

    print(f"[run_fda_multi] Done. Output root: {output_root}")


if __name__ == "__main__":
    main()
