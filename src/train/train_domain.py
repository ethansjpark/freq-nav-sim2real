from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.fda import fourier_swap
from src.data.synthetic_loader import SyntheticRealPairDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Base domain adaptation training/preprocessing loop (FDA-focused)."
    )
    parser.add_argument("--config", default="configs/fda.yaml", help="FDA YAML config path.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=-1)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--output-dir", default="experiments/domain")
    return parser.parse_args()


def _as_uint8_hwc(batch: torch.Tensor) -> np.ndarray:
    # BCHW [0,1] -> BHWC [0,255] uint8
    arr = (batch.clamp(0.0, 1.0).permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
    return arr


def _fft_highfreq_energy(img_bhwc: np.ndarray, cutoff_ratio: float = 0.1) -> float:
    # Mean high-frequency spectral magnitude over a batch of RGB images.
    b, h, w, _ = img_bhwc.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cy, cx = h // 2, w // 2
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    cutoff = min(h, w) * cutoff_ratio
    hf_mask = (dist >= cutoff).astype(np.float32)

    energies: List[float] = []
    for i in range(b):
        fft_img = np.fft.fftshift(np.fft.fft2(img_bhwc[i], axes=(0, 1)))
        mag = np.abs(fft_img).mean(axis=2)  # average over channels
        energies.append(float((mag * hf_mask).mean()))
    return float(np.mean(energies)) if energies else 0.0


def run_epoch(
    loader: DataLoader,
    beta: float,
    out_dir: Path,
    epoch_idx: int,
    save_images: bool,
    max_batches: int,
) -> Dict[str, float]:
    l1_values: List[float] = []
    hf_src_values: List[float] = []
    hf_adapt_values: List[float] = []
    steps = 0

    epoch_img_dir = out_dir / "samples" / f"epoch_{epoch_idx:03d}"
    if save_images:
        epoch_img_dir.mkdir(parents=True, exist_ok=True)

    for batch_idx, (src_batch, tgt_batch, src_paths, _tgt_paths) in enumerate(loader):
        if 0 <= max_batches <= batch_idx:
            break

        src_uint8 = _as_uint8_hwc(src_batch)
        tgt_uint8 = _as_uint8_hwc(tgt_batch)

        adapted = []
        for i in range(src_uint8.shape[0]):
            adapted_img = fourier_swap(src_uint8[i], tgt_uint8[i], beta=beta)
            adapted.append(adapted_img)
        adapted = np.stack(adapted, axis=0)

        l1 = float(np.mean(np.abs(src_uint8.astype(np.float32) - adapted.astype(np.float32))))
        hf_src = _fft_highfreq_energy(src_uint8)
        hf_adapt = _fft_highfreq_energy(adapted)

        l1_values.append(l1)
        hf_src_values.append(hf_src)
        hf_adapt_values.append(hf_adapt)
        steps += 1

        if save_images:
            for i in range(adapted.shape[0]):
                stem = Path(src_paths[i]).stem
                out_path = epoch_img_dir / f"{batch_idx:04d}_{i:02d}_{stem}.jpg"
                cv2.imwrite(str(out_path), cv2.cvtColor(adapted[i], cv2.COLOR_RGB2BGR))

    metrics = {
        "steps": float(steps),
        "mean_l1_src_to_adapt": float(np.mean(l1_values)) if l1_values else 0.0,
        "mean_hf_energy_src": float(np.mean(hf_src_values)) if hf_src_values else 0.0,
        "mean_hf_energy_adapt": float(np.mean(hf_adapt_values)) if hf_adapt_values else 0.0,
    }
    return metrics


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = yaml.safe_load(Path(args.config).read_text())
    source_dir = cfg["source_dir"]
    target_dir = cfg["target_dir"]
    beta = float(cfg.get("beta", 0.01))
    image_size = int(cfg.get("image_size", 224))

    dataset = SyntheticRealPairDataset(
        source_root=source_dir,
        target_root=target_dir,
        image_size=image_size,
        as_tensor=True,
        return_paths=True,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: List[Dict[str, float]] = []
    for epoch in range(args.epochs):
        metrics = run_epoch(
            loader=loader,
            beta=beta,
            out_dir=out_dir,
            epoch_idx=epoch,
            save_images=args.save_images,
            max_batches=args.max_batches,
        )
        all_metrics.append(metrics)
        print(
            f"[train_domain] epoch={epoch + 1}/{args.epochs} "
            f"steps={int(metrics['steps'])} "
            f"l1={metrics['mean_l1_src_to_adapt']:.3f} "
            f"hf_src={metrics['mean_hf_energy_src']:.3f} "
            f"hf_adapt={metrics['mean_hf_energy_adapt']:.3f}"
        )

    summary = {
        "config": args.config,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "beta": beta,
        "image_size": image_size,
        "num_samples": len(dataset),
        "history": all_metrics,
    }
    summary_path = out_dir / "domain_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[train_domain] wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
