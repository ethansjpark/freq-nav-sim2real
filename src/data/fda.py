import yaml
import argparse
import cv2
import numpy as np
from pathlib import Path

def fourier_swap(src, tgt, beta=0.01):
    """Minimal placeholder FDA implementation."""
    # TODO: Replace with full frequency swap
    src_fft = np.fft.fft2(src)
    tgt_fft = np.fft.fft2(tgt)
    mixed_fft = (1 - beta) * src_fft + beta * tgt_fft
    mixed = np.fft.ifft2(mixed_fft).real
    return np.clip(mixed, 0, 255).astype(np.uint8)

def main(cfg):
    src_dir = Path(cfg["source_dir"])
    tgt_dir = Path(cfg["target_dir"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    tgt_imgs = list(tgt_dir.glob("*.jpg"))

    for src_path in src_dir.glob("*.jpg"):
        src = cv2.imread(str(src_path))
        tgt = cv2.imread(str(tgt_imgs[0]))  # TODO: random target sampling
        mixed = fourier_swap(src, tgt, beta=cfg["beta"])
        cv2.imwrite(str(out_dir / src_path.name), mixed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    main(cfg)
