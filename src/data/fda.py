import yaml
import argparse
import cv2
import random
import numpy as np
from pathlib import Path



def fourier_swap(src, tgt, beta=0.01):
    """
    Fourier Domain Adaptation (FDA)
    Swaps low-frequency amplitudes from target into source.
    """

    # FFT + shift
    src_fft = np.fft.fftshift(np.fft.fft2(src, axes=(0, 1)))
    tgt_fft = np.fft.fftshift(np.fft.fft2(tgt, axes=(0, 1)))

    h, w, _ = src.shape
    b = int(min(h, w) * beta)

    h_mid, w_mid = h // 2, w // 2

    mixed_fft = src_fft.copy()

    # Swaping low-frequency regions
    mixed_fft[
        h_mid - b:h_mid + b,
        w_mid - b:w_mid + b
    ] = tgt_fft[
        h_mid - b:h_mid + b,
        w_mid - b:w_mid + b
    ]


    # Inversing FFT
    mixed = np.fft.ifft2(np.fft.ifftshift(mixed_fft), axes=(0, 1))
    mixed = np.real(mixed)


    return np.clip(mixed, 0, 255).astype(np.uint8)

def main(cfg):
    src_dir = Path(cfg["source_dir"])
    tgt_dir = Path(cfg["target_dir"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    tgt_imgs = list(tgt_dir.glob("*.jpg"))
    assert len(tgt_imgs) > 0, "No target images found."

    for src_path in src_dir.glob("*.jpg"):
        src = cv2.imread(str(src_path))

        tgt_path = random.choice(tgt_imgs)
        tgt = cv2.imread(str(tgt_path))

        mixed = fourier_swap(src, tgt, beta=cfg["beta"])
        cv2.imwrite(str(out_dir / src_path.name), mixed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    main(cfg)