from __future__ import annotations

import random
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

VALID_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _list_images(root: Path, recursive: bool = False) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Image root does not exist: {root}")

    if recursive:
        paths = [p for p in root.rglob("*") if p.suffix.lower() in VALID_EXTS]
    else:
        paths = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
    return sorted(paths)


def _load_image(path: Path, image_size: int | None, rgb: bool) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")

    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if image_size is not None:
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return img


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


class SyntheticDataset(Dataset):
    """
    Dataset for loading synthetic navigation frames.
    """

    def __init__(
        self,
        root: str | Path,
        image_size: int | None = None,
        recursive: bool = False,
        as_tensor: bool = False,
        rgb: bool = True,
        return_path: bool = False,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.as_tensor = as_tensor
        self.rgb = rgb
        self.return_path = return_path
        self.paths = _list_images(self.root, recursive=recursive)
        if not self.paths:
            raise ValueError(f"No images found in {self.root}")

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = _load_image(path, image_size=self.image_size, rgb=self.rgb)
        sample = _to_tensor(img) if self.as_tensor else img
        if self.return_path:
            return sample, str(path)
        return sample

    def __len__(self) -> int:
        return len(self.paths)


class SyntheticRealPairDataset(Dataset):
    """
    Paired source/target dataset for domain adaptation training.

    Source images are indexed deterministically; target images are sampled
    uniformly at random at each access, matching common FDA preprocessing style.
    """

    def __init__(
        self,
        source_root: str | Path,
        target_root: str | Path,
        image_size: int | None = None,
        recursive: bool = False,
        as_tensor: bool = False,
        rgb: bool = True,
        return_paths: bool = False,
        seed: int = 0,
    ) -> None:
        self.source_root = Path(source_root)
        self.target_root = Path(target_root)
        self.image_size = image_size
        self.as_tensor = as_tensor
        self.rgb = rgb
        self.return_paths = return_paths
        self.source_paths = _list_images(self.source_root, recursive=recursive)
        self.target_paths = _list_images(self.target_root, recursive=recursive)
        if not self.source_paths:
            raise ValueError(f"No source images found in {self.source_root}")
        if not self.target_paths:
            raise ValueError(f"No target images found in {self.target_root}")
        self._rng = random.Random(seed)

    def __getitem__(self, idx: int):
        src_path = self.source_paths[idx]
        tgt_path = self._rng.choice(self.target_paths)

        src = _load_image(src_path, image_size=self.image_size, rgb=self.rgb)
        tgt = _load_image(tgt_path, image_size=self.image_size, rgb=self.rgb)

        if self.as_tensor:
            src_out = _to_tensor(src)
            tgt_out = _to_tensor(tgt)
        else:
            src_out = src
            tgt_out = tgt

        if self.return_paths:
            return src_out, tgt_out, str(src_path), str(tgt_path)
        return src_out, tgt_out

    def __len__(self) -> int:
        return len(self.source_paths)


def collect_image_paths(roots: Sequence[str | Path], recursive: bool = False) -> List[str]:
    """Utility for scripts that need explicit image-file manifests."""
    paths: List[str] = []
    for root in roots:
        for path in _list_images(Path(root), recursive=recursive):
            paths.append(str(path))
    return sorted(paths)
