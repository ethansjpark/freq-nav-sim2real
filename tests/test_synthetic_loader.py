"""Tests for synthetic data loading."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.synthetic_loader import (
    SyntheticDataset,
    SyntheticRealPairDataset,
    _list_images,
    _load_image,
    _to_tensor,
    collect_image_paths,
)


@pytest.fixture
def image_dir(tmp_path):
    """Create a directory with dummy images."""
    import cv2

    for i in range(5):
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"img_{i:02d}.png"), img)
    return tmp_path


@pytest.fixture
def paired_dirs(tmp_path):
    """Create source and target image directories."""
    import cv2

    src = tmp_path / "source"
    tgt = tmp_path / "target"
    src.mkdir()
    tgt.mkdir()
    for i in range(3):
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        cv2.imwrite(str(src / f"src_{i}.png"), img)
        cv2.imwrite(str(tgt / f"tgt_{i}.png"), img)
    return src, tgt


class TestListImages:
    def test_finds_images(self, image_dir):
        paths = _list_images(image_dir)
        assert len(paths) == 5

    def test_sorted(self, image_dir):
        paths = _list_images(image_dir)
        names = [p.name for p in paths]
        assert names == sorted(names)

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _list_images(tmp_path / "nonexistent")

    def test_ignores_non_image_files(self, image_dir):
        (image_dir / "readme.txt").write_text("not an image")
        paths = _list_images(image_dir)
        assert len(paths) == 5


class TestLoadImage:
    def test_loads_rgb(self, image_dir):
        from pathlib import Path

        paths = _list_images(image_dir)
        img = _load_image(paths[0], image_size=None, rgb=True)
        assert img.ndim == 3
        assert img.shape[2] == 3

    def test_resizes(self, image_dir):
        paths = _list_images(image_dir)
        img = _load_image(paths[0], image_size=16, rgb=True)
        assert img.shape[:2] == (16, 16)


class TestToTensor:
    def test_shape_and_range(self):
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        t = _to_tensor(img)
        assert t.shape == (3, 32, 32)
        assert t.min() >= 0.0
        assert t.max() <= 1.0
        assert t.dtype == torch.float32


class TestSyntheticDataset:
    def test_length(self, image_dir):
        ds = SyntheticDataset(image_dir)
        assert len(ds) == 5

    def test_returns_numpy(self, image_dir):
        ds = SyntheticDataset(image_dir)
        sample = ds[0]
        assert isinstance(sample, np.ndarray)

    def test_returns_tensor(self, image_dir):
        ds = SyntheticDataset(image_dir, as_tensor=True)
        sample = ds[0]
        assert isinstance(sample, torch.Tensor)
        assert sample.shape[0] == 3

    def test_returns_path(self, image_dir):
        ds = SyntheticDataset(image_dir, return_path=True)
        sample, path = ds[0]
        assert isinstance(path, str)

    def test_empty_dir_raises(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(ValueError, match="No images"):
            SyntheticDataset(tmp_path / "empty")


class TestSyntheticRealPairDataset:
    def test_length(self, paired_dirs):
        src, tgt = paired_dirs
        ds = SyntheticRealPairDataset(src, tgt)
        assert len(ds) == 3

    def test_returns_pair(self, paired_dirs):
        src, tgt = paired_dirs
        ds = SyntheticRealPairDataset(src, tgt)
        s, t = ds[0]
        assert isinstance(s, np.ndarray)
        assert isinstance(t, np.ndarray)

    def test_returns_paths(self, paired_dirs):
        src, tgt = paired_dirs
        ds = SyntheticRealPairDataset(src, tgt, return_paths=True)
        s, t, sp, tp = ds[0]
        assert isinstance(sp, str)
        assert isinstance(tp, str)

    def test_tensor_mode(self, paired_dirs):
        src, tgt = paired_dirs
        ds = SyntheticRealPairDataset(src, tgt, as_tensor=True)
        s, t = ds[0]
        assert isinstance(s, torch.Tensor)
        assert isinstance(t, torch.Tensor)


class TestCollectImagePaths:
    def test_collects_from_multiple_roots(self, paired_dirs):
        src, tgt = paired_dirs
        paths = collect_image_paths([src, tgt])
        assert len(paths) == 6
