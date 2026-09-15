"""Shared fixtures for tests."""
from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def sample_image(rng):
    """Random 64x64 RGB uint8 image."""
    return rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_pair(rng):
    """Pair of random 64x64 RGB uint8 images (src, tgt)."""
    src = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    tgt = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    return src, tgt


@pytest.fixture
def sample_obs_tensor():
    """Random BCHW float tensor in [0,1] simulating a batch of observations."""
    torch.manual_seed(0)
    return torch.rand(2, 3, 64, 64)
