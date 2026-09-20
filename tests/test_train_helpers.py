"""Tests for training helper functions."""
from __future__ import annotations

import numpy as np
import pytest
import torch


class TestToTorchObs:
    """Test _to_torch_obs from train_nav.py."""

    def test_shape(self):
        from src.train.train_nav import _to_torch_obs

        obs = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        t = _to_torch_obs(obs, device=torch.device("cpu"), image_size=32)
        assert t.shape == (1, 3, 32, 32)

    def test_range(self):
        from src.train.train_nav import _to_torch_obs

        obs = np.full((64, 64, 3), 255, dtype=np.uint8)
        t = _to_torch_obs(obs, device=torch.device("cpu"), image_size=32)
        assert t.max() <= 1.0
        assert t.min() >= 0.0

    def test_rejects_non_3d(self):
        from src.train.train_nav import _to_torch_obs

        obs = np.zeros((64, 64), dtype=np.uint8)
        with pytest.raises(ValueError, match="HWC"):
            _to_torch_obs(obs, device=torch.device("cpu"), image_size=32)


class TestAsUint8HWC:
    """Test _as_uint8_hwc from train_domain.py."""

    def test_conversion(self):
        from src.train.train_domain import _as_uint8_hwc

        batch = torch.rand(2, 3, 16, 16)
        result = _as_uint8_hwc(batch)
        assert result.shape == (2, 16, 16, 3)
        assert result.dtype == np.uint8
        assert result.max() <= 255
        assert result.min() >= 0

    def test_clamps_out_of_range(self):
        from src.train.train_domain import _as_uint8_hwc

        batch = torch.tensor([[[[2.0]], [[-1.0]], [[0.5]]]])
        result = _as_uint8_hwc(batch)
        assert result[0, 0, 0, 0] == 255
        assert result[0, 0, 0, 1] == 0


class TestFFTHighfreqEnergy:
    """Test _fft_highfreq_energy from train_domain.py."""

    def test_returns_float(self):
        from src.train.train_domain import _fft_highfreq_energy

        img = np.random.randint(0, 256, (2, 32, 32, 3), dtype=np.uint8)
        result = _fft_highfreq_energy(img)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_uniform_image_low_energy(self):
        from src.train.train_domain import _fft_highfreq_energy

        img = np.full((1, 32, 32, 3), 128, dtype=np.uint8)
        result = _fft_highfreq_energy(img)
        assert result < 1.0

    def test_noisy_image_higher_energy(self):
        from src.train.train_domain import _fft_highfreq_energy

        uniform = np.full((1, 32, 32, 3), 128, dtype=np.uint8)
        noisy = np.random.randint(0, 256, (1, 32, 32, 3), dtype=np.uint8)
        e_uniform = _fft_highfreq_energy(uniform)
        e_noisy = _fft_highfreq_energy(noisy)
        assert e_noisy > e_uniform


class TestConfigHelpers:
    def test_as_float(self):
        from src.train.train_nav import _as_float

        assert _as_float({"lr": 0.001}, "lr", 0.1) == pytest.approx(0.001)
        assert _as_float({}, "lr", 0.1) == pytest.approx(0.1)

    def test_as_int(self):
        from src.train.train_nav import _as_int

        assert _as_int({"steps": 1000}, "steps", 500) == 1000
        assert _as_int({}, "steps", 500) == 500
