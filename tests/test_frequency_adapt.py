"""Tests for frequency-domain perturbation (freq_adapt)."""
from __future__ import annotations

import torch
import pytest

from src.train.frequency_adapt import _freq_adapt_py, _radial_lowpass_mask, freq_adapt


class TestRadialLowpassMask:
    def test_shape(self):
        mask = _radial_lowpass_mask(64, 64, radius=16, device=torch.device("cpu"))
        assert mask.shape == (64, 64)

    def test_center_is_one(self):
        mask = _radial_lowpass_mask(64, 64, radius=16, device=torch.device("cpu"))
        assert mask[32, 32] == 1.0

    def test_corners_are_zero(self):
        mask = _radial_lowpass_mask(64, 64, radius=10, device=torch.device("cpu"))
        assert mask[0, 0] == 0.0
        assert mask[0, 63] == 0.0
        assert mask[63, 0] == 0.0
        assert mask[63, 63] == 0.0

    def test_values_binary(self):
        mask = _radial_lowpass_mask(32, 32, radius=8, device=torch.device("cpu"))
        unique = torch.unique(mask)
        assert all(v in (0.0, 1.0) for v in unique.tolist())


class TestFreqAdaptPy:
    def test_output_shape(self, sample_obs_tensor):
        result = _freq_adapt_py(sample_obs_tensor, radius=8, noise_std=1.0)
        assert result.shape == sample_obs_tensor.shape

    def test_output_range(self, sample_obs_tensor):
        result = _freq_adapt_py(sample_obs_tensor, radius=8, noise_std=1.0)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_radius_zero_returns_input(self, sample_obs_tensor):
        result = _freq_adapt_py(sample_obs_tensor, radius=0, noise_std=1.0)
        torch.testing.assert_close(result, sample_obs_tensor)

    def test_rejects_non_4d(self):
        bad = torch.rand(3, 64, 64)
        with pytest.raises(ValueError, match="Expected BCHW"):
            _freq_adapt_py(bad)

    def test_different_radius_gives_different_result(self, sample_obs_tensor):
        torch.manual_seed(0)
        r1 = _freq_adapt_py(sample_obs_tensor, radius=4)
        torch.manual_seed(0)
        r2 = _freq_adapt_py(sample_obs_tensor, radius=16)
        assert not torch.equal(r1, r2)


class TestFreqAdaptDispatch:
    def test_dispatch_returns_correct_shape(self, sample_obs_tensor):
        result = freq_adapt(sample_obs_tensor, radius=8)
        assert result.shape == sample_obs_tensor.shape
