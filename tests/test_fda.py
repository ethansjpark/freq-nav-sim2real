"""Tests for Fourier Domain Adaptation (FDA)."""
from __future__ import annotations

import numpy as np
import pytest

from src.data.fda import _fourier_swap_py, fourier_swap


class TestFourierSwapPy:
    def test_output_shape(self, sample_image_pair):
        src, tgt = sample_image_pair
        result = _fourier_swap_py(src, tgt, beta=0.01)
        assert result.shape == src.shape

    def test_output_dtype(self, sample_image_pair):
        src, tgt = sample_image_pair
        result = _fourier_swap_py(src, tgt, beta=0.01)
        assert result.dtype == np.uint8

    def test_output_range(self, sample_image_pair):
        src, tgt = sample_image_pair
        result = _fourier_swap_py(src, tgt, beta=0.05)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_beta_zero_preserves_source(self, sample_image_pair):
        src, tgt = sample_image_pair
        result = _fourier_swap_py(src, tgt, beta=0.0)
        np.testing.assert_allclose(result.astype(float), src.astype(float), atol=1)

    def test_different_beta_gives_different_result(self, sample_image_pair):
        src, tgt = sample_image_pair
        r1 = _fourier_swap_py(src, tgt, beta=0.01)
        r2 = _fourier_swap_py(src, tgt, beta=0.1)
        assert not np.array_equal(r1, r2)

    def test_deterministic(self, sample_image_pair):
        src, tgt = sample_image_pair
        r1 = _fourier_swap_py(src, tgt, beta=0.05)
        r2 = _fourier_swap_py(src, tgt, beta=0.05)
        np.testing.assert_array_equal(r1, r2)


class TestFourierSwapDispatch:
    def test_dispatch_returns_same_shape(self, sample_image_pair):
        src, tgt = sample_image_pair
        result = fourier_swap(src, tgt, beta=0.01)
        assert result.shape == src.shape
        assert result.dtype == np.uint8
