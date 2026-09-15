"""Tests for SPL metric computation."""
from __future__ import annotations

import pytest

from src.utils.metrics import compute_spl


class TestComputeSPL:
    def test_failure_returns_zero(self):
        assert compute_spl(success=False, path_len=10.0, shortest_path=5.0) == 0.0

    def test_optimal_path(self):
        spl = compute_spl(success=True, path_len=5.0, shortest_path=5.0)
        assert spl == pytest.approx(1.0)

    def test_longer_path(self):
        spl = compute_spl(success=True, path_len=10.0, shortest_path=5.0)
        assert spl == pytest.approx(0.5)

    def test_very_long_path(self):
        spl = compute_spl(success=True, path_len=100.0, shortest_path=5.0)
        assert spl == pytest.approx(0.05)

    def test_path_shorter_than_shortest(self):
        spl = compute_spl(success=True, path_len=3.0, shortest_path=5.0)
        assert spl == pytest.approx(1.0)

    def test_zero_shortest_path_raises(self):
        with pytest.raises(ZeroDivisionError):
            compute_spl(success=True, path_len=0.0, shortest_path=0.0)
