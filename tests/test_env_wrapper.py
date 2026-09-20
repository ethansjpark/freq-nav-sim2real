"""Tests for Habitat environment wrapper."""
from __future__ import annotations

import numpy as np
import pytest

from src.habitat_env.env_wrapper import HabitatWrapper


class TestHabitatWrapper:
    def test_import_error_without_habitat(self):
        with pytest.raises(ImportError, match="Habitat is not installed"):
            HabitatWrapper("configs/habitat_pointnav.yaml")

    def test_extract_rgb_valid(self):
        wrapper = HabitatWrapper.__new__(HabitatWrapper)
        obs = {"rgb": np.zeros((64, 64, 3), dtype=np.uint8)}
        rgb = wrapper._extract_rgb(obs)
        assert rgb.shape == (64, 64, 3)

    def test_extract_rgb_missing_key(self):
        wrapper = HabitatWrapper.__new__(HabitatWrapper)
        with pytest.raises(KeyError, match="rgb"):
            wrapper._extract_rgb({"depth": np.zeros((64, 64))})
