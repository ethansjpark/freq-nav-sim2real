"""Tests for MockPointNavEnv."""
from __future__ import annotations

import numpy as np
import pytest

from src.habitat_env.mock_env import MockPointNavConfig, MockPointNavEnv, make_mock_env


@pytest.fixture
def env():
    cfg = MockPointNavConfig(image_size=32, max_episode_steps=50, seed=42)
    return MockPointNavEnv(cfg)


class TestMockPointNavEnv:
    def test_reset_returns_rgb(self, env):
        obs = env.reset()
        assert obs.shape == (32, 32, 3)
        assert obs.dtype == np.uint8

    def test_step_returns_tuple(self, env):
        env.reset()
        obs, reward, done, info = env.step(0)
        assert obs.shape == (32, 32, 3)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_info_keys(self, env):
        env.reset()
        _, _, _, info = env.step(0)
        assert "success" in info
        assert "path_length" in info
        assert "shortest_path" in info
        assert "distance_to_goal" in info

    def test_three_actions_valid(self, env):
        env.reset()
        for action in [0, 1, 2]:
            obs, _, _, _ = env.step(action)
            assert obs.shape == (32, 32, 3)

    def test_invalid_action_raises(self, env):
        env.reset()
        with pytest.raises(ValueError, match="Unsupported action"):
            env.step(99)

    def test_episode_terminates(self, env):
        env.reset()
        done = False
        steps = 0
        while not done and steps < 200:
            _, _, done, _ = env.step(0)
            steps += 1
        assert done

    def test_max_steps_terminates(self):
        cfg = MockPointNavConfig(image_size=16, max_episode_steps=5, seed=0)
        env = MockPointNavEnv(cfg)
        env.reset()
        for _ in range(4):
            _, _, done, _ = env.step(1)
            if done:
                break
        if not done:
            _, _, done, _ = env.step(1)
            assert done

    def test_success_reward(self):
        cfg = MockPointNavConfig(
            image_size=16, max_episode_steps=500, success_distance=100.0, seed=0
        )
        env = MockPointNavEnv(cfg)
        env.reset()
        _, reward, done, info = env.step(0)
        if info["distance_to_goal"] <= 100.0:
            assert reward == 1.0
            assert info["success"] == 1.0

    def test_seed_determinism(self):
        cfg = MockPointNavConfig(image_size=16, seed=7)
        env1 = MockPointNavEnv(cfg)
        env2 = MockPointNavEnv(cfg)
        obs1 = env1.reset()
        obs2 = env2.reset()
        np.testing.assert_array_equal(obs1, obs2)

        for _ in range(5):
            o1, r1, d1, i1 = env1.step(0)
            o2, r2, d2, i2 = env2.step(0)
            np.testing.assert_array_equal(o1, o2)
            assert r1 == r2
            assert d1 == d2


class TestMakeMockEnv:
    def test_factory_returns_working_env(self):
        cfg = MockPointNavConfig(image_size=16, seed=0)
        env = make_mock_env(cfg)
        obs = env.reset()
        assert obs.shape[0] == 16
        assert obs.shape[1] == 16
        assert obs.shape[2] == 3
