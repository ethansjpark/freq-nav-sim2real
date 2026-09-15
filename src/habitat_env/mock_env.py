from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

try:
    import mock_env_cpp as _mock_env_cpp
except ImportError:
    _mock_env_cpp = None


@dataclass
class MockPointNavConfig:
    image_size: int = 224
    max_episode_steps: int = 200
    success_distance: float = 0.2
    step_size: float = 0.15
    turn_angle_deg: float = 15.0
    world_extent: float = 5.0
    seed: int = 0


def make_mock_env(cfg: MockPointNavConfig):
    """Return C++ env when available, otherwise Python."""
    if _mock_env_cpp is not None:
        return _mock_env_cpp.MockPointNavEnv(
            image_size=cfg.image_size,
            max_episode_steps=cfg.max_episode_steps,
            success_distance=cfg.success_distance,
            step_size=cfg.step_size,
            turn_angle_deg=cfg.turn_angle_deg,
            world_extent=cfg.world_extent,
            seed=cfg.seed,
        )
    return MockPointNavEnv(cfg)


class MockPointNavEnv:
    """
    Lightweight PointNav-like environment for local smoke testing.
    It preserves the 3-action interface:
      0 -> forward, 1 -> turn left, 2 -> turn right
    """

    def __init__(self, cfg: MockPointNavConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.step_count = 0
        self.path_length = 0.0
        self._pos = np.zeros(2, dtype=np.float32)
        self._goal = np.zeros(2, dtype=np.float32)
        self._heading_rad = 0.0
        self._shortest_path = 0.0

    def _sample_pos(self) -> np.ndarray:
        lo = -self.cfg.world_extent
        hi = self.cfg.world_extent
        return self.rng.uniform(lo, hi, size=(2,)).astype(np.float32)

    def _distance_to_goal(self) -> float:
        return float(np.linalg.norm(self._goal - self._pos))

    def _render_obs(self) -> np.ndarray:
        # Encodes relative goal direction + distance into an RGB frame.
        h = w = self.cfg.image_size
        img = np.zeros((h, w, 3), dtype=np.uint8)

        rel = self._goal - self._pos
        dist = max(1e-6, np.linalg.norm(rel))
        rel_angle = np.arctan2(rel[1], rel[0]) - self._heading_rad
        rel_angle = (rel_angle + np.pi) % (2 * np.pi) - np.pi

        x = int(((rel_angle / np.pi) * 0.5 + 0.5) * (w - 1))
        y = int((min(dist / (2 * self.cfg.world_extent), 1.0)) * (h - 1))

        # Horizontal stripe at y encodes normalized distance.
        img[y : min(y + 4, h), :, 1] = 160
        # Vertical stripe at x encodes relative bearing.
        img[:, x : min(x + 4, w), 2] = 200

        # Goal marker brightness increases when close.
        proximity = max(0.0, 1.0 - dist / (2 * self.cfg.world_extent))
        img[:, :, 0] = int(255 * proximity)
        return img

    def reset(self) -> np.ndarray:
        self.step_count = 0
        self.path_length = 0.0
        self._pos = self._sample_pos()
        self._goal = self._sample_pos()
        self._heading_rad = float(self.rng.uniform(-np.pi, np.pi))
        self._shortest_path = self._distance_to_goal()
        return self._render_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        self.step_count += 1
        prev_pos = self._pos.copy()

        if action == 1:
            self._heading_rad += np.deg2rad(self.cfg.turn_angle_deg)
        elif action == 2:
            self._heading_rad -= np.deg2rad(self.cfg.turn_angle_deg)
        elif action == 0:
            delta = np.array(
                [np.cos(self._heading_rad), np.sin(self._heading_rad)],
                dtype=np.float32,
            )
            self._pos = self._pos + self.cfg.step_size * delta
            self._pos = np.clip(self._pos, -self.cfg.world_extent, self.cfg.world_extent)
        else:
            raise ValueError(f"Unsupported action: {action}")

        self.path_length += float(np.linalg.norm(self._pos - prev_pos))
        dist = self._distance_to_goal()
        success = dist <= self.cfg.success_distance
        done = success or self.step_count >= self.cfg.max_episode_steps

        # Sparse success reward only (no reward shaping).
        reward = 1.0 if success else 0.0
        info = {
            "success": float(success),
            "path_length": float(self.path_length),
            "shortest_path": float(self._shortest_path),
            "distance_to_goal": float(dist),
        }
        return self._render_obs(), reward, done, info
