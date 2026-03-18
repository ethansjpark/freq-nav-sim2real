from __future__ import annotations

from typing import Any, Dict, Tuple

try:
    import habitat
except Exception:  # pragma: no cover - optional dependency
    habitat = None


class HabitatWrapper:
    """Thin Habitat-Lab wrapper exposing RGB + scalar reward/done."""

    def __init__(self, config_path: str):
        if habitat is None:
            raise ImportError(
                "Habitat is not installed or failed to import. "
                "Install habitat-lab/habitat-sim to use HabitatWrapper."
            )
        self.cfg = habitat.get_config(config_path)
        self.env = habitat.Env(config=self.cfg)

    def _extract_rgb(self, obs: Dict[str, Any]) -> Any:
        if "rgb" not in obs:
            raise KeyError("Expected `rgb` key in Habitat observation.")
        return obs["rgb"]

    def reset(self) -> Any:
        obs = self.env.reset()
        return self._extract_rgb(obs)

    def step(self, action: int) -> Tuple[Any, float, bool, Dict[str, Any]]:
        out = self.env.step(action)

        if isinstance(out, tuple):
            # gym / gymnasium style: (obs, reward, done, info) or (obs, reward, terminated, truncated, info)
            if len(out) == 4:
                obs, reward, done, info = out
            elif len(out) == 5:
                obs, reward, terminated, truncated, info = out
                done = bool(terminated or truncated)
            else:
                raise ValueError(f"Unexpected step tuple length: {len(out)}")
            return self._extract_rgb(obs), float(reward), bool(done), dict(info)

        # Some Habitat APIs return a structured object.
        if hasattr(out, "observation") and hasattr(out, "reward") and hasattr(out, "done"):
            info = getattr(out, "info", {})
            return self._extract_rgb(out.observation), float(out.reward), bool(out.done), dict(info)

        raise TypeError("Unsupported step() return format from Habitat environment.")
