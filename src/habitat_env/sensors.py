from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class RGBSensorConfig:
    uuid: str = "rgb"
    width: int = 224
    height: int = 224
    hfov: float = 79.0
    position: List[float] | None = None
    sensor_subtype: str = "PINHOLE"

    def __post_init__(self) -> None:
        if self.position is None:
            self.position = [0.0, 1.25, 0.0]


class RGBSensor:
    """
    Base RGB sensor spec builder for Habitat(-Lab/-Sim) configs.
    """

    def __init__(self, cfg: RGBSensorConfig | None = None) -> None:
        self.cfg = cfg or RGBSensorConfig()

    def to_habitat_dict(self) -> Dict[str, Any]:
        return {
            "TYPE": "HabitatSimRGBSensor",
            "UUID": self.cfg.uuid,
            "WIDTH": int(self.cfg.width),
            "HEIGHT": int(self.cfg.height),
            "HFOV": float(self.cfg.hfov),
            "POSITION": list(self.cfg.position),
            "SENSOR_SUBTYPE": self.cfg.sensor_subtype,
        }


def inject_rgb_sensor(habitat_cfg: Dict[str, Any], sensor: RGBSensor | None = None) -> Dict[str, Any]:
    """
    Injects an RGB sensor spec into a Habitat-style config dictionary.
    Returns the same dictionary for convenience.
    """
    sensor = sensor or RGBSensor()
    simulator = habitat_cfg.setdefault("SIMULATOR", {})
    agent_0 = simulator.setdefault("AGENT_0", {})
    sensors = agent_0.setdefault("SENSORS", [])
    if "RGB_SENSOR" not in sensors:
        sensors.append("RGB_SENSOR")
    simulator["RGB_SENSOR"] = sensor.to_habitat_dict()
    return habitat_cfg
