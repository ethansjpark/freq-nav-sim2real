from habitat_sim.utils.common import d3_40_colors_rgb
import habitat
import numpy as np

class HabitatWrapper:
    def __init__(self, config_path):
        self.cfg = habitat.get_config(config_path)
        self.env = habitat.Env(config=self.cfg)

    def reset(self):
        obs = self.env.reset()
        return obs["rgb"]

    def step(self, action):
        out = self.env.step(action)
        return out.observation["rgb"], out.reward, out.done
