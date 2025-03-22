from isaaclab.utils import configclass
from isaaclab.terrains.height_field import hf_terrains_cfg
from isaaclabex.terrains.height_field import parkour_hf_terrains
from dataclasses import MISSING


@configclass
class ParkourHurdleTerrainCfg(hf_terrains_cfg.HfRandomUniformTerrainCfg):
    """Configuration for a stepping stones height field terrain."""

    function = parkour_hf_terrains.parkour_hurdle_terrain

    num_stones: int = MISSING   #= 6

    hurdle_y: float = MISSING  # = 2.5
    hurdle_x_range: tuple[float, float, float] = MISSING  # = [0.2, 0.4]
    hurdle_z_range: tuple[float, float, float] = MISSING  # = (0.15, 0.3)
    hurdle_z_noise: tuple[float, float, float] = MISSING  # = (-0.1, 0.1)
    hurdle_distance_x_range: tuple[float, float, float] = MISSING  # = [1.2, 2.2]
    hurdle_distance_y_range: tuple[float, float, float] = MISSING  # = [-0.4, 0.4]

    platform_space: float = MISSING  # = 2.5
    platform_height: float = MISSING  # = 0
