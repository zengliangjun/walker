from __future__ import annotations
from typing import TYPE_CHECKING
import scipy.interpolate as interpolate
from isaaclab.terrains.height_field import hf_terrains_cfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
import numpy as np

if TYPE_CHECKING:
    from isaaclabex.terrains.height_field import parkour_hf_cfg

import random

def random_uniform_terrain_copy(difficulty: float, cfg: hf_terrains_cfg.HfRandomUniformTerrainCfg) -> np.ndarray:
    """Generate a terrain with height sampled uniformly from a specified range.

    .. image:: ../../_static/terrains/height_field/random_uniform_terrain.jpg
       :width: 40%
       :align: center

    Note:
        The :obj:`difficulty` parameter is ignored for this terrain.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.

    Raises:
        ValueError: When the downsampled scale is smaller than the horizontal scale.
    """
    # check parameters
    # -- horizontal scale
    if cfg.downsampled_scale is None:
        cfg.downsampled_scale = cfg.horizontal_scale
    elif cfg.downsampled_scale < cfg.horizontal_scale:
        raise ValueError(
            "Downsampled scale must be larger than or equal to the horizontal scale:"
            f" {cfg.downsampled_scale} < {cfg.horizontal_scale}."
        )

    # switch parameters to discrete units
    # -- horizontal scale
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- downsampled scale
    width_downsampled = int(cfg.size[0] / cfg.downsampled_scale)
    length_downsampled = int(cfg.size[1] / cfg.downsampled_scale)
    # -- height
    height_min = int(cfg.noise_range[0] / cfg.vertical_scale)
    height_max = int(cfg.noise_range[1] / cfg.vertical_scale)
    height_step = int(cfg.noise_step / cfg.vertical_scale)

    # create range of heights possible
    height_range = np.arange(height_min, height_max + height_step, height_step)
    # sample heights randomly from the range along a grid
    height_field_downsampled = np.random.choice(height_range, size=(width_downsampled, length_downsampled))
    # create interpolation function for the sampled heights
    x = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_downsampled)
    y = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_downsampled)
    func = interpolate.RectBivariateSpline(x, y, height_field_downsampled)

    # interpolate the sampled heights to obtain the height field
    x_upsampled = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_pixels)
    y_upsampled = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_pixels)
    z_upsampled = func(x_upsampled, y_upsampled)
    # round off the interpolated heights to the nearest vertical step
    return z_upsampled


@height_field_to_mesh
def parkour_hurdle_terrain(difficulty: float, cfg: parkour_hf_cfg.ParkourHurdleTerrainCfg):

    _random_terrain = random_uniform_terrain_copy(difficulty, cfg)
    _sampled = np.zeros_like(_random_terrain)

    hurdle_x = cfg.hurdle_x_range[0] + difficulty * (cfg.hurdle_x_range[1] - cfg.hurdle_x_range[0])
    hurdle_z = cfg.hurdle_z_range[0] + difficulty * (cfg.hurdle_z_range[1] - cfg.hurdle_z_range[0])
    hurdle_z_min = hurdle_z + difficulty * cfg.hurdle_z_noise[0]
    hurdle_z_max = hurdle_z + difficulty * cfg.hurdle_z_noise[1]

    print("difficulty  ", difficulty, hurdle_x, (hurdle_z_min, hurdle_z_max))

    _goals = np.zeros((cfg.num_stones + 2, 2))
    # terrain.height_field_raw[:] = -200
    _pixX = round(cfg.size[1] / cfg.horizontal_scale)
    _mid_pix_y_ORG = round(cfg.size[0] / cfg.horizontal_scale) // 2  # length is actually y width
    mid_pix_y = _mid_pix_y_ORG
    hurdle_pix_y = round(cfg.hurdle_y / cfg.horizontal_scale) // 2 

    dis_pix_x_min = round(cfg.hurdle_distance_x_range[0] / cfg.horizontal_scale)
    dis_pix_x_max = round(cfg.hurdle_distance_x_range[1] / cfg.horizontal_scale)
    dis_pix_y_min = round(cfg.hurdle_distance_y_range[0] / cfg.horizontal_scale)
    dis_pix_y_max = round(cfg.hurdle_distance_y_range[1] / cfg.horizontal_scale)

    # half_valid_width = round(np.random.uniform(y_range[1]+0.2, y_range[1]+1) / terrain.horizontal_scale)
    #hurdle_pix_z_min = round(hurdle_z_min / cfg.vertical_scale)
    #hurdle_pix_z_max = round(hurdle_z_max / cfg.vertical_scale)

    hurdle_pix_space = round(cfg.platform_space / cfg.horizontal_scale)
    platform_pix_height = round(cfg.platform_height / cfg.vertical_scale)
    _sampled[:, :hurdle_pix_space] = platform_pix_height

    hurdle_pix_x = round(hurdle_x / cfg.horizontal_scale)
    # stone_width = round(stone_width / terrain.horizontal_scale)

    # incline_height = round(incline_height / terrain.vertical_scale)
    # last_incline_height = round(last_incline_height / terrain.vertical_scale)

    _start_x = hurdle_pix_space
    _goals[0] = [hurdle_pix_space - 1, mid_pix_y]
    #_last_dis_x = _start_x
    for i in range(cfg.num_stones):
        dis_pix_x = np.random.randint(dis_pix_x_min, dis_pix_x_max)
        dis_pix_y = np.random.randint(dis_pix_y_min, dis_pix_y_max)
        _start_x += dis_pix_x

        mid_pix_y += dis_pix_y
        _start_y = max(mid_pix_y - hurdle_pix_y, 0)
        _end_y = min(mid_pix_y + hurdle_pix_y, _mid_pix_y_ORG * 2 - 1)

        _z = hurdle_z_min + random.random() * (hurdle_z_max - hurdle_z_min)
        hurdle_pix_z = round(_z / cfg.vertical_scale)
        print("^^^^^  ", i, (_z, hurdle_pix_z), (hurdle_x, hurdle_pix_x), _pixX)

        _sampled[_start_y: _end_y, _start_x - hurdle_pix_x // 2: _start_x + hurdle_pix_x // 2] = hurdle_pix_z

        #_last_dis_x = _start_x
        _goals[i + 1] = [_start_x, mid_pix_y]

    final_dis_x = _start_x + np.random.randint(dis_pix_x_min, dis_pix_x_max)
    # import ipdb; ipdb.set_trace()
    if final_dis_x > _sampled.shape[1]:
        final_dis_x = _sampled.shape[1]# - 0.5 // terrain.horizontal_scale
    _goals[-1] = [final_dis_x, _mid_pix_y_ORG]

    _terrain = _random_terrain + _sampled
    return _terrain.astype(np.int16)
