from isaaclab.utils import configclass
from dataclasses import MISSING
from isaaclabex.terrains import ext_generator_cfg
from isaaclabex.terrains.trimesh import parkour_mesh_terrains

@configclass
class MeshRepeatedHurdleTerrainCfg(ext_generator_cfg.SubTerrainBaseCfg):
    """Configuration for a stepping stones height field terrain."""

    function = parkour_mesh_terrains.repeated_hurdles_terrain

    @configclass
    class ObjectCfg:
        """Configuration of repeated objects."""

        num_objects: int = MISSING
        """The number of objects to add to the terrain."""
        width: float = MISSING  # X
        length: float = MISSING  # Y
        height: float = MISSING  # Z

        distance_x: float = MISSING
        distance_y: float = MISSING

    platform_x_space: float = MISSING

    object_params_start: ObjectCfg = MISSING
    """The object curriculum parameters at the start of the curriculum."""
    object_params_end: ObjectCfg = MISSING
    """The object curriculum parameters at the end of the curriculum."""
    max_height_noise: float = 0.0
