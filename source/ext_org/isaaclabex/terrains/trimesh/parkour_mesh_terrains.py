from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import trimesh
import random
import isaaclab.terrains.trimesh.utils as mesh_utils_terrains

if TYPE_CHECKING:
    from isaaclabex.terrains.trimesh import parkour_mesh_cfg


def make_box(
    width: float,
    length: float,
    height: float,
    center: tuple[float, float, float],
    max_yx_angle: float = 0,
    degrees: bool = True,
) -> trimesh.Trimesh:
    """Generate a box mesh with a random orientation.

    Args:
        length: The length (along x) of the box (in m).
        width: The width (along y) of the box (in m).
        height: The height of the cylinder (in m).
        center: The center of the cylinder (in m).
        max_yx_angle: The maximum angle along the y and x axis. Defaults to 0.
        degrees: Whether the angle is in degrees. Defaults to True.

    Returns:
        A trimesh.Trimesh object for the cylinder.
    """
    # create a pose for the cylinder
    transform = np.eye(4)
    transform[0:3, -1] = np.asarray(center)
    # create the box
    dims = (width, length, height)
    return trimesh.creation.box(dims, transform=transform)


def repeated_hurdles_terrain(
    difficulty: float, cfg  : parkour_mesh_cfg.MeshRepeatedHurdleTerrainCfg
) -> dict:
    #tuple[list[trimesh.Trimesh], np.ndarray]:

    # Resolve the terrain configuration
    # -- pass parameters to make calling simpler
    cp_0 = cfg.object_params_start
    cp_1 = cfg.object_params_end
    # -- common parameters
    num_objects = cp_0.num_objects + int(difficulty * (cp_1.num_objects - cp_0.num_objects))
    width = cp_0.width + difficulty * (cp_1.width - cp_0.width)
    length = cp_0.length + difficulty * (cp_1.length - cp_0.length)
    height = cp_0.height + difficulty * (cp_1.height - cp_0.height)
    # -- object specific parameters
    # note: SIM114 requires duplicated logical blocks under a single body.
    _goals = np.zeros((num_objects + 2, 3), dtype= np.float32)

    # initialize list of meshes
    meshes_list = list()
    # compute quantities
    origin = np.asarray((0.5 * cfg.platform_x_space, 0.5 * cfg.size[1], 0.5 * height))
    _goals[0] = origin

    _center_x = cfg.platform_x_space
    _center_y = 0.5 * cfg.size[1]

    _kwargs = {"width": width,
               "length": length}

    _center = np.zeros((3,))

    for i in range(num_objects):
        _dis_x = cp_0.distance_x + random.random() * (cp_1.distance_x - cp_0.distance_x)
        _dis_y = cp_0.distance_y + random.random() * (cp_1.distance_y - cp_0.distance_y)

        _center_x += _dis_x
        _center_y += _dis_y

        ob_height = height + np.random.uniform(-cfg.max_height_noise, cfg.max_height_noise)

        _goals[i + 1] = np.asarray((_center_x, _center_y, ob_height))

        if ob_height > 0.0:
            _center[0] = _center_x
            _center[1] = _center_y
            _kwargs["height"] = ob_height
            _kwargs["center"] = _center
            _mesh = make_box(**_kwargs)

            meshes_list.append(_mesh)

    _dis_x = cp_0.distance_x + random.random() * (cp_1.distance_x - cp_0.distance_x)
    _center_x += _dis_x
    _center_x = (_center_x + cfg.size[0]) / 2
    _goals[- 1] = np.asarray((_center_x, 0.5 * cfg.size[1], 0))

    # generate a ground plane for the terrain
    ground_plane = mesh_utils_terrains.make_plane(cfg.size, height=0.0, center_zero=False)
    meshes_list.append(ground_plane)

    return {"meshes": meshes_list, "origin": origin, "goals": _goals}
