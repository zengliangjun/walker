
from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def comand_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)

    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("base_velocity")
    command_term = env.command_manager.get_term("base_velocity")

    XY = torch.norm(command[:, :2], dim=1) < 0.01
    Z = torch.abs(command[:, 2]) < 0.01

    nostanding = torch.logical_not(torch.logical_and(XY, Z)).nonzero(as_tuple=False).flatten().cpu().numpy().tolist()
    env_ids = env_ids.cpu().numpy().tolist()
    env_ids = set(env_ids) & set(nostanding)

    env_ids = list(env_ids) #env_ids[_nostanding]
    if len(env_ids) == 0:
        return torch.tensor(1.0, device=asset.device)

    _distances = torch.norm(asset.data.root_link_pos_w[env_ids, :3] - command_term.resample_orgs[env_ids, :3], dim=1)
    _velocity = _distances / command_term.resample_times[env_ids]
    _velocityXY = torch.norm(command[env_ids, :2], dim=1)

    _ups = _velocity > _velocityXY * 0.7
    _downs = _velocity < _velocityXY * 0.3

    command_term.update_curriculum(env_ids, _ups, _downs)

    return command_term.current_levels.mean()
    '''
    _linear = env.reward_manager._episode_sums["base_linear_velocity"][env_ids] / env.max_episode_length_s > 0.8 * 1.5
    _angular = env.reward_manager._episode_sums["base_angular_velocity"][env_ids] / env.max_episode_length_s > 0.8 * 1
    '''
