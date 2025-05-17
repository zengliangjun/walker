from __future__ import annotations
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

'''
params={"command_name": "base_velocity", "curriculum_level_down_threshold": 40}

'''

def command_curriculum_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int],
      command_name
) -> torch.Tensor:

    '''
    command = env.command_manager.get_term(command_name)
    if 0 != len(env_ids):
        average_episode_length = env.curriculum_manager.average_episode_length

        move_down = average_episode_length < curriculum_level_down_threshold
        move_up = average_episode_length > curriculum_level_up_threshold
        command.update_curriculum(env_ids, move_down, move_up)
    '''
    command = env.command_manager.get_term(command_name)
    _steps = env.cfg.max_iterations // int (command.cfg.max_curriculum_levels * 4)
    _levels = env.common_step_counter // _steps + 1
    command.curriculum_up_levels(_levels)

    # return the mean terrain level
    return command.current_levels
