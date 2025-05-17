from __future__ import annotations
import torch
from collections.abc import Sequence
from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass
from typing import TYPE_CHECKING
from dataclasses import MISSING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

'''
params={"reward_curriculum_levels": "base_velocity", "reward_curriculum_names": 40}

'''

@configclass
class RewardTermCfgWithCurriculum(RewardTermCfg):
    start_weight: float = MISSING
    end_weight: float = MISSING

_start_init_weight = False

def reward_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int],
    reward_curriculum_levels: int = 6,
    reward_curriculum_names: Sequence[str] = MISSING
) -> torch.Tensor:

    '''
    command = env.command_manager.get_term(command_name)
    if 0 != len(env_ids):
        average_episode_length = env.curriculum_manager.average_episode_length

        move_down = average_episode_length < curriculum_level_down_threshold
        move_up = average_episode_length > curriculum_level_up_threshold
        command.update_curriculum(env_ids, move_down, move_up)
    '''
    global _start_init_weight

    _steps = env.cfg.max_iterations // int (reward_curriculum_levels * 1.5)
    _level = env.common_step_counter // _steps

    if env.common_step_counter < _steps and not _start_init_weight:
        _start_init_weight = True

        reward_manager = env.reward_manager
        for reward_name in reward_curriculum_names:
            _cfg: RewardTermCfgWithCurriculum = reward_manager.get_term_cfg(reward_name)
            _cfg.weight = _cfg.start_weight
            print(f"init weight {reward_name}  {_cfg.weight}")

    elif _start_init_weight and 0 == env.common_step_counter % _steps:

        if _level <= reward_curriculum_levels:

            reward_manager = env.reward_manager
            for reward_name in reward_curriculum_names:
                _cfg: RewardTermCfgWithCurriculum = reward_manager.get_term_cfg(reward_name)

                _cfg.weight = _cfg.start_weight + \
                            (_cfg.end_weight - _cfg.start_weight) * _level / reward_curriculum_levels
                print(f"update to {_level}/{reward_curriculum_levels} weight{reward_name}  {_cfg.weight}")

    _level = min(_level, reward_curriculum_levels)
    # return the mean terrain level
    return torch.tensor(_level, device=env.device)
