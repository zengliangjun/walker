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
_current_level = 0
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
    global _start_init_weight, _current_level

    _start_steps = env.cfg.max_iterations // 4
    _step = _start_steps * 2 // reward_curriculum_levels

    if env.common_step_counter < _start_steps and not _start_init_weight:
        _start_init_weight = True

        reward_manager = env.reward_manager
        for reward_name in reward_curriculum_names:
            _cfg: RewardTermCfgWithCurriculum = reward_manager.get_term_cfg(reward_name)
            _cfg.weight = _cfg.start_weight
            print(f"init weight {reward_name}  {_cfg.weight}")

    elif env.common_step_counter > _start_steps:

        _level = (env.common_step_counter - _start_steps) // _step
        _level = min(_level, reward_curriculum_levels)

        if _current_level != _level:

            reward_manager = env.reward_manager
            for reward_name in reward_curriculum_names:
                _cfg: RewardTermCfgWithCurriculum = reward_manager.get_term_cfg(reward_name)

                _cfg.weight = _cfg.start_weight + \
                            (_cfg.end_weight - _cfg.start_weight) * _level / reward_curriculum_levels
                print(f"update to {_level}/{reward_curriculum_levels} weight{reward_name}  {_cfg.weight}")

            _current_level = _level

    # return the mean terrain level
    return torch.tensor(_current_level, device=env.device)
