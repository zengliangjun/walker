from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def bipeds_foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float, foot_names: list[str]
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]

    _foots = asset.find_bodies(foot_names, preserve_order=True)
    _heights = asset.data.body_pos_w[:, _foots[0], 2]

    ## gait_bipedscommand
    _command = env.command_manager.get_command("base_velocity")

    _duty_cycles = _command[:, 3 + 3: 4 + 3]  # vvel_command 3 + one_hot 2 + frequencie 1
    _lift_heights = _command[:, 4 + 3: 5 + 3]
    _phases = _command[:, 5 + 3:]             # 2

    _swing_cycles = 1 - _duty_cycles
    _swing_phases = _phases / _swing_cycles
    _swing_phases = torch.clip(_swing_phases, min=0, max=1)
    _target_height = _lift_heights * torch.sin(_swing_phases * torch.pi)

    _error = _heights - _target_height
    return torch.norm(_error, dim=1)

def bipeds_leg_joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg,
     joint_names: list[str], joint_weights: list[float],
     feet_names: list[str], feet_weights: list[float]) -> torch.Tensor:

    """Penalize joint position error from default on the articulation."""
    asset: Articulation = env.scene[asset_cfg.name]

    _command = env.command_manager.get_command("base_velocity")
    #
    _duty_cycles = _command[:, 3 + 3: 4 + 3]   # N * 1
    _phases = _command[:, 5 + 3:]   # N * 2

    _swing_cycles = 1 - _duty_cycles
    _swing_phases = _phases / _swing_cycles
    _swing_phases = torch.clip(_swing_phases, min=0, max=1)   # N * 2

    # joints penalize
    _joint_weights = torch.tensor(joint_weights, device=env.device)[None, :]

    _joints_ids, _ = asset.find_joints(joint_names, preserve_order=True)
    _joint_diff_pos = asset.data.joint_pos[:, _joints_ids] - asset.data.default_joint_pos[:, _joints_ids]

    _size = len(_joints_ids) // 2
    _joint_swing_phases = torch.repeat_interleave(_swing_phases[:, :, None], _size, dim=-1)   # N * 2
    _joint_swing_phases = torch.flatten(_joint_swing_phases, start_dim=1)

    _joint_penalize = _joint_diff_pos * _joint_weights * torch.cos(_joint_swing_phases * torch.pi)

    #
    _duty_phases = (_phases - _swing_cycles) / _duty_cycles
    _duty_phases = torch.clip(_duty_phases, min=0, max=1)   # N * 2

    # feet penalize
    _feet_weights = torch.tensor(feet_weights, device=env.device)[None, :]
    _feet_ids, _ = asset.find_joints(feet_names, preserve_order=True)
    _feet_diff_pos = asset.data.joint_pos[:, _feet_ids] - asset.data.default_joint_pos[:, _feet_ids]

    _size = len(_feet_ids) // 2
    _duty_phases = torch.repeat_interleave(_duty_phases[:, :, None], _size, dim=-1)   # N * 2
    _duty_phases = torch.flatten(_duty_phases, start_dim=1)

    _feet_swing_phases = torch.repeat_interleave(_swing_phases[:, :, None], _size, dim=-1)   # N * 2
    _feet_swing_phases = torch.flatten(_feet_swing_phases, start_dim=1)
    _feet_swing_phases = (_feet_swing_phases != 0).float()

    _feet_penalize = _feet_diff_pos * _feet_weights * (torch.sin(_duty_phases * torch.pi) + _feet_swing_phases)

    if env.cfg.env_debug_flags:
        print("duty", _duty_cycles, "_phases", _phases, "swing", _swing_phases, "duty", _duty_phases)
        #print("leg_joint_penalty", \
        #      'swing_phases', torch.cos(_joint_swing_phases * torch.pi), \
        #      'feet', _feet_swing_phases, \
        #      'duty_phases',  torch.sin(_duty_phases * torch.pi))

    _penalize = torch.cat((_joint_penalize, _feet_penalize), dim=-1)
    _penalize = torch.linalg.norm(_penalize, dim=1)
    return _penalize

