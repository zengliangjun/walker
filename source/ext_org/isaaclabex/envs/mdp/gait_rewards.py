from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float, foot_names: list[str]
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]

    _foots = asset.find_bodies(foot_names, preserve_order=True)
    _heights = asset.data.body_pos_w[:, _foots[0], 2]

    _command = env.command_manager.get_command("base_velocity")

    _duty_cycle_ids = [4, 8, 12, 16]
    _lift_height_ids = [5, 9, 13, 17]
    _phase_ids = [6, 10, 14, 18]

    _duty_cycles = _command[:, _duty_cycle_ids]
    _lift_heights = _command[:, _lift_height_ids]
    _phases = _command[:, _phase_ids]

    _swing_cycles = 1 - _duty_cycles
    _swing_phases = _phases / _swing_cycles
    _swing_phases = torch.clip(_swing_phases, min=0, max=1)
    _target_height = _lift_heights * torch.sin(_swing_phases * torch.pi)

    _error = torch.square(_heights - _target_height)
    return torch.exp(-torch.sum(_error, dim=1) / std)


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, joint_names: list[str], weights: list[float]) -> torch.Tensor:

    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    #cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    #body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    _joints = asset.find_joints(joint_names, preserve_order=True)
    _weights = torch.tensor(weights, device=env.device)[None, :]

    reward = asset.data.joint_pos - asset.data.default_joint_pos

    _command = env.command_manager.get_command("base_velocity")
    _duty_cycle_ids = [4, 8, 12, 16]
    _phase_ids = [6, 10, 14, 18]

    _duty_cycles = _command[:, _duty_cycle_ids]
    _phases = _command[:, _phase_ids]

    _swing_cycles = 1 - _duty_cycles
    _swing_phases = _phases / _swing_cycles
    _swing_phases = torch.clip(_swing_phases, min=0, max=1)

    _swing_phases = torch.repeat_interleave(_swing_phases[:, :, None], 3, dim=-1)
    _swing_phases = torch.flatten(_swing_phases, start_dim=1)

    reward = reward[:, _joints[0]] * _weights * torch.cos(_swing_phases * torch.pi)
    reward = torch.linalg.norm(reward, dim=1)
    #return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)
    return reward
