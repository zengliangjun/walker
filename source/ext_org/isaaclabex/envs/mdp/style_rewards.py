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

    _duty_cycles = _command[:, 5 + 3: 6 + 3]
    _lift_heights = _command[:, 6 + 3: 7 + 3]
    _phases = _command[:, 7 + 3:]

    _swing_cycles = 1 - _duty_cycles
    _swing_phases = _phases / _swing_cycles
    _swing_phases = torch.clip(_swing_phases, min=0, max=1)
    _target_height = _lift_heights * torch.sin(_swing_phases * torch.pi)

    _error = torch.square(_heights - _target_height)
    return torch.exp(-torch.sum(_error, dim=1) / std)


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, joint_names: list[str], weights: list[float]) -> torch.Tensor:

    """Penalize joint position error from default on the articulation."""
    asset: Articulation = env.scene[asset_cfg.name]
    _joints = asset.find_joints(joint_names, preserve_order=True)
    _weights = torch.tensor(weights, device=env.device)[None, :]

    reward = asset.data.joint_pos - asset.data.default_joint_pos
    _command = env.command_manager.get_command("base_velocity")

    _duty_cycles = _command[:, 5 + 3: 6 + 3]
    _phases = _command[:, 7 + 3:]

    _swing_cycles = 1 - _duty_cycles
    _swing_phases = _phases / _swing_cycles
    _swing_phases = torch.clip(_swing_phases, min=0, max=1)

    _swing_phases = torch.repeat_interleave(_swing_phases[:, :, None], 3, dim=-1)
    _swing_phases = torch.flatten(_swing_phases, start_dim=1)

    reward = reward[:, _joints[0]] * _weights * torch.cos(_swing_phases * torch.pi)
    reward = torch.linalg.norm(reward, dim=1)
    return reward


def bipeds_foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float, foot_names: list[str]
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]

    _foots = asset.find_bodies(foot_names, preserve_order=True)
    _heights = asset.data.body_pos_w[:, _foots[0], 2]

    _command = env.command_manager.get_command("base_velocity")

    _duty_cycles = _command[:, 3 + 3: 4 + 3]
    _lift_heights = _command[:, 4 + 3: 5 + 3]
    _phases = _command[:, 5 + 3:]

    _swing_cycles = 1 - _duty_cycles
    _swing_phases = _phases / _swing_cycles
    _swing_phases = torch.clip(_swing_phases, min=0, max=1)
    _target_height = _lift_heights * torch.sin(_swing_phases * torch.pi)

    _error = torch.square(_heights - _target_height)
    return torch.exp(-torch.sum(_error, dim=1) / std)

def bipeds_joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, joint_names: list[str], weights: list[float]) -> torch.Tensor:

    """Penalize joint position error from default on the articulation."""
    asset: Articulation = env.scene[asset_cfg.name]
    _joints = asset.find_joints(joint_names, preserve_order=True)
    _weights = torch.tensor(weights, device=env.device)[None, :]

    reward = asset.data.joint_pos - asset.data.default_joint_pos
    _command = env.command_manager.get_command("base_velocity")

    _duty_cycles = _command[:, 3 + 3: 4 + 3]   # N * 1
    _phases = _command[:, 5 + 3:]   # N * 2

    _swing_cycles = 1 - _duty_cycles
    _swing_phases = _phases / _swing_cycles
    _swing_phases = torch.clip(_swing_phases, min=0, max=1)   # N * 2

    _swing_phases = torch.repeat_interleave(_swing_phases[:, :, None], 9, dim=-1)   # N * 2
    _swing_phases = torch.flatten(_swing_phases, start_dim=1)

    reward = reward[:, _joints[0]] * _weights * torch.cos(_swing_phases * torch.pi)
    reward = torch.linalg.norm(reward, dim=1)
    return reward
