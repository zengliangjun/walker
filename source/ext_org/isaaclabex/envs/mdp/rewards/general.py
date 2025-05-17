from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def penalty_space_body(
        env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, base_width: int, body_names: list[str]) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]

    body_ids, _ = asset.find_bodies(body_names, preserve_order=True)
    body_pos_w = asset.data.body_pos_w[:, body_ids]

    #_qat_w = asset.data.root_link_quat_w[:, None, :].repeat(1, base_pos_w.shape[1], 1)
    #base_pos_b = math_utils.quat_rotate_inverse(_qat_w, base_pos_w)
    _qat_w = asset.data.root_link_quat_w[:, None, :].repeat(1, body_pos_w.shape[1], 1)
    body_pos_b = math_utils.quat_rotate_inverse(_qat_w, body_pos_w)

    body_width = body_pos_b[:, : : 2, 1] - body_pos_b[:, 1: : 2, 1]
    #base_width[:, :] = 0.4

    _penalize = torch.abs(body_width / base_width) - 1
    _penalize = torch.linalg.norm(_penalize, dim=1)
    return _penalize

def penalty_joint_freeze(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg,
     joint_names: list[str], joint_weights: list[float]) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]

    _joints_ids, _ = asset.find_joints(joint_names, preserve_order=True)
    _joint_diff_pos = asset.data.joint_pos[:, _joints_ids] - asset.data.default_joint_pos[:, _joints_ids]
    _joint_weights = torch.tensor(joint_weights, device=env.device)[None, :]

    _penalize = _joint_diff_pos * _joint_weights
    _penalize = torch.linalg.norm(_penalize, dim=1)
    return _penalize
