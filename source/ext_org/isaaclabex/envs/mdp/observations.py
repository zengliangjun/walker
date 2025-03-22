
from __future__ import annotations

import torch
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def velocity_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    return env.command_manager.get_command(command_name)[:, :3]


def gait_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    return env.command_manager.get_command(command_name)[:, 3:]
