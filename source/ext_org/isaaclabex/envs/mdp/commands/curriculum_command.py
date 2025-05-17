from __future__ import annotations

from typing import TYPE_CHECKING
import torch
from collections.abc import Sequence
from isaaclab.envs.mdp.commands import velocity_command
from torch.nn import functional as F
from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_cfg import CurriculumCommandCfg


class CurriculumCommand(velocity_command.UniformVelocityCommand):

    cfg: CurriculumCommandCfg

    def __init__(self, cfg: CurriculumCommandCfg, env: ManagerBasedEnv):
        super(CurriculumCommand, self).__init__(cfg, env)
        if not self.cfg.is_curriculum:
            return
        self._current_levels = torch.ones((self.num_envs, ), dtype=torch.float32, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        super(CurriculumCommand, self)._resample_command(env_ids)
        if not self.cfg.is_curriculum:
            return
        self.vel_command_b[env_ids] *= (self._current_levels[env_ids].unsqueeze(1) / self.cfg.max_curriculum_levels)

    '''
    called by curriculum
    '''
    def update_curriculum(self, env_ids, move_down, move_up):
        if not self.cfg.is_curriculum:
            return
        if len(env_ids) != 0:

            ids_downs = env_ids[move_down]
            ids_ups = env_ids[move_up]

            self._current_levels[ids_downs] -= 0.5
            self._current_levels[ids_ups] += 0.5
            self._current_levels[env_ids] = torch.clamp(self._current_levels[env_ids], 1.0, self.cfg.max_curriculum_levels)

    @property
    def current_levels(self):
        if not self.cfg.is_curriculum:
            return torch.tensor([0], device = self.device)

        return torch.mean(self._current_levels)
