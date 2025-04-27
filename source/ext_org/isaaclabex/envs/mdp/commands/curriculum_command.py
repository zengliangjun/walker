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
    def __init__(self, cfg: CurriculumCommandCfg, env: ManagerBasedEnv):
        super(CurriculumCommand, self).__init__(cfg, env)
        if not self.cfg.is_curriculum:
            return

        self.current_levels = torch.ones((self.num_envs, ), dtype=torch.float32, device=self.device)
        self.resample_orgs = torch.zeros(self.num_envs, 3, device=self.device)
        self.resample_times = torch.zeros(self.num_envs, device=self.device)

    def _resample(self, env_ids: Sequence[int]):
        super(CurriculumCommand, self)._resample(env_ids)
        if not self.cfg.is_curriculum:
            return
        if len(env_ids) != 0:
            asset: Articulation = self._env.scene[self.cfg.asset_name]
            self.resample_orgs[env_ids, :3] = asset.data.root_link_pos_w[env_ids, :3]
            self.resample_times[env_ids] = 0.0

    def _resample_command(self, env_ids: Sequence[int]):
        super(CurriculumCommand, self)._resample_command(env_ids)
        if not self.cfg.is_curriculum:
            return

        self.vel_command_b[env_ids] *= (self.current_levels[env_ids].unsqueeze(1) / self.cfg.max_curriculum)

    def compute(self, dt: float):
        if self.cfg.is_curriculum:
            self.resample_times += dt
        super(CurriculumCommand, self).compute(dt)

    def update_curriculum(self, env_ids, _ups, _downs):
        if not self.cfg.is_curriculum:
            return
        if len(env_ids) == 0:
            return
        env_ids = torch.tensor(env_ids, dtype=torch.int64, device=self.device)

        _ids_ups = env_ids[_ups.cpu().numpy()]
        _ids_downs = env_ids[_downs.cpu().numpy()]

        self.current_levels[_ids_ups] += 0.5
        #self.current_levels[_ids_downs] -= 0.5

        torch.clamp_(self.current_levels[env_ids], 1.0, self.cfg.max_curriculum)
