from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from torch.nn import functional as F
from collections.abc import Sequence
from .curriculum_command import CurriculumCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_cfg import PosGaitCommandCfg


class PosGaitCommand(CurriculumCommand):

    def __init__(self, cfg: PosGaitCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        #  pos_height, pos_pitch, pos_roll, gait_frequencie, gait_lift_height,  
        #  gait_style  4, phase
        self.posgait_command_b = torch.zeros(self.num_envs, 10, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "PosGaitCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg
    
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return torch.cat((self.vel_command_b, self.posgait_command_b), dim=1)
    
    def _update_metrics(self):
        super()._update_metrics()
        # TODO

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)

        r = torch.empty(len(env_ids), device=self.device)

        self.posgait_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_height)
        self.posgait_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_pitch)
        self.posgait_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_roll)
        self.posgait_command_b[env_ids, 3] = r.uniform_(*self.cfg.ranges.gait_frequencie)
        self.posgait_command_b[env_ids, 4] = r.uniform_(*self.cfg.ranges.gait_lift_height)

        gait_style = torch.randint_like(r, 0, 4, device=self.device)
        self.posgait_command_b[env_ids, 5: 9] = F.one_hot(gait_style, num_classes = 4)

        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.posgait_command_b[standing_env_ids, :] = 0

    def _update_command(self):
        super()._update_command()
        _step_phase = self._env.step_dt * self.posgait_command_b[:, 3]
        self.posgait_command_b[:, 9] += _step_phase
