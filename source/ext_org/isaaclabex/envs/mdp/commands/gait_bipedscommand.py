from __future__ import annotations

from typing import TYPE_CHECKING
import torch
from collections.abc import Sequence
from torch.nn import functional as F
from .curriculum_command import CurriculumCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_cfg import BipedsStyleCommandCfg

class BipedsStyleCommand(CurriculumCommand):

    def __init__(self, cfg: BipedsStyleCommandCfg, env: ManagerBasedEnv):
        super(BipedsStyleCommand, self).__init__(cfg, env)

        #  style  2
        #  frequencie, duty_cycle, height, 2
        self.phases = torch.zeros(self.num_envs, 2, device=self.device)  # record the phase of the gait
        self.styles = torch.zeros(self.num_envs, 1, dtype=torch.long, device=self.device)
        self.style_command_b = torch.zeros(self.num_envs, 7, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "StyleCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return torch.cat([self.vel_command_b, self.style_command_b], dim=1)

    def _update_metrics(self):
        # time for which the command was executed
        super(BipedsStyleCommand, self)._update_metrics()

    def _resample_command(self, env_ids: Sequence[int]):
        super(BipedsStyleCommand, self)._resample_command(env_ids)

        r = torch.empty(len(env_ids), device=self.device)
        styles = torch.randint_like(r, 0, 100, device=self.device)
        styles = (styles > 33).long()
        _styles_one_hot = F.one_hot(styles, num_classes=2)

        _frequencie = r.uniform_(*self.cfg.ranges.frequencie)
        _duty_cycle = r.uniform_(*self.cfg.ranges.duty_cycle)
        _height = r.uniform_(*self.cfg.ranges.height)

        # phase
        _phase = torch.empty((len(env_ids), 2), device=self.device)
        _styles_synchronous = styles == 0
        _phase[_styles_synchronous, :] = torch.tensor((0.0, 0.0), device=self.device)

        _styles_asynchronous = styles == 1
        _phase[_styles_asynchronous, :] = torch.tensor((0.0, - 0.5), device=self.device)

        # sample
        self.phases[env_ids, :] = _phase
        self.styles[env_ids, :] = styles[:, None]

        self.style_command_b[env_ids, : 2] = _styles_one_hot.float()
        self.style_command_b[env_ids, 2] = _frequencie
        self.style_command_b[env_ids, 3] = _duty_cycle
        self.style_command_b[env_ids, 4] = _height

        # standing
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.style_command_b[standing_env_ids, :] = 0
        self.styles[standing_env_ids, :] = 0
        self.phases[standing_env_ids, :] = 0

    def _update_command(self):
        super(BipedsStyleCommand, self)._update_command()

        _step_phase = self._env.step_dt * self.style_command_b[:, 2: 3] # _frequencie
        self.phases = self.phases + _step_phase

        _phases = torch.clip(self.phases, min=0)
        torch.remainder(_phases, 1, out=_phases)

        self.style_command_b[:, 5:] = _phases
