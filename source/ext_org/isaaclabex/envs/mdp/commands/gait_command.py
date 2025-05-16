from __future__ import annotations

from typing import TYPE_CHECKING
import torch
from collections.abc import Sequence
from isaaclabex.envs.mdp.commands import curriculum_command
from torch.nn import functional as F

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_cfg import StyleCommandCfg

class StyleCommand(curriculum_command.CurriculumCommand):

    def __init__(self, cfg: StyleCommandCfg, env: ManagerBasedEnv):
        super(StyleCommand, self).__init__(cfg, env)

        #  style  4
        #  frequencie, duty_cycle, height, 4
        self.phases = torch.zeros(self.num_envs, 4, device=self.device)  # record the phase of the gait
        self.styles = torch.zeros(self.num_envs, 1, dtype=torch.long, device=self.device)
        self.style_command_b = torch.zeros(self.num_envs, 11, device=self.device)

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
        super(StyleCommand, self)._update_metrics()

    def _resample_command(self, env_ids: Sequence[int]):
        super(StyleCommand, self)._resample_command(env_ids)

        r = torch.empty(len(env_ids), device=self.device)
        styles = torch.randint_like(r, 0, 4, dtype=torch.long, device=self.device)
        _styles_one_hot = F.one_hot(styles, num_classes=4)

        _frequencie = r.uniform_(*self.cfg.ranges.frequencie)
        _duty_cycle = r.uniform_(*self.cfg.ranges.duty_cycle)
        _height = r.uniform_(*self.cfg.ranges.height)

        # phase
        _phase = torch.empty((len(env_ids), 4), device=self.device)
        _styles_four_beats = styles == 0
        _phase[_styles_four_beats, :] = torch.tensor((0.0, -0.25, -0.5, -0.75), device=self.device)

        _styles_pace = styles == 1
        _phase[_styles_pace, :] = torch.tensor((0.0, - 0.5, 0.0, - 0.5), device=self.device)
        _styles_trot = styles == 2
        _phase[_styles_trot, :] = torch.tensor((0.0, - 0.5, - 0.5, 0.0), device=self.device)
        _styles_leap = styles == 3
        _phase[_styles_leap, :] = torch.tensor((0.0, 0, 0, 0.0), device=self.device)

        # sample
        self.phases[env_ids, :] = _phase
        self.styles[env_ids, :] = styles[:, None]

        self.style_command_b[env_ids, : 4] = _styles_one_hot.float()
        self.style_command_b[env_ids, 4] = _frequencie
        self.style_command_b[env_ids, 5] = _duty_cycle
        self.style_command_b[env_ids, 6] = _height

        # standing
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.style_command_b[standing_env_ids, :] = 0
        self.styles[standing_env_ids, :] = 0
        self.phases[standing_env_ids, :] = 0

    def _update_command(self):
        super(StyleCommand, self)._update_command()

        _step_phase = self._env.step_dt * self.style_command_b[:, : 1]
        self.phases = self.phases + _step_phase

        _phases = torch.clip(self.phases, min=0)
        torch.remainder(_phases, 1, out=_phases)

        self.style_command_b[:, 7:] = _phases
