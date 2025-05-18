
from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from isaaclab.managers import CurriculumManager
from collections.abc import Sequence

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class CurriculumManagerEx(CurriculumManager):

    def __init__(self, cfg: object, env: ManagerBasedRLEnv):
        super(CurriculumManagerEx, self).__init__(cfg, env)
        self.average_episode_length = torch.tensor(0, device=self.device, dtype=torch.long) # num_compute_average_epl last termination episode length

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if len(env_ids) != 0:
            # calcute average_episode_length first
            num = len(env_ids)
            current_average_episode_length = torch.mean(self._env.episode_length_buf[env_ids], dtype=torch.float)

            num_compute_average_epl = self._env.cfg.num_compute_average_epl
            self.average_episode_length = self.average_episode_length * (1 - num / num_compute_average_epl) + current_average_episode_length * (num / num_compute_average_epl)

        return super(CurriculumManagerEx, self).reset(env_ids)
