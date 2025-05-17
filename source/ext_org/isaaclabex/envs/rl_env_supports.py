# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: np.ndarray | None
from __future__ import annotations

import gymnasium as gym
import torch
from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.envs.common import VecEnvStepReturn
from isaaclabex.envs.rl_env_supports_cfg import ManagerBasedRLEnv_ExtendsCfg
from isaaclabex.managers import curriculum_manager_support
if False:
    from isaaclab.managers import RewardManager
    from isaaclab.ui.widgets import ManagerLiveVisualizer
    from isaaclabex.managers.recovery import RecoveryManager

class ManagerBasedRLEnv_Extends(ManagerBasedRLEnv):
    def __init__(self, cfg: ManagerBasedRLEnv_ExtendsCfg, render_mode: str | None = None, **kwargs):
        super(ManagerBasedRLEnv_Extends, self).__init__(cfg=cfg)

    def load_managers(self):
        super(ManagerBasedRLEnv_Extends, self).load_managers()

        self.curriculum_manager = curriculum_manager_support.CurriculumManagerEx(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: Extends", self.curriculum_manager)

        # prepare the managers
        # -- recovery manager
        if False:
            self.recovery_manager = RecoveryManager(self.cfg.recoverys, self)
            print("[INFO] Recovery Manager: ", self.recovery_manager)
            self.recovery_reward_manager = RewardManager(self.cfg.recovery_rewards, self)
            print("[INFO] Recovery Reward Manager: ", self.recovery_reward_manager)

    def setup_manager_visualizers(self):
        super(ManagerBasedRLEnv_Extends, self).setup_manager_visualizers()
        # -- recovery manager
        if False:
            self.manager_visualizers["recovery_manager"] = ManagerLiveVisualizer(manager=self.recovery_manager)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        if False:
            self.recovery_manager.compute()
            self.reward_buf = self.recovery_reward_manager.compute(dt=self.step_dt)
        _result = super(ManagerBasedRLEnv_Extends, self).step(action)
        return _result
