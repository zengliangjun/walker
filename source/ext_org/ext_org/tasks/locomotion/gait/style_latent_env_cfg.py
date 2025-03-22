
from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclabex.envs.mdp import observations
from . import spot_env_cfg
from . import style_env_cfg

@configclass
class StyleObservationsCfg(spot_env_cfg.SpotObservationsCfg):

    @configclass
    class PolicyCfg(spot_env_cfg.SpotObservationsCfg.PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            self.velocity_commands = ObsTerm(func=observations.velocity_commands, params={"command_name": "base_velocity"})

    @configclass
    class PrivilegedCfg(spot_env_cfg.SpotObservationsCfg.PrivilegedCfg):
        def __post_init__(self):
            super().__post_init__()
            self.velocity_commands = ObsTerm(func=observations.velocity_commands, params={"command_name": "base_velocity"})

    @configclass
    class StyleCfg(ObsGroup):
        gait = ObsTerm(func=observations.gait_commands, params={"command_name": "base_velocity"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()
    style: StyleCfg = StyleCfg()

@configclass
class StyleLatentEnvCfg(style_env_cfg.StyleEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()
        self.observations = StyleObservationsCfg()


@configclass
class StyleLatentEnvCfg_PLAY(style_env_cfg.StyleEnvCfg_PLAY):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()
        self.observations = StyleObservationsCfg()
