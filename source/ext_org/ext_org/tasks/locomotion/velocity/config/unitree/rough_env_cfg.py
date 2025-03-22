from isaaclab.utils import configclass

from ext_org.tasks.locomotion.velocity.config.anymal_d import rough_env_cfg
from . import env_cfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG # isort: skip


@configclass
class Go2RoughEnvCfg(rough_env_cfg.AnymalDRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to anymal-d
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        env_cfg.env_cfg_post_init(self)

@configclass
class Go2RoughEnvCfg_PLAY(rough_env_cfg.AnymalDRoughEnvCfg_PLAY):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        env_cfg.env_cfg_post_init(self)