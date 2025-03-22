from isaaclab.utils import configclass
from ext_org.tasks.locomotion.velocity.config.anymal_d import flat_env_cfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG # isort: skip

from . import env_cfg

@configclass
class Go2FlatEnvCfg(flat_env_cfg.AnymalDFlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        env_cfg.env_cfg_post_init(self)

@configclass
class Go2FlatEnvCfg_PLAY(flat_env_cfg.AnymalDFlatEnvCfg_PLAY):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        env_cfg.env_cfg_post_init(self)
