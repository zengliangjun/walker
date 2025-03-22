from isaaclab.utils import configclass
from ext_org.tasks.locomotion.velocity.config.anymal_d.agents import rsl_rl_ppo_cfg

@configclass
class Go2RoughPPORunnerCfg(rsl_rl_ppo_cfg.AnymalDRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "Go2_rough"


@configclass
class Go2DFlatPPORunnerCfg(rsl_rl_ppo_cfg.AnymalDFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "Go2_flat"
