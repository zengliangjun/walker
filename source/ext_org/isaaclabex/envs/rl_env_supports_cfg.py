
from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg

@configclass
class ManagerBasedRLEnv_ExtendsCfg(ManagerBasedRLEnvCfg):

    recoverys: object | None = None
    """Command settings. Defaults to None, in which case no commands are generated.

    Please refer to the :class:`isaaclab.managers.CommandManager` class for more details.
    """

    '''
    used for calcute average_episode_length
    '''
    num_compute_average_epl: int = 10000
