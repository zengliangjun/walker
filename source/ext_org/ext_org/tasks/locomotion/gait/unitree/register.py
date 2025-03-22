import gymnasium as gym
from ext_org.tasks.locomotion.gait.unitree import rsl_rl_ppo_cfg
from ext_org.tasks.locomotion.gait import spot_env_cfg, gait_env_cfg, style_env_cfg

gym.register(
    id="Quadruped-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": spot_env_cfg.SpotFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{rsl_rl_ppo_cfg.__name__}:SpotFlatPPORunnerCfg",
    },
)

gym.register(
    id="Quadruped-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": spot_env_cfg.SpotFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{rsl_rl_ppo_cfg.__name__}:SpotFlatPPORunnerCfg",
    },
)

gym.register(
    id="Quadruped-Go2Gait-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": gait_env_cfg.GaitEnvCfg,
        "rsl_rl_cfg_entry_point": f"{rsl_rl_ppo_cfg.__name__}:GaitPPORunnerCfg",
    },
)

gym.register(
    id="Quadruped-Go2Gait-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": gait_env_cfg.GaitEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{rsl_rl_ppo_cfg.__name__}:GaitPPORunnerCfg",
    },
)

gym.register(
    id="Quadruped-Go2Style-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": style_env_cfg.StyleEnvCfg,
        "rsl_rl_cfg_entry_point": f"{rsl_rl_ppo_cfg.__name__}:StylePPORunnerCfg",
    },
)

gym.register(
    id="Quadruped-Go2Style-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": style_env_cfg.StyleEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{rsl_rl_ppo_cfg.__name__}:StylePPORunnerCfg",
    },
)

########################################################
from ext_org.tasks.locomotion.gait import style_latent_env_cfg

gym.register(
    id="Quadruped-Go2StyleLatent-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": style_latent_env_cfg.StyleLatentEnvCfg,
        "rsl_rl_cfg_entry_point": f"{rsl_rl_ppo_cfg.__name__}:StyleLatentPPORunnerCfg",
    },
)

gym.register(
    id="Quadruped-Go2StyleLatent-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": style_latent_env_cfg.StyleLatentEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{rsl_rl_ppo_cfg.__name__}:StyleLatentPPORunnerCfg",
    },
)
