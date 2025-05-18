import gymnasium as gym
from . import h1_stylelatent_env_cfg, g123dof_stylelatent_env_cfg, rsl_rl_ppo_cfg

gym.register(
    id="H1StyleLatent-v0",
    entry_point="isaaclabex.envs.rl_env_supports:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": h1_stylelatent_env_cfg.StyleLatentEnvCfg,
        "rsl_rl_cfg_entry_point": f"{rsl_rl_ppo_cfg.__name__}:StyleLatentPPORunnerCfg",
    },
)

gym.register(
    id="H1StyleLatent-Play-v0",
    entry_point="isaaclabex.envs.rl_env_supports:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": h1_stylelatent_env_cfg.StyleLatentEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{rsl_rl_ppo_cfg.__name__}:StyleLatentPPORunnerCfg",
    },
)

gym.register(
    id="G123StyleLatent-v0",
    entry_point="isaaclabex.envs.rl_env_supports:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": g123dof_stylelatent_env_cfg.StyleLatentEnvCfg,
        "rsl_rl_cfg_entry_point": f"{rsl_rl_ppo_cfg.__name__}:StyleG123LatentPPORunnerCfg",
    },
)

gym.register(
    id="G123StyleLatent-Play-v0",
    entry_point="isaaclabex.envs.rl_env_supports:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": g123dof_stylelatent_env_cfg.StyleLatentEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{rsl_rl_ppo_cfg.__name__}:StyleG123LatentPPORunnerCfg",
    },
)
