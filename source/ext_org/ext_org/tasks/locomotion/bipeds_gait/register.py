import gymnasium as gym
from . import rsl_rl_ppo_cfg, latent_env_cfg

gym.register(
    id="H1StyleLatent-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": latent_env_cfg.StyleLatentEnvCfg,
        "rsl_rl_cfg_entry_point": f"{rsl_rl_ppo_cfg.__name__}:StyleLatentPPORunnerCfg",
    },
)

gym.register(
    id="H1StyleLatent-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": latent_env_cfg.StyleLatentEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{rsl_rl_ppo_cfg.__name__}:StyleLatentPPORunnerCfg",
    },
)
