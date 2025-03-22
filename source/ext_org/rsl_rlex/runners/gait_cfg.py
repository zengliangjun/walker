from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

@configclass
class StylePpoActorCriticCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks."""

    style_vae_class_name: str = MISSING # "VAE"
    style_encoder_dims: list[int] = MISSING
    style_latent_dim: int = MISSING
