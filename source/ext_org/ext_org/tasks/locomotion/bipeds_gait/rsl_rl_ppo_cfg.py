from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from rsl_rlex.runners.gait_cfg import StylePpoActorCriticCfg


@configclass
class StyleLatentPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 500
    experiment_name = "h1_stylelatent_flat"
    empirical_normalization = True
    store_code_state = False
    policy = StylePpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",

        style_vae_class_name="VAE",
        style_encoder_dims=[64, 256, 128],
        style_latent_dim=32,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="StylePPO",
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0025,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
