from __future__ import annotations

import os

import torch
import time
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter
from collections import deque
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, EmpiricalNormalization
from rsl_rl.utils import store_code_state

import rsl_rl
from rsl_rl.runners import OnPolicyRunner

from rsl_rlex.models.vae import VAE
from rsl_rlex.algorithms.gait_ppo import StylePPO


class GaitPolicyRunner(OnPolicyRunner):
    """On-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        # super(GaitPolicyRunner, self).__init__(env, train_cfg, log_dir, device)

        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]

        assert "critic" in extras["observations"]
        num_critic_obs = extras["observations"]["critic"].shape[1]

        assert "style" in extras["observations"]
        num_style = extras["observations"]["style"].shape[1]

        style_latent_dim = self.policy_cfg.pop("style_latent_dim")

        style_vae_class = eval(self.policy_cfg.pop("style_vae_class_name"))
        style_encoder_dims = self.policy_cfg.pop("style_encoder_dims")

        style_vae: VAE = style_vae_class(num_style,
                                         latent_dim=style_latent_dim,
                                         mid_dims=style_encoder_dims
                                        ).to(self.device)

        actor_critic_class = eval(self.policy_cfg.pop("class_name"))  # ActorCritic
        actor_critic: ActorCritic | ActorCriticRecurrent = actor_critic_class(
            num_obs + style_latent_dim,
            num_obs + num_critic_obs + num_style,
            self.env.num_actions, **self.policy_cfg
        ).to(self.device)
        alg_class = eval(self.alg_cfg.pop("class_name"))  # SamplePPO
        self.alg: StylePPO = alg_class(actor_critic, style_vae=style_vae, device=self.device, **self.alg_cfg)

        # store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[num_critic_obs], until=1.0e8).to(self.device)
            self.style_normalizer = torch.nn.Identity()  #EmpiricalNormalization(shape=[num_style], until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity()  # no normalization
            self.critic_obs_normalizer = torch.nn.Identity()  # no normalization
            self.style_normalizer = torch.nn.Identity()  # no normalization

        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_critic_obs],
            [self.env.num_actions],
            [num_style],
        )

        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise AssertionError("logger type not found")

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs, extras = self.env.get_observations()
        critic_obs = extras["observations"].get("critic", obs)
        style = extras["observations"].get("style")

        obs = obs.to(self.device)
        critic_obs = critic_obs.to(self.device)
        style = style.to(self.device)

        self.train_mode()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):

                    actions = self.alg.act(obs, critic_obs, style)
                    obs, rewards, dones, infos = self.env.step(actions)

                    ######################################
                    obs = self.obs_normalizer(obs)
                    critic_obs = self.critic_obs_normalizer(infos["observations"]["critic"])
                    style = self.style_normalizer(infos["observations"].get("style"))

                    style = style.to(self.device)

                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(rewards, dones, infos)
                    ######################################

                    if self.log_dir is not None:
                        # Book keeping
                        # note: we changed logging to use "log" instead of "episode" to avoid confusion with
                        # different types of logging data (rewards, curriculum, etc.)
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(obs, critic_obs, style)

            mean_value_loss, mean_surrogate_loss, \
                mean_entropy, mean_rnd_loss, mean_symmetry_loss, \
                ext_loss_items = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Logging info and save checkpoint
            if self.log_dir is not None:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()

            # Save code state
            if it == start_iter:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        ext_loss_items = locs["ext_loss_items"]
        super(GaitPolicyRunner, self).log(locs, width, pad)
        for key, value in ext_loss_items.items():
            print(f"""{f'ext loss {key}':>{pad}} {value:.4f}\n""")
            self.writer.add_scalar(f"Loss/ext_{key}", value, locs["it"])

    def save(self, path, infos=None):
        saved_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        saved_dict["model_style"] = self.alg.style_vae.state_dict()
        saved_dict["optimizer_style"] = self.alg.style_optimizer.state_dict()

        # -- Save RND model if used
        if self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        # -- Save observation normalizer if used
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["critic_obs_norm_state_dict"] = self.critic_obs_normalizer.state_dict()

            saved_dict["style_norm_state_dict"] = self.style_normalizer.state_dict()

        torch.save(saved_dict, path)

        # Upload model to external logging service
        if self.logger_type in ["neptune", "wandb"]:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        # -- Load PPO model
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        self.alg.style_vae.load_state_dict(loaded_dict["model_style"])

        # -- Load RND model if used
        if self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        # -- Load observation normalizer if used
        if self.empirical_normalization:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])

            self.style_normalizer.load_state_dict(loaded_dict["style_norm_state_dict"])

        # -- Load optimizer if used
        if load_optimizer:
            # -- PPO
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            self.alg.style_optimizer.load_state_dict(loaded_dict["optimizer_style"])
            # -- RND optimizer if used
            if self.alg.rnd:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])

        # -- Load current learning iteration
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        from rsl_rlex.models.gait_inference import StyleInference
        _inference = StyleInference()

        if device is not None:
            _inference = _inference.to(device)
        else:
            _inference = _inference.to(self.device)

        obs_normalizer = self.obs_normalizer
        style_normalizer = self.style_normalizer

        actor = self.alg.actor_critic.actor
        style_encoder = self.alg.style_vae.encoder

        if device is not None:
            obs_normalizer = obs_normalizer.to(device)
            style_normalizer = style_normalizer.to(device)
            actor = actor.to(device)
            style_encoder = style_encoder.to(device)

        setattr(_inference, "obs_normalizer", obs_normalizer)
        setattr(_inference, "style_normalizer", style_normalizer)

        setattr(_inference, "actor", actor)
        setattr(_inference, "style_encoder", style_encoder)

        return _inference

    def train_mode(self):
        super(GaitPolicyRunner, self).train_mode()
        self.alg.style_vae.train()

        if self.empirical_normalization:
            self.style_normalizer.train()

    def eval_mode(self):
        super(GaitPolicyRunner, self).eval_mode()
        self.alg.style_vae.eval()

        if self.empirical_normalization:
            self.style_normalizer.eval()
