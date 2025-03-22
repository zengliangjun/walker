from __future__ import annotations

import math

from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from isaaclabex.envs.mdp import style_rewards
from isaaclabex.envs.mdp.commands import commands_cfg
import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
from . import spot_env_cfg

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    base_velocity = commands_cfg.StyleCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=0,
        heading_command=False,
        debug_vis=True,

        ranges=commands_cfg.StyleCommandCfg.Ranges(
            lin_vel_x=(-2.0, 6.0),
            lin_vel_y=(-1.5, 1.5),
            ang_vel_z=(- 2.0, 2.0),
            heading=(-math.pi, math.pi),

            frequencie=(0.8, 4),
            duty_cycle=(0.3, 0.7),
            height=(0.05, 0.25),
        ),
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    base_angular_velocity = RewardTermCfg(
        func=spot_mdp.base_angular_velocity_reward,
        weight=5.0,
        params={"std": 2.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    base_linear_velocity = RewardTermCfg(
        func=spot_mdp.base_linear_velocity_reward,
        weight=5.0,
        params={"std": 1.0, "ramp_rate": 0.5, "ramp_at_vel": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )

    foot_clearance = RewardTermCfg(
        func=style_rewards.foot_clearance_reward,
        weight=2.0,
        params={
            "std": 0.05,
            "asset_cfg": SceneEntityCfg("robot"),
            "foot_names": ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        },
    )

    # -- penalties
    action_smoothness = RewardTermCfg(func=spot_mdp.action_smoothness_penalty, weight=-1.0)
    base_motion = RewardTermCfg(
        func=spot_mdp.base_motion_penalty, weight=-.5, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    base_orientation = RewardTermCfg(
        func=spot_mdp.base_orientation_penalty, weight=-3.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    foot_slip = RewardTermCfg(
        func=spot_mdp.foot_slip_penalty,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 1.0,
        },
    )
    joint_acc = RewardTermCfg(
        func=spot_mdp.joint_acceleration_penalty,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )
    joint_pos = RewardTermCfg(
        func=style_rewards.joint_position_penalty,
        weight=-1.4,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_names": ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
                            'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
                            'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
                            'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'],
            "weights": [2.5, 0.6, 0.6,
                        2.5, 0.6, 0.6,
                        2.5, 0.6, 0.6,
                        2.5, 0.6, 0.6]
        },
    )
    joint_torques = RewardTermCfg(
        func=spot_mdp.joint_torques_penalty,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_vel = RewardTermCfg(
        func=spot_mdp.joint_velocity_penalty,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_joint", ".*thigh_joint"])},
    )

@configclass
class StyleEnvCfg(spot_env_cfg.SpotFlatEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum = None


class StyleEnvCfg_PLAY(StyleEnvCfg):

    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        self.commands.base_velocity.resampling_time_range = (3.0, 5.0)

        # make a smaller scene for play
        self.episode_length_s = 500.0
        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None

        # reduce the number of terrains to save memory
        #if self.scene.terrain.terrain_generator is not None:
        #self.scene.terrain.terrain_generator.num_rows = 2
        #self.scene.terrain.terrain_generator.num_cols = 2
        #self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
