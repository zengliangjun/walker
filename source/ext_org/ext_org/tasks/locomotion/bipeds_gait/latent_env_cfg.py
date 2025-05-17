from __future__ import annotations

import math

from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm, CurriculumTermCfg
from isaaclabex.envs.mdp.commands import commands_cfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclabex.envs.mdp import privileged_observations
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclabex.envs.mdp import observations
from isaaclab_assets import H1_CFG


from isaaclabex.envs.mdp.curriculums import rewards


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    base_velocity = commands_cfg.BipedsStyleCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=0,
        heading_command=False,
        debug_vis=True,

        is_curriculum = True,
        max_curriculum_levels=4,

        ranges=commands_cfg.BipedsStyleCommandCfg.Ranges(
            lin_vel_x=(-0.5, 2.5),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(- 1.0, 1.0),
            heading=(-math.pi, math.pi),

            stride=(0.25, 0.45),
            duty_cycle=(0.3, 0.7),
            height=(0.05, 0.35),
        ),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # `` observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=observations.velocity_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.5, n_max=0.5)
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(PolicyCfg):
        foot_pos = ObsTerm(
            func=privileged_observations.feet_pos,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_link")
            },
            noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        contact_forces = ObsTerm(
            func=privileged_observations.feet_contact_forces,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link")
            },
            noise=Unoise(n_min=-0.1, n_max=0.1)
        )

        contact_status = ObsTerm(
            func=privileged_observations.feet_contact_status,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
                "threshold": 1.0,
            },
        )

        rigid_body_mass = ObsTerm(
            func=privileged_observations.rigid_body_mass,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

        joint_acc = ObsTerm(
            func=privileged_observations.joint_acc,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
            noise=Unoise(n_min=-0.1, n_max=0.1))

        joint_stiffness = ObsTerm(
            func=privileged_observations.joint_stiffness,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

        joint_damping = ObsTerm(
            func=privileged_observations.joint_damping,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

        joint_torques = ObsTerm(
            func=privileged_observations.joint_torques,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

    @configclass
    class StyleCfg(ObsGroup):
        gait = ObsTerm(func=observations.gait_commands, params={"command_name": "base_velocity"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()
    style: StyleCfg = StyleCfg()

@configclass
class EventCfg:
    """Configuration for randomization."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*torso_link"),
            "mass_distribution_params": (-2.5, 2.5),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-1.5, 1.5),
                "y": (-1.0, 1.0),
                "z": (-0.5, 0.5),
                "roll": (-0.7, 0.7),
                "pitch": (-0.7, 0.7),
                "yaw": (-1.0, 1.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=spot_mdp.reset_joints_around_default,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-2.5, 2.5),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )

from isaaclabex.envs.mdp.rewards import bipeds_style, general

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    reward_track_lin_vel_xy_exp = rewards.RewardTermCfgWithCurriculum(
        func=mdp.track_lin_vel_xy_exp,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
        weight=1.0,
        start_weight=4.0,
        end_weight=1.0,
    )
    reward_track_ang_vel_z_exp = rewards.RewardTermCfgWithCurriculum(
        func=mdp.track_ang_vel_z_exp, params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
        weight=0.5,
        start_weight=4.0,
        end_weight=1.0,
    )

    # -- penalties
    penalize_foot_clearance = rewards.RewardTermCfgWithCurriculum(
        func=bipeds_style.bipeds_foot_clearance_reward,
        weight=-.5,
        params={
            "std": 0.05,
            "asset_cfg": SceneEntityCfg("robot"),
            "foot_names": ["left_ankle_link", "right_ankle_link"]
        },
        start_weight=-.1,
        end_weight=-.5,
    )

    penalize_space_feet = rewards.RewardTermCfgWithCurriculum(
        func=general.penalty_space_body,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "base_width": 0.4,
            "body_names": ['left_knee_link', 'right_knee_link',
                           'left_ankle_link', 'right_ankle_link'],
        },
        start_weight=-.1,
        end_weight=-0.5
    )

    penalize_foot_slip = rewards.RewardTermCfgWithCurriculum(
        func=spot_mdp.foot_slip_penalty,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_link"),
            "threshold": 1.0,
        },
        start_weight=-.3,
        end_weight=-1.0
    )


    penalize_action_smoothness = rewards.RewardTermCfgWithCurriculum(
        func=spot_mdp.action_smoothness_penalty,
        weight=-1.0,
        start_weight=-.1,
        end_weight=-1.0)

    penalize_base_motion = rewards.RewardTermCfgWithCurriculum(
        func=spot_mdp.base_motion_penalty, weight=-.5, params={"asset_cfg": SceneEntityCfg("robot")},
        start_weight=-.1,
        end_weight=-.5
    )

    penalize_base_orientation = rewards.RewardTermCfgWithCurriculum(
        func=spot_mdp.base_orientation_penalty, weight=-3.0, params={"asset_cfg": SceneEntityCfg("robot")},
        start_weight=-.5,
        end_weight=-3.0
    )

    ##################################################
    # join style
    penalize_leg_joint_pos = rewards.RewardTermCfgWithCurriculum(
        func=bipeds_style.bipeds_leg_joint_position_penalty,
        weight=-2.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_names": ['left_hip_yaw', 'left_hip_roll', 'left_hip_pitch',
                            'left_knee',
                            #'left_shoulder_yaw',  'left_shoulder_roll', 'left_shoulder_pitch', 'left_elbow'

                            'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch',
                            'right_knee',
                            # 'right_shoulder_yaw', 'right_shoulder_roll', 'right_shoulder_pitch', 'right_elbow'
                            ],
            "joint_weights": [1, 1, 0.5, 1,
                              #  5, 5, 4, 5,
                              1, 1, 0.5, 1,
                              #  5, 5, 4, 5,
                            ],

            "feet_names": ['left_ankle', 'right_ankle'],
            "feet_weights": [1, 1],
        },
        start_weight=-.1,
        end_weight=-2.5
    )

    penalize_shoulder_joint_freeze_pos = rewards.RewardTermCfgWithCurriculum(
        func=general.penalty_joint_freeze,
        weight=-2,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_names": ['left_shoulder_yaw',  'left_shoulder_roll', 'left_shoulder_pitch', 'left_elbow',
                            'right_shoulder_yaw', 'right_shoulder_roll', 'right_shoulder_pitch', 'right_elbow'
                            ],

            "joint_weights": [1.5, 1.5, 1, 1.5,
                        1.5, 1.5, 1, 1.5]
        },
        start_weight=-.3,
        end_weight=-2
    )

    # join acc
    penalize_joint_acc = rewards.RewardTermCfgWithCurriculum(
        func=spot_mdp.joint_acceleration_penalty,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
        start_weight=-4.0e-5,
        end_weight=-1.0e-4
    )

    # join torso
    penalize_joint_deviation_torso = rewards.RewardTermCfgWithCurriculum(
        func=mdp.joint_deviation_l1, weight=-5, params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso")},
        start_weight=-.5,
        end_weight=-5
    )

    # join torques
    penalize_joint_torques = rewards.RewardTermCfgWithCurriculum(
        func=spot_mdp.joint_torques_penalty,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
        start_weight=-1.0e-4,
        end_weight=-5.0e-4
    )

    # join vel
    penalize_joint_vel = rewards.RewardTermCfgWithCurriculum(
        func=spot_mdp.joint_velocity_penalty,
        weight=-1.0e-4,
        params={"asset_cfg":
                SceneEntityCfg("robot",
                               joint_names=[".*hip_yaw",
                                            ".*hip_roll",
                                            "torso",
                                            ".*shoulder_roll",
                                            ".*shoulder_yaw",
                                            ".*_elbow"])},
        start_weight=-1.0e-5,
        end_weight=-1.0e-4
    )

    #termination_penalty = RewardTermCfg(func=mdp.is_terminated, weight=-4.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*torso_link"]), "threshold": 1.0},
    )
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )


from isaaclabex.envs.mdp.curriculums import command
@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    commands_levels = CurriculumTermCfg(
        func=command.command_curriculum_levels_vel,
        params={"command_name": "base_velocity"} )

    reward_levels = CurriculumTermCfg(
        func=rewards.reward_levels_vel,
        params={"reward_curriculum_levels": 10,
                 "reward_curriculum_names": [

                'reward_track_lin_vel_xy_exp',
                'reward_track_ang_vel_z_exp',
                'penalize_foot_clearance',
                'penalize_space_feet',
                'penalize_foot_slip',
                'penalize_action_smoothness',
                'penalize_base_motion',
                #'penalize_base_orientation',
                'penalize_leg_joint_pos',
                'penalize_shoulder_joint_freeze_pos',
                'penalize_joint_deviation_torso',
                'penalize_joint_acc',
                'penalize_joint_torques',
                'penalize_joint_vel']} )


from isaaclab_tasks.manager_based.locomotion.velocity import velocity_env_cfg
from isaaclabex.envs.rl_env_supports_cfg import ManagerBasedRLEnv_ExtendsCfg
@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnv_ExtendsCfg):
    """Configuration for the locomotion velocity-tracking environment."""
    # Scene settings
    scene = velocity_env_cfg.MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class StyleLatentEnvCfg(LocomotionVelocityRoughEnvCfg):

    def __post_init__(self):
        """Post initialization."""
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        super(StyleLatentEnvCfg, self).__post_init__()
        self.scene.robot = H1_CFG.copy().replace(prim_path="{ENV_REGEX_NS}/Robot")

@configclass
class StyleLatentEnvCfg_PLAY(StyleLatentEnvCfg):
    env_debug_flags: bool = True
    curriculum = None

    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        self.commands.base_velocity.is_curriculum = False
        # self.commands.base_velocity.resampling_time_range = (3.0, 5.0)

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
