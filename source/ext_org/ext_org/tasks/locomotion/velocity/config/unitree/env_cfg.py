import ext_org.tasks.locomotion.velocity.mdp as mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

'''
['base', 
'FL_hip', 
'FL_thigh', 
'FL_calf', 
'FL_foot', 

'FR_hip', 
'FR_thigh', 
'FR_calf', 
'FR_foot', 

'Head_upper', 
'Head_lower', 

'RL_hip', 
'RL_thigh', 
'RL_calf', 
'RL_foot', 
'RR_hip', 
'RR_thigh', 
'RR_calf', 
'RR_foot']

'''


def env_cfg_post_init(self):

    # reduce action scale
    self.actions.joint_pos.scale = 0.25

    # event
    self.events.push_robot = None
    self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
    self.events.add_base_mass.params["asset_cfg"].body_names = "base"
    if hasattr(self.events, "base_external_force_torque") and self.events.base_external_force_torque != None:
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
    self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    self.events.reset_base.params = {
        "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
        "velocity_range": {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },
    }

    # rewards
    self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
    self.rewards.feet_air_time.weight = 0.01
    self.rewards.undesired_contacts = None
    self.rewards.dof_torques_l2.weight = -0.0002
    self.rewards.track_lin_vel_xy_exp.weight = 1.5
    self.rewards.track_ang_vel_z_exp.weight = 0.75
    self.rewards.dof_acc_l2.weight = -2.5e-7
