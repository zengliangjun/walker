from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab.envs.mdp.commands import commands_cfg
from isaaclabex.envs.mdp.commands import curriculum_command
from isaaclabex.envs.mdp.commands import gait_command, gait_bipedscommand
from isaaclabex.envs.mdp.commands import pos_command

@configclass
class CurriculumCommandCfg(commands_cfg.UniformVelocityCommandCfg):
    class_type: type = curriculum_command.CurriculumCommand

    is_curriculum: bool = False
    max_curriculum: int = 1


@configclass
class GaitCommandCfg(CurriculumCommandCfg):
    class_type: type = gait_command.GaitCommand

    frequencie: float = 1.6
    duty_cycle: float = 0.8
    lift_height: float = 0.25
    phase_offset: tuple[float, float, float, float] = (0.0, 0.25, 0.5, 0.75)

    @configclass
    class Ranges(commands_cfg.UniformVelocityCommandCfg.Ranges):
        frequencie: tuple[float, float] = MISSING
        duty_cycle: tuple[float, float] = MISSING
        lift_height: tuple[float, float] = MISSING
        #phase_offset: tuple[float, float] = (0, 1)

    ranges: Ranges = MISSING

@configclass
class StyleCommandCfg(CurriculumCommandCfg):
    class_type: type = gait_command.StyleCommand

    # gait style
    '''
    四节拍   four beats(crawl 爬， bound 跳跃)
    左右对称，前后同步 (pace 踱步，
    左右反对称，斜对脚同步(trot 小跑，
    全同步(leap 跳跃，
    '''
    style: int = 0  #
    # gait frequencie
    frequencie: float = 1.6

    duty_cycle: float = 0.6
    # gait lift_height
    height: float = 0.25 ## lift_height for four beats(crawl, bound)  (腾空，height)

    @configclass
    class Ranges(commands_cfg.UniformVelocityCommandCfg.Ranges):
        frequencie: tuple[float, float] = MISSING
        duty_cycle: tuple[float, float] = MISSING
        height: tuple[float, float] = MISSING

    ranges: Ranges = MISSING


@configclass
class PosGaitCommandCfg(CurriculumCommandCfg):
    class_type: type = pos_command.PosGaitCommand

    # pos height control [0.8 ~ 1]
    pos_height: float = MISSING
    # pitch roll control [- 30 / 180 ~   30 / 180]
    pos_pitch: float = MISSING
    # pos roll control [- 30 / 180 ~   30 / 180]
    pos_roll: float = MISSING

    # gait frequencie
    gait_frequencie: float = MISSING

    gait_duty_cycle: float = MISSING
    # gait lift_height
    gait_height: float = MISSING ## lift_height for four beats(crawl, bound)  (腾空，height)
    # gait style
    '''
    四节拍   four beats(crawl 爬， bound 跳跃)
    左右对称，前后同步 (pace 踱步，
    左右反对称，斜对脚同步(trot 小跑，
    全同步(leap 跳跃，
    '''
    gait_style: int = MISSING  #

    @configclass
    class Ranges(commands_cfg.UniformVelocityCommandCfg.Ranges):

        pos_height: tuple[float, float] = MISSING
        pos_pitch: tuple[float, float] = MISSING
        pos_roll: tuple[float, float] = MISSING

        gait_frequencie: tuple[float, float] = MISSING
        gait_duty_cycle: tuple[float, float] = MISSING
        gait_height: tuple[float, float] = MISSING

@configclass
class BipedsStyleCommandCfg(CurriculumCommandCfg):
    class_type: type = gait_bipedscommand.BipedsStyleCommand

    # gait style
    '''
    同步
    不同步
    '''
    style: int = 0  #
    # gait frequencie
    frequencie: float = 1.6

    duty_cycle: float = 0.6
    # gait lift_height
    height: float = 0.25

    @configclass
    class Ranges(commands_cfg.UniformVelocityCommandCfg.Ranges):
        frequencie: tuple[float, float] = MISSING
        duty_cycle: tuple[float, float] = MISSING
        height: tuple[float, float] = MISSING

    ranges: Ranges = MISSING