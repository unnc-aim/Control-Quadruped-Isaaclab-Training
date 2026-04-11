import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise, AdditiveGaussianNoiseCfg as Gnoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from .mdp.terminations import joint_pos_out_of_manual_limit
from .mdp.terrain_cfg import PhantomX_ROUGH_TERRAINS_CFG
from .mdp import CPGPositionActionCfg

##
# Pre-defined configs
##
import sys
from pathlib import Path

_PROJECT_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_PATH))
from assets.Mastiff_CFG import Mastiff_CONFIG as _ROBOT_CONFIG

##
# Scene definition
##


@configclass
class MyTerrainSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=PhantomX_ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot: ArticulationCfg = _ROBOT_CONFIG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=_ROBOT_CONFIG.spawn.replace(activate_contact_sensors=True),
        init_state=_ROBOT_CONFIG.init_state.replace(
            pos=(0.0, 0.0, 0.6),
        ),
    )

    # Height scanner mounted at the robot base
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Mastiff/Body_v1",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=(2.5, 2.5)),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Mastiff/.*",
        history_length=3,
        track_air_time=True,
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 20.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    cpg = CPGPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        step_height=0.035,
        step_length=0.06,
        step_frequency=1.2,
        step_direction=1.0,
        step_height_min=0.0,
        step_height_max=0.08,
        step_length_min=0.0,
        step_length_max=0.12,
        step_frequency_min=0.0,
        step_frequency_max=3.0,
        center_offset=0.12,
        ground_height=-0.07,
        legs_config={
            "FL": {
                "coxa": "HAA_FRONT_LEFT",
                "femur": "HFE_FRONT_LEFT",
                "tibia": "KFE_FRONT_LEFT",
                "body_angle": -45.0,
                "phase_offset_deg": 0.0,
                "side": "left",
            },
            "FR": {
                "coxa": "HAA_FRONT_RIGHT",
                "femur": "HFE_FRONT_RIGHT",
                "tibia": "KFE_FRONT_RIGHT",
                "body_angle": 45.0,
                "phase_offset_deg": 180.0,
                "side": "right",
            },
            "RL": {
                "coxa": "HAA_REAR_LEFT",
                "femur": "HFE_REAR_LEFT",
                "tibia": "KFE_REAR_LEFT",
                "body_angle": -135.0,
                "phase_offset_deg": 180.0,
                "side": "left",
            },
            "RR": {
                "coxa": "HAA_REAR_RIGHT",
                "femur": "HFE_REAR_RIGHT",
                "tibia": "KFE_REAR_RIGHT",
                "body_angle": 135.0,
                "phase_offset_deg": 0.0,
                "side": "right",
            },
        },
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_orientation = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.001, n_max=0.001),
            clip=(-1.0, 1.0),
        )
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for randomization events."""

    # Startup events
    # physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "static_friction_range": (0.6, 1.0),
    #         "dynamic_friction_range": (0.6, 0.8),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 64,
    #     },
    # )
    # add_base_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="Body_v1"),
    #         "mass_distribution_params": (-1.0, 2.0),
    #         "operation": "add",
    #     },
    # )

    # Reset events
    # base_external_force_torque = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="Body_v1"),
    #         "force_range": (0.0, 0.0),
    #         "torque_range": (-0.0, 0.0),
    #     },
    # )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.25, 0.25),
            "velocity_range": (0.0, 0.0),
        },
    )

    # Interval events (push disturbances)
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(10.0, 15.0),
    #     params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # ---------- 正向奖励 ----------
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=5.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # track_ang_vel_z_exp = RewTerm(
    #     func=mdp.track_ang_vel_z_exp,
    #     weight=20.5,
    #     params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    # )
    # threshold 从 0.5s → 0.2s：
    #   - 0.5s 的阈值下，两腿长时间腾空也能持续得分，诱导"蹦跳"
    #   - 0.2s 与对角小跑(trot)的单步腾空时间匹配，迫使机器人频繁交替接地
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=1.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="Foot_.*"),
            "command_name": "base_velocity",
            "threshold": 0.2,
        },
    )

    # ---------- 惩罚项 ----------
    # 竖向速度：蹦跳时 vz 幅度大，加重惩罚
    # lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)

    # 俯仰/横滚角速度：蹦跳/扭胯时此项很大，是抑制两足蹦的关键
    # 原来 -0.05 太弱，改为 -0.5
    # ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)

    # 脚滑惩罚
    # feet_slide = RewTerm(
    #     func=mdp.feet_slide,
    #     weight=-0.25,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="Foot_.*"),
    #         "asset_cfg": SceneEntityCfg("robot", body_names="Foot_.*"),
    #     },
    # )

    # 身体保持水平：地形上适当放宽（原 -2.5 flat task → -1.0）
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)

    # 目标站立高度：防止机器人半蹲/"趴着走"
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-20.0,
        params={"target_height": 0.65},
    )

    # 关节加速度：平滑关节运动，抑制抽搐/急停急走
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)

    # 关节扭矩：减少不必要的力输出，使步态更省力
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-7)

    # 动作变化率：相邻帧动作差异过大则惩罚，抑制抖动
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.002)

    # 大腿非期望接触（软约束替代硬终止）
    undesired_thigh_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="Thigh.*"),
            "threshold": 1.0,
        },
    )



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    command_update = DoneTerm(
        func=mdp.command_resample, params={"command_name": "base_velocity"}
    )
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="Body_v1"),
            "threshold": 1.0,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment configuration
##


@configclass
class MastiffTerrainEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Mastiff locomotion velocity-tracking environment on rough terrain."""

    # Scene settings
    scene: MyTerrainSceneCfg = MyTerrainSceneCfg(num_envs=4096, env_spacing=3.0)
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
        self.decimation = 4  # 120Hz / 4 = 30Hz RL control frequency
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1.0 / 120.0  # 120Hz physics simulation
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**16
        # Convex Decomposition 产生更多碰撞图元，需要更大的 collision stack
        self.sim.physx.gpu_collision_stack_size = 128 * 2**20  # 128 MB（默认约16MB不够）
        # sensor update periods
        if self.scene.contact_sensor is not None:
            self.scene.contact_sensor.update_period = self.sim.dt
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.sim.dt * 2  # 60 Hz


@configclass
class MastiffTerrainEnvCfg_PLAY(MastiffTerrainEnvCfg):
    """Play configuration: fewer envs, visualization enabled."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = 5
        self.scene.height_scanner.debug_vis = True
        self.commands.base_velocity.resampling_time_range = (10000.0, 10000.0)
        self.curriculum.terrain_levels = None  # type: ignore[assignment]
