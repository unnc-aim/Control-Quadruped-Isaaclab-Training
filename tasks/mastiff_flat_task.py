import math
from dataclasses import MISSING
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
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
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from .mdp.terminations import joint_pos_out_of_manual_limit
from .mdp import CPGPositionActionCfg
from . import mdp as custom_mdp
##
# Pre-defined configs
##

import sys
from pathlib import Path
_PROJECT_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_PATH))
from assets.Mastiff_CFG import Mastiff_CONFIG as _ROBOT_CONFIG

# Height-related defaults for easier tuning.
DESIRED_BASE_HEIGHT_M = 0.35
CPG_GROUND_HEIGHT_M = -0.35

# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0]),
        spawn=GroundPlaneCfg(),
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
            pos=(0.0, 0.0, DESIRED_BASE_HEIGHT_M),
        ),
    )
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Mastiff/Body_v1",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=(2.5, 2.5)),
        debug_vis=False,
        mesh_prim_paths=["/World/GroundPlane"],
    )
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Mastiff/.*",
        history_length=3,
        track_air_time=True
    )
       
    
    
##
# MDP settings
##


@configclass
class CommandsCfg:
    # """Command specifications for the MDP."""
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(20.0, 30.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(1.0, 1.0),
            lin_vel_y=(0, 0),
            ang_vel_z=(0, 0),
            heading=(0,0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    cpg = CPGPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        step_height=0.1,
        step_length=0.2,
        step_frequency=1.2,
        step_direction=1.0,
        gait_type="trot",
        swing_vel_limits=(0.1, -0.2),
        step_height_min=0.0,
        step_height_max=0.08,
        step_length_min=0.0,
        step_length_max=0.12,
        step_frequency_min=0.0,
        step_frequency_max=3.0,
        command_name="base_velocity",
        command_speed_to_step_length=0.07,
        command_speed_to_frequency=0.1,
        command_ang_vel_to_turn_rate=0.0,
        command_min_step_length=0.03,
        yaw_step_length_max=0.04,
        step_height_residual_scale=0.008,
        step_length_residual_scale=0.012,
        step_frequency_residual_scale=0.2,
        turn_rate_residual_scale=0.1,
        debug_print_enabled=True,
        debug_print_interval=120,
        debug_env_index=0,
        swap_haa_hfe_targets=False,
        center_offset=-0.0269,
        ground_height=CPG_GROUND_HEIGHT_M,
        stance_depth=0.01,
        legs_config={
            "FL": {
                "coxa": "HAA_FRONT_LEFT",
                "femur": "HFE_FRONT_LEFT",
                "tibia": "KFE_FRONT_LEFT",
                "body_angle": 0.0,
                "phase_offset_deg": 0.0,
                "direction_multiplier": 1.0,
                "side": "left",
            },
            "FR": {
                "coxa": "HAA_FRONT_RIGHT",
                "femur": "HFE_FRONT_RIGHT",
                "tibia": "KFE_FRONT_RIGHT",
                "body_angle": 0.0,
                "phase_offset_deg": 180.0,
                "direction_multiplier": 1.0,
                "side": "right",
            },
            "RL": {
                "coxa": "HAA_REAR_LEFT",
                "femur": "HFE_REAR_LEFT",
                "tibia": "KFE_REAR_LEFT",
                "body_angle": 0.0,
                "phase_offset_deg": 180.0,
                "direction_multiplier": 1.0,
                "side": "left",
            },
            "RR": {
                "coxa": "HAA_REAR_RIGHT",
                "femur": "HFE_REAR_RIGHT",
                "tibia": "KFE_REAR_RIGHT",
                "body_angle": 0.0,
                "phase_offset_deg": 0.0,
                "direction_multiplier": 1.0,
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
        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_orientation = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_orientation = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
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

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""
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


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # Avoid "free" survival reward that can trap policy in crouching local optimum.
    alive = RewTerm(func=mdp.is_alive, weight=0.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=8.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=2.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="Foot_.*"),
            "command_name": "base_velocity",
            "threshold": 0.2,
        },
    )
    gait_diagonal_symmetry = RewTerm(
        func=custom_mdp.diagonal_gait_symmetry,
        # Gait timing is encoded by CPG phase settings; keep this term off by default.
        weight=2.0,
        params={
            # FL/RR
            "fl_rr_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "HFE_FRONT_LEFT",
                    "KFE_FRONT_LEFT",
                    "HFE_REAR_RIGHT",
                    "KFE_REAR_RIGHT",
                ],
            ),
            # FR/RL
            "fr_rl_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "HFE_FRONT_RIGHT",
                    "KFE_FRONT_RIGHT",
                    "HFE_REAR_LEFT",
                    "KFE_REAR_LEFT",
                ],
            ),
            "scale": 10.0,
        },
    )
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="Foot_.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="Foot_.*"),
        },
    )
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-10.0,
        params={"target_height": DESIRED_BASE_HEIGHT_M},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)

    undesired_hip_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-50.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="Hip_.*"),
            "threshold": 1.0,
        },
    )

    undesired_base_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-50.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="Body_v1"),
            "threshold": 1.0,
        },
    )
    undesired_thigh_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-15.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="Thigh.*"),
            "threshold": 1.0,
        },
    )
    
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-5.0e-6)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.002)
    # -- optional penalties
    # dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)


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
    hip_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="Hip_.*"),
            "threshold": 1.0,
        },
    )
    thigh_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="Thigh.*"),
            "threshold": 500.0,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass
    # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment configuration
##


@configclass
class MastiffFlatEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Mastiff locomotion velocity-tracking environment."""
    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=3.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4  # 120Hz / 4 = 30Hz RL control frequency
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1.0 / 120.0  # 120Hz physics simulation
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**16
        # sensor update periods
        if self.scene.contact_sensor is not None:
            self.scene.contact_sensor.update_period = self.sim.dt
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.sim.dt * 2
        
