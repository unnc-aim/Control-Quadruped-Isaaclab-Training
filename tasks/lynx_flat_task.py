import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from .mdp import QuadrupedGaitActionCfg
from .mdp.quadruped_gait_generator import QuadrupedGaitGenerator, QuadrupedGeometry
from . import mdp as custom_mdp

import sys
from pathlib import Path

import torch

_PROJECT_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_PATH))
from assets.Lynxc_CFG import Lynxc_CONFIG as _ROBOT_CONFIG

DESIRED_BASE_HEIGHT_M = float(_ROBOT_CONFIG.init_state.pos[2])
CPG_GROUND_HEIGHT_M = -0.25
USE_ZERO_POSE_STANDING_TARGET = True
LYNX_L_COXA_M = 0.075
LYNX_L_FEMUR_M = math.sqrt(0.0602**2 + 0.22**2)
LYNX_L_TIBIA_M = math.sqrt(0.303431**2 + 0.0455**2 + 0.03**2)
LYNX_BODY_LENGTH_M = 0.260
LYNX_BODY_WIDTH_M = 0.150
LYNX_TERMINAL_CONTACT_BODIES = ["Foot_02", "Calf_02", "Calf", "Calf_03"]
LYNX_HIP_BODIES = ["Hip", "Hip_03", "Hip_02", "Hip_01"]
LYNX_THIGH_BODIES = ["Thigh_02", "Thigh", "Thigh_03", "Thigh_01"]
LYNX_BASE_BODIES = ["Body"]


def _resolve_lynx_standing_pose_deg() -> tuple[float, float, float]:
    geometry = QuadrupedGeometry(
        l_coxa=LYNX_L_COXA_M,
        l_femur=LYNX_L_FEMUR_M,
        l_tibia=LYNX_L_TIBIA_M,
        femur_zero_angle_global=math.radians(90.0),
        tibia_zero_angle_relative=math.radians(180.0),
    )
    side_signs = torch.tensor([+1.0, -1.0, +1.0, -1.0], dtype=torch.float64)
    generator = QuadrupedGaitGenerator(
        geometry=geometry,
        leg_order=("FL", "FR", "RL", "RR"),
        side_signs=side_signs,
        phase_offsets=[0.0, math.pi, math.pi, 0.0],
        device="cpu",
        dtype=torch.float64,
    )
    foot_targets = torch.tensor(
        [
            [0.020, +LYNX_L_COXA_M, -0.245],
            [0.020, -LYNX_L_COXA_M, -0.245],
            [0.020, +LYNX_L_COXA_M, -0.245],
            [0.020, -LYNX_L_COXA_M, -0.245],
        ],
        dtype=torch.float64,
    )
    standing_targets, valid_ik = generator.solve_ik(foot_targets, side_signs)
    if not bool(valid_ik.all()):
        raise RuntimeError("Failed to derive a valid Lynx standing pose.")
    standing_deg = torch.rad2deg(standing_targets[0]).tolist()
    return float(standing_deg[0]), float(standing_deg[1]), float(standing_deg[2])


if USE_ZERO_POSE_STANDING_TARGET:
    LYNX_STANDING_HAA_DEG = 0.0
    LYNX_STANDING_HFE_DEG = 0.0
    LYNX_STANDING_KFE_DEG = 0.0
else:
    LYNX_STANDING_HAA_DEG, LYNX_STANDING_HFE_DEG, LYNX_STANDING_KFE_DEG = _resolve_lynx_standing_pose_deg()


@configclass
class LynxSceneCfg(InteractiveSceneCfg):
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
        init_state=_ROBOT_CONFIG.init_state.replace(pos=(0.0, 0.0, DESIRED_BASE_HEIGHT_M)),
    )
    contact_sensor = ContactSensorCfg(
        # LynxC.usd is composed under /Robot/LynxC/* and the collision bodies live below that subtree.
        prim_path="{ENV_REGEX_NS}/Robot/LynxC/.*",
        history_length=3,
        track_air_time=True,
    )


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(20.0, 30.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0, 0.0),
            lin_vel_y=(0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0, 0.0)
            # lin_vel_x=(0.30, 0.60),
            # lin_vel_y=(-0.20, 0.20),
            # ang_vel_z=(-0.50, 0.50),
            # heading=(-math.pi / 2.0, math.pi / 2.0),
        ),
    )


@configclass
class ActionsCfg:
    cpg = QuadrupedGaitActionCfg(
        asset_name="robot",
        l_coxa=LYNX_L_COXA_M,
        l_femur=LYNX_L_FEMUR_M,
        l_tibia=LYNX_L_TIBIA_M,
        body_length=LYNX_BODY_LENGTH_M,
        body_width=LYNX_BODY_WIDTH_M,
        joint_names=[".*"],
        step_height=0.040,
        step_length=0.014,
        step_frequency=2.5,
        step_direction=1.0,
        gait_type="trot",
        step_height_min=0.0,
        step_height_max=0.080,
        step_length_min=0.0,
        step_length_max=0.140,
        step_frequency_min=0.0,
        step_frequency_max=3.0,
        command_name="base_velocity",
        command_speed_to_step_length=0.020,
        command_speed_to_frequency=0.100,
        command_ang_vel_to_turn_rate=0.250,
        command_min_step_length=0.020,
        yaw_step_length_max=0.020,
        step_height_residual_scale=0.008,
        step_length_residual_scale=0.012,
        step_frequency_residual_scale=0.2,
        turn_rate_residual_scale=0.1,
        debug_print_enabled=False,
        lock_base_in_air=True,
        femur_zero_angle_global_deg=90.0,
        tibia_zero_angle_relative_deg=180.0,
        standing_haa_deg=LYNX_STANDING_HAA_DEG,
        standing_hfe_deg=LYNX_STANDING_HFE_DEG,
        standing_kfe_deg=LYNX_STANDING_KFE_DEG,
        center_offset=0.020,
        ground_height=CPG_GROUND_HEIGHT_M,
        legs_config={
            "FL": {
                "coxa": "FL0",
                "femur": "FL1",
                "tibia": "FL2",
                "phase_offset_deg": 0.0,
                "side": "left",
                "haa_sign": +1.0,
                "hfe_sign": +1.0,
                "kfe_sign": +1.0,
                "hip_xy": (0.126157, 0.075),
            },
            "FR": {
                "coxa": "FR0",
                "femur": "FR1",
                "tibia": "FR2",
                "phase_offset_deg": 180.0,
                "side": "right",
                "haa_sign": -1.0,
                "hfe_sign": -1.0,
                "kfe_sign": -1.0,
                "hip_xy": (0.126157, -0.075),
            },
            "RL": {
                "coxa": "RL0",
                "femur": "RL1",
                "tibia": "RL2",
                "phase_offset_deg": 180.0,
                "side": "left",
                "haa_sign": +1.0,
                "hfe_sign": +1.0,
                "kfe_sign": +1.0,
                "hip_xy": (-0.133843, 0.075),
            },
            "RR": {
                "coxa": "RR0",
                "femur": "RR1",
                "tibia": "RR2",
                "phase_offset_deg": 0.0,
                "side": "right",
                "haa_sign": -1.0,
                "hfe_sign": -1.0,
                "kfe_sign": -1.0,
                "hip_xy": (-0.133843, -0.075),
            },
        },
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_orientation = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_orientation = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
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
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=LYNX_TERMINAL_CONTACT_BODIES),
            "command_name": "base_velocity",
            "threshold": 0.2,
        },
    )
    gait_diagonal_symmetry = RewTerm(
        func=custom_mdp.diagonal_gait_symmetry,
        weight=2.0,
        params={
            "fl_rr_cfg": SceneEntityCfg("robot", joint_names=["FL1", "FL2", "RR1", "RR2"]),
            "fr_rl_cfg": SceneEntityCfg("robot", joint_names=["FR1", "FR2", "RL1", "RL2"]),
            "scale": 10.0,
        },
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=LYNX_TERMINAL_CONTACT_BODIES),
            "asset_cfg": SceneEntityCfg("robot", body_names=LYNX_TERMINAL_CONTACT_BODIES),
        },
    )
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-10.0,
        params={"target_height": DESIRED_BASE_HEIGHT_M},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)
    undesired_Hip_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-50.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=LYNX_HIP_BODIES), "threshold": 1.0},
    )
    undesired_base_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-50.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=LYNX_BASE_BODIES), "threshold": 1.0},
    )
    undesired_Thigh_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-15.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=LYNX_THIGH_BODIES), "threshold": 1.0},
    )
    # undesired_Calf_contact = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=0.0,
    #     params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["Calf_01"]), "threshold": 1.0},
    # )
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-5.0e-6)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.0005)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    command_update = DoneTerm(func=mdp.command_resample, params={"command_name": "base_velocity"})
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=LYNX_BASE_BODIES), "threshold": 1.0},
    )
    Hip_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=LYNX_HIP_BODIES), "threshold": 1.0},
    )
    Thigh_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=LYNX_THIGH_BODIES), "threshold": 500.0},
    )


@configclass
class LynxFlatEnvCfg(ManagerBasedRLEnvCfg):
    scene: LynxSceneCfg = LynxSceneCfg(num_envs=4096, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**16
        contact_sensor = getattr(self.scene, "contact_sensor", None)
        if contact_sensor is not None:
            contact_sensor.update_period = self.sim.dt
