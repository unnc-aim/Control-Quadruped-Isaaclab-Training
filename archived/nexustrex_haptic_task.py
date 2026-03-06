# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(project_root)
from assets.NexusTrex_CFG import NexusTrex_CONFIG  # Scene definition
from .mdp.terrain_cfg import NEXUS_ROUGH_TERRAINS_CFG  # Terrain configuration
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

    # # ground terrain
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="generator",
    #     terrain_generator=NEXUS_ROUGH_TERRAINS_CFG,
    #     max_init_terrain_level=5,
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #     ),
    #     visual_material=sim_utils.MdlFileCfg(
    #         mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
    #         project_uvw=True,
    #         texture_scale=(0.25, 0.25),
    #     ),
    #     debug_vis=False,
    # )
    # robots

    robot: ArticulationCfg = NexusTrex_CONFIG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=NexusTrex_CONFIG.spawn.replace(activate_contact_sensors=True),
    )
    
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/BipedalWheeledRobot/.*",
        history_length=3,
        track_air_time=True
    )
    
    # height_scanner = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/BipedalWheeledRobot/chassis_base",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    #     ray_alignment="yaw",
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    #     debug_vis=True,
    #     mesh_prim_paths=["/World/ground"],
    # )
    
    


##
# MDP settings
##


@configclass
class CommandsCfg:
    # """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(0.0,0.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    left_leg_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["DriveLeftMotor", "DriveLeftPassive"],
        scale=0.5,
        use_default_offset=True,
    )
    right_leg_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["DriveRightMotor", "DriveRighttPassive"],
        scale=0.5,
        use_default_offset=True,
    )
    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["MotorRight", "MotorLeft"],
        scale=10.0,
        use_default_offset=True,
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
        # joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        joint_effort = ObsTerm(func=mdp.joint_effort)
        actions = ObsTerm(func=mdp.last_action)
        # height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     clip=(-1.0, 1.0),
        # )
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


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

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminated = RewTerm(
        func=mdp.is_terminated_term,
        weight=-100.0,
        params={"term_keys": "out_of_joint_limit"}
    )
    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # wheel_ground_1 = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=1.0,
    #     params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names= ['wheel_01']), "threshold": 1.0},
    # )   
    # wheel_ground = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=1.0,
    #     params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names= ['wheel']), "threshold": 1.0},
    # )   
    # -- penalties
    # lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    # ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2, weight=-0.1, params={"target_height": 0.2}
    )
    # dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    # dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-10.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="chassis_base"), "threshold": 1.0},
    )
    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    # dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    command_update = DoneTerm(
        func=mdp.command_resample, params={"command_name": "base_velocity"}
    )
    # base_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="chassis_base"),
    #         "threshold": 1.0,
    #     },
    # )
    out_of_joint_limit = DoneTerm(
        func=joint_pos_out_of_manual_limit,
        params={
            "bounds": (-0.525, 0.525),  
            "joint_names": ['DriveRightMotor', 'DriveRighttPassive', 'DriveLeftPassive', 'DriveLeftMotor']
        }
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
class NexusTrexBasicEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
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
        self.decimation = 4
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        # self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        if self.scene.contact_sensor is not None:
            self.scene.contact_sensor.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        # if getattr(self.curriculum, "terrain_levels", None) is not None:
        #     if self.scene.terrain.terrain_generator is not None:
        #         self.scene.terrain.terrain_generator.curriculum = True
        # else:
        #     if self.scene.terrain.terrain_generator is not None:
        #         self.scene.terrain.terrain_generator.curriculum = False
