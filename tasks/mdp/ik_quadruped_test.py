from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MDP_DIR = Path(__file__).resolve().parent
ISAACLAB_SOURCE_PATH = Path.home() / "IsaacLab" / "source" / "isaaclab"

if ISAACLAB_SOURCE_PATH.exists():
    sys.path.insert(0, str(ISAACLAB_SOURCE_PATH))


def _install_isaaclab_import_shims() -> None:
    isaaclab_pkg = sys.modules.setdefault("isaaclab", types.ModuleType("isaaclab"))

    assets_pkg = sys.modules.setdefault("isaaclab.assets", types.ModuleType("isaaclab.assets"))
    articulation_mod = sys.modules.setdefault(
        "isaaclab.assets.articulation",
        types.ModuleType("isaaclab.assets.articulation"),
    )
    managers_pkg = sys.modules.setdefault("isaaclab.managers", types.ModuleType("isaaclab.managers"))
    action_manager_mod = sys.modules.setdefault(
        "isaaclab.managers.action_manager",
        types.ModuleType("isaaclab.managers.action_manager"),
    )
    utils_mod = sys.modules.setdefault("isaaclab.utils", types.ModuleType("isaaclab.utils"))

    class _StubArticulation:
        pass

    class _StubActionTerm:
        def __init__(self, cfg=None, env=None):
            self.cfg = cfg
            self._env = env
            self.env = env

    class _StubActionTermCfg:
        pass

    def _configclass(cls):
        return cls

    articulation_mod.Articulation = getattr(articulation_mod, "Articulation", _StubArticulation)
    action_manager_mod.ActionTerm = getattr(action_manager_mod, "ActionTerm", _StubActionTerm)
    action_manager_mod.ActionTermCfg = getattr(action_manager_mod, "ActionTermCfg", _StubActionTermCfg)
    utils_mod.configclass = getattr(utils_mod, "configclass", _configclass)

    assets_pkg.articulation = articulation_mod
    managers_pkg.action_manager = action_manager_mod
    isaaclab_pkg.assets = assets_pkg
    isaaclab_pkg.managers = managers_pkg
    isaaclab_pkg.utils = utils_mod


def _load_mdp_modules():
    _install_isaaclab_import_shims()
    tasks_pkg = types.ModuleType("tasks")
    tasks_pkg.__path__ = [str(PROJECT_ROOT / "tasks")]
    sys.modules.setdefault("tasks", tasks_pkg)

    mdp_pkg = types.ModuleType("tasks.mdp")
    mdp_pkg.__path__ = [str(MDP_DIR)]
    sys.modules.setdefault("tasks.mdp", mdp_pkg)

    generator_name = "tasks.mdp.quadruped_gait_generator"
    generator_path = MDP_DIR / "quadruped_gait_generator.py"
    generator_spec = importlib.util.spec_from_file_location(generator_name, generator_path)
    if generator_spec is None or generator_spec.loader is None:
        raise RuntimeError(f"Failed to load {generator_name} from {generator_path}")
    generator_module = importlib.util.module_from_spec(generator_spec)
    sys.modules[generator_name] = generator_module
    generator_spec.loader.exec_module(generator_module)

    action_name = "tasks.mdp.quadruped_gait_action"
    action_path = MDP_DIR / "quadruped_gait_action.py"
    action_spec = importlib.util.spec_from_file_location(action_name, action_path)
    if action_spec is None or action_spec.loader is None:
        raise RuntimeError(f"Failed to load {action_name} from {action_path}")
    action_module = importlib.util.module_from_spec(action_spec)
    sys.modules[action_name] = action_module
    action_spec.loader.exec_module(action_module)
    return generator_module, action_module


_generator_module, _action_module = _load_mdp_modules()
QuadrupedGaitGenerator = _generator_module.QuadrupedGaitGenerator
QuadrupedGeometry = _generator_module.QuadrupedGeometry
QuadrupedGaitAction = _action_module.QuadrupedGaitAction
QuadrupedGaitActionCfg = _action_module.QuadrupedGaitActionCfg

# ==================== 1. Geometry and action config ====================

MM_PER_M = 1000.0
L_COXA_M = 0.12005
L_FEMUR_M = 0.260
L_TIBIA_M = 0.300

BODY_LENGTH_M = 0.520
BODY_WIDTH_M = 0.180
HIP_Z_M = 0.0

LEG_ORDER = ("FL", "FR", "RL", "RR")
LEG_CONFIGS = {
    "FL": {
        "hip": np.array([BODY_LENGTH_M / 2.0, BODY_WIDTH_M / 2.0, HIP_Z_M]),
        "color": "tab:blue",
    },
    "FR": {
        "hip": np.array([BODY_LENGTH_M / 2.0, -BODY_WIDTH_M / 2.0, HIP_Z_M]),
        "color": "tab:orange",
    },
    "RL": {
        "hip": np.array([-BODY_LENGTH_M / 2.0, BODY_WIDTH_M / 2.0, HIP_Z_M]),
        "color": "tab:green",
    },
    "RR": {
        "hip": np.array([-BODY_LENGTH_M / 2.0, -BODY_WIDTH_M / 2.0, HIP_Z_M]),
        "color": "tab:red",
    },
}

JOINT_NAME_ORDER = [
    "HAA_FRONT_LEFT",
    "HAA_FRONT_RIGHT",
    "HAA_REAR_LEFT",
    "HAA_REAR_RIGHT",
    "HFE_FRONT_LEFT",
    "HFE_FRONT_RIGHT",
    "HFE_REAR_LEFT",
    "HFE_REAR_RIGHT",
    "KFE_FRONT_LEFT",
    "KFE_FRONT_RIGHT",
    "KFE_REAR_LEFT",
    "KFE_REAR_RIGHT",
]

HEADLESS_OUTPUT_GIF = Path("./ik_quadruped_action_planner.gif")
HEADLESS_OUTPUT_PNG = Path("/tmp/ik_quadruped_action_planner_first_frame.png")

# Mirror the current Mastiff flat-task action settings.
COMMAND = np.array([0.922, 0.010, 0.345], dtype=np.float64)
DT = 1.0 / 60.0
TOTAL_STEPS = 120


# ==================== 2. Lightweight action adapter ====================


class _NullAsset:
    def __init__(self, num_joints: int):
        self.num_joints = num_joints
        self.last_target = None

    def set_joint_position_target(self, target: torch.Tensor) -> None:
        self.last_target = target.clone()


class GaitActionPlannerAdapter:
    _resolve_zero_angle_deg = QuadrupedGaitAction._resolve_zero_angle_deg
    _resolve_standing_joint_targets = QuadrupedGaitAction._resolve_standing_joint_targets
    _compute_turn_step = QuadrupedGaitAction._compute_turn_step

    def __init__(self, cfg: QuadrupedGaitActionCfg, command_xyz: np.ndarray):
        self.cfg = cfg
        self.device = torch.device("cpu")
        self.num_envs = 1
        self._dt = DT
        self._step_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.float64)
        self._lock_base_in_air = False
        self._locked_root_pose = None
        self._locked_root_vel = None
        self._command = torch.tensor(command_xyz, dtype=torch.float64, device=self.device).unsqueeze(0)
        self._asset = _NullAsset(num_joints=len(JOINT_NAME_ORDER))
        self._last_debug = None

        femur_zero_deg = self._resolve_zero_angle_deg(
            cfg.femur_zero_angle_global_deg,
            cfg.femur_rest_angle_global_deg,
            "femur_zero_angle_global_deg",
            "femur_rest_angle_global_deg",
        )
        tibia_zero_deg = self._resolve_zero_angle_deg(
            cfg.tibia_zero_angle_relative_deg,
            cfg.tibia_rest_angle_relative_deg,
            "tibia_zero_angle_relative_deg",
            "tibia_rest_angle_relative_deg",
        )
        geometry = QuadrupedGeometry(
            l_coxa=cfg.l_coxa,
            l_femur=cfg.l_femur,
            l_tibia=cfg.l_tibia,
            femur_zero_angle_global=math.radians(femur_zero_deg),
            tibia_zero_angle_relative=math.radians(tibia_zero_deg),
        )

        self.legs: list[dict] = []
        leg_names: list[str] = []
        leg_side_signs: list[float] = []
        leg_phase_offsets: list[float] = []
        leg_hip_xy: list[tuple[float, float]] = []
        leg_joint_signs: list[tuple[float, float, float]] = []

        joint_name_to_idx = {name: idx for idx, name in enumerate(JOINT_NAME_ORDER)}
        for leg_name, leg_conf in cfg.legs_config.items():
            c_idx = joint_name_to_idx[leg_conf["coxa"]]
            f_idx = joint_name_to_idx[leg_conf["femur"]]
            t_idx = joint_name_to_idx[leg_conf["tibia"]]
            side_sign = 1.0 if str(leg_conf.get("side", "left")).lower() == "left" else -1.0
            phase_offset = math.radians(float(leg_conf.get("phase_offset_deg", 0.0)))
            hip_xy = tuple(float(v) for v in leg_conf.get("hip_xy", (0.0, side_sign * 0.5 * cfg.body_width)))
            joint_signs = (
                float(leg_conf.get("haa_sign", 1.0)),
                float(leg_conf.get("hfe_sign", 1.0)),
                float(leg_conf.get("kfe_sign", 1.0)),
            )
            self.legs.append(
                {
                    "name": leg_name,
                    "coxa_idx": c_idx,
                    "femur_idx": f_idx,
                    "tibia_idx": t_idx,
                    "side_sign": side_sign,
                    "phase_offset": phase_offset,
                    "hip_xy": hip_xy,
                    "joint_signs": joint_signs,
                }
            )
            leg_names.append(leg_name)
            leg_side_signs.append(side_sign)
            leg_phase_offsets.append(phase_offset)
            leg_hip_xy.append(hip_xy)
            leg_joint_signs.append(joint_signs)

        self._leg_count = len(self.legs)
        self._generator = QuadrupedGaitGenerator(
            geometry=geometry,
            leg_order=tuple(leg_names),
            side_signs=leg_side_signs,
            phase_offsets=leg_phase_offsets,
            device=self.device,
            dtype=torch.float64,
        )
        self._leg_phases = torch.tensor(leg_phase_offsets, device=self.device, dtype=torch.float64).unsqueeze(0)
        self._initial_leg_phases = self._leg_phases[0].clone()
        self._leg_side_signs = torch.tensor(leg_side_signs, device=self.device, dtype=torch.float64)
        self._leg_hip_xy = torch.tensor(leg_hip_xy, device=self.device, dtype=torch.float64)
        self._leg_joint_signs = torch.tensor(leg_joint_signs, device=self.device, dtype=torch.float64)
        self._leg_coxa_indices = torch.tensor([leg["coxa_idx"] for leg in self.legs], device=self.device, dtype=torch.long)
        self._leg_femur_indices = torch.tensor([leg["femur_idx"] for leg in self.legs], device=self.device, dtype=torch.long)
        self._leg_tibia_indices = torch.tensor([leg["tibia_idx"] for leg in self.legs], device=self.device, dtype=torch.long)
        self._standing_joint_targets = self._resolve_standing_joint_targets()
        self._nominal_home_positions = self._generator.nominal_standing_foot_positions(
            self._standing_joint_targets,
            self._leg_side_signs,
        )

        self._processed_actions = torch.zeros(self.num_envs, len(JOINT_NAME_ORDER), device=self.device, dtype=torch.float64)
        self._raw_actions = torch.zeros(self.num_envs, self._leg_count * 4, device=self.device, dtype=torch.float64)
        residual_shape = (self.num_envs, self._leg_count)
        self._step_height_residual = torch.zeros(residual_shape, device=self.device, dtype=torch.float64)
        self._step_length_residual = torch.zeros(residual_shape, device=self.device, dtype=torch.float64)
        self._frequency_residual = torch.zeros(residual_shape, device=self.device, dtype=torch.float64)
        self._turn_rate_residual = torch.zeros(residual_shape, device=self.device, dtype=torch.float64)

    def _get_command(self) -> torch.Tensor:
        return self._command

    def _maybe_log_debug(
        self,
        command: torch.Tensor,
        step_vectors: torch.Tensor,
        foot_targets: torch.Tensor,
        raw_planner_joint_targets: torch.Tensor,
        planner_joint_targets: torch.Tensor,
        sim_joint_targets: torch.Tensor,
        active_mask: torch.Tensor,
        valid_ik: torch.Tensor,
    ) -> None:
        self._last_debug = {
            "command": command.detach().cpu().clone(),
            "step_vectors": step_vectors.detach().cpu().clone(),
            "foot_targets": foot_targets.detach().cpu().clone(),
            "raw_planner_joint_targets": raw_planner_joint_targets.detach().cpu().clone(),
            "planner_joint_targets": planner_joint_targets.detach().cpu().clone(),
            "sim_joint_targets": sim_joint_targets.detach().cpu().clone(),
            "active_mask": active_mask.detach().cpu().clone(),
            "valid_ik": valid_ik.detach().cpu().clone(),
            "phases": self._leg_phases.detach().cpu().clone(),
            "processed_actions": self._processed_actions.detach().cpu().clone(),
        }

    def step(self) -> dict[str, np.ndarray]:
        QuadrupedGaitAction.apply_actions(self)
        if self._last_debug is None:
            raise RuntimeError("QuadrupedGaitAction.apply_actions did not produce debug tensors.")

        planner_targets = self._last_debug["planner_joint_targets"][0].to(dtype=torch.float64)
        fk_points_local = self._generator.forward_kinematics(planner_targets, self._leg_side_signs)
        return {
            "phase_deg": np.degrees(self._last_debug["phases"][0].numpy()),
            "step_vectors_m": self._last_debug["step_vectors"][0].numpy(),
            "target_local_m": self._last_debug["foot_targets"][0].numpy(),
            "raw_planner_q_deg": np.degrees(self._last_debug["raw_planner_joint_targets"][0].numpy()),
            "planner_q_deg": np.degrees(planner_targets.numpy()),
            "sim_q_deg": np.degrees(self._last_debug["sim_joint_targets"][0].numpy()),
            "valid_ik": self._last_debug["valid_ik"][0].numpy(),
            "fk_local_m": fk_points_local.detach().cpu().numpy(),
        }


# ==================== 3. Config builder ====================


def build_action_cfg() -> QuadrupedGaitActionCfg:
    cfg = QuadrupedGaitActionCfg()
    cfg.l_coxa = L_COXA_M
    cfg.l_femur = L_FEMUR_M
    cfg.l_tibia = L_TIBIA_M
    cfg.femur_zero_angle_global_deg = -150.0
    cfg.tibia_zero_angle_relative_deg = 15.0
    cfg.standing_haa_deg = 0.0
    cfg.standing_hfe_deg = 0.0
    cfg.standing_kfe_deg = 40.0
    cfg.body_length = BODY_LENGTH_M
    cfg.body_width = BODY_WIDTH_M
    cfg.step_height = 0.03
    cfg.step_length = 0.18
    cfg.step_frequency = 2.0
    cfg.step_direction = 1.0
    cfg.center_offset = -0.0269
    cfg.ground_height = -0.35
    cfg.stand_when_command_zero = True
    cfg.default_forward_command = 0.0
    cfg.command_speed_to_step_length = 0.07
    cfg.command_speed_to_frequency = 0.1
    cfg.command_ang_vel_to_turn_rate = 0.0
    cfg.command_min_step_length = 0.03
    cfg.command_lin_speed_deadband = 1.0e-3
    cfg.yaw_step_length_max = 0.04
    cfg.step_height_min = 0.0
    cfg.step_height_max = 0.2
    cfg.step_length_min = 0.0
    cfg.step_length_max = 0.22
    cfg.step_frequency_min = 0.0
    cfg.step_frequency_max = 3.0
    cfg.step_height_residual_scale = 0.008
    cfg.step_length_residual_scale = 0.012
    cfg.step_frequency_residual_scale = 0.2
    cfg.turn_rate_residual_scale = 0.1
    cfg.debug_print_enabled = False
    cfg.clip_joint_targets = False
    cfg.lock_base_in_air = False
    cfg.legs_config = {
        "FL": {
            "coxa": "HAA_FRONT_LEFT",
            "femur": "HFE_FRONT_LEFT",
            "tibia": "KFE_FRONT_LEFT",
            "phase_offset_deg": 0.0,
            "side": "left",
            "haa_sign": +1.0,
            "hfe_sign": +1.0,
            "kfe_sign": -1.0,
            "hip_xy": (BODY_LENGTH_M / 2.0, BODY_WIDTH_M / 2.0),
        },
        "FR": {
            "coxa": "HAA_FRONT_RIGHT",
            "femur": "HFE_FRONT_RIGHT",
            "tibia": "KFE_FRONT_RIGHT",
            "phase_offset_deg": 180.0,
            "side": "right",
            "haa_sign": +1.0,
            "hfe_sign": -1.0,
            "kfe_sign": -1.0,
            "hip_xy": (BODY_LENGTH_M / 2.0, -BODY_WIDTH_M / 2.0),
        },
        "RL": {
            "coxa": "HAA_REAR_LEFT",
            "femur": "HFE_REAR_LEFT",
            "tibia": "KFE_REAR_LEFT",
            "phase_offset_deg": 180.0,
            "side": "left",
            "haa_sign": -1.0,
            "hfe_sign": +1.0,
            "kfe_sign": -1.0,
            "hip_xy": (-BODY_LENGTH_M / 2.0, BODY_WIDTH_M / 2.0),
        },
        "RR": {
            "coxa": "HAA_REAR_RIGHT",
            "femur": "HFE_REAR_RIGHT",
            "tibia": "KFE_REAR_RIGHT",
            "phase_offset_deg": 0.0,
            "side": "right",
            "haa_sign": -1.0,
            "hfe_sign": -1.0,
            "kfe_sign": -1.0,
            "hip_xy": (-BODY_LENGTH_M / 2.0, -BODY_WIDTH_M / 2.0),
        },
    }
    return cfg


# ==================== 4. Plot helpers ====================


def hip_world_positions_m() -> np.ndarray:
    return np.stack([LEG_CONFIGS[leg_name]["hip"] for leg_name in LEG_ORDER], axis=0)


def set_equal_axes(ax, points_mm, margin_mm=60.0, tick_step_mm=100.0):
    mins = points_mm.min(axis=0) - margin_mm
    maxs = points_mm.max(axis=0) + margin_mm
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    half_span = 0.5 * span

    x_lim = (center[0] - half_span, center[0] + half_span)
    y_lim = (center[1] - half_span, center[1] + half_span)
    z_lim = (center[2] - half_span, center[2] + half_span)

    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_zlim(*z_lim)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_xticks(np.arange(tick_step_mm * np.floor(x_lim[0] / tick_step_mm), x_lim[1] + tick_step_mm, tick_step_mm))
    ax.set_yticks(np.arange(tick_step_mm * np.floor(y_lim[0] / tick_step_mm), y_lim[1] + tick_step_mm, tick_step_mm))
    ax.set_zticks(np.arange(tick_step_mm * np.floor(z_lim[0] / tick_step_mm), z_lim[1] + tick_step_mm, tick_step_mm))
    return x_lim, y_lim, z_lim


def add_body(ax):
    x_front = BODY_LENGTH_M * MM_PER_M / 2.0
    x_rear = -BODY_LENGTH_M * MM_PER_M / 2.0
    y_left = BODY_WIDTH_M * MM_PER_M / 2.0
    y_right = -BODY_WIDTH_M * MM_PER_M / 2.0
    z = HIP_Z_M * MM_PER_M

    body_vertices = [[
        [x_front, y_left, z],
        [x_front, y_right, z],
        [x_rear, y_right, z],
        [x_rear, y_left, z],
    ]]
    body = Poly3DCollection(body_vertices, facecolors="lightgray", edgecolors="black", linewidths=1.5, alpha=0.45)
    ax.add_collection3d(body)

    hip_points_mm = hip_world_positions_m() * MM_PER_M
    ax.scatter(hip_points_mm[:, 0], hip_points_mm[:, 1], hip_points_mm[:, 2], color="black", s=25, label="HAA origins")
    for leg_name, leg_cfg in LEG_CONFIGS.items():
        text_pos = leg_cfg["hip"] * MM_PER_M + np.array([0.0, 0.0, 25.0])
        ax.text(text_pos[0], text_pos[1], text_pos[2], leg_name, color=leg_cfg["color"], fontsize=10)


def add_ground_plane(ax, x_lim, y_lim, ground_z_mm):
    ground_vertices = [[
        [x_lim[0], y_lim[0], ground_z_mm],
        [x_lim[1], y_lim[0], ground_z_mm],
        [x_lim[1], y_lim[1], ground_z_mm],
        [x_lim[0], y_lim[1], ground_z_mm],
    ]]
    ground = Poly3DCollection(ground_vertices, facecolors="tab:green", edgecolors="none", alpha=0.08)
    ax.add_collection3d(ground)


# ==================== 5. Animation ====================


def run_animation():
    cfg = build_action_cfg()
    planner = GaitActionPlannerAdapter(cfg, COMMAND)

    frames = []
    hip_world_m = hip_world_positions_m()
    for _ in range(TOTAL_STEPS):
        frame = planner.step()
        frame["target_world_m"] = hip_world_m + frame["target_local_m"]
        frame["fk_world_m"] = hip_world_m[:, None, :] + frame["fk_local_m"]
        frames.append(frame)

    standing_targets_deg = np.array([
        [cfg.standing_haa_deg, cfg.standing_hfe_deg, cfg.standing_kfe_deg]
        for _ in LEG_ORDER
    ])
    standing_home_mm = planner._nominal_home_positions.detach().cpu().numpy() * MM_PER_M
    print("Standing planner joint targets (deg):")
    for leg_idx, leg_name in enumerate(LEG_ORDER):
        print(f"  {leg_name}: {np.round(standing_targets_deg[leg_idx], 3)}")
    print("Standing home foot local (mm):")
    for leg_idx, leg_name in enumerate(LEG_ORDER):
        print(f"  {leg_name}: {np.round(standing_home_mm[leg_idx], 3)}")
    print(f"Action-style command: {np.round(COMMAND, 3)}")
    print(f"Generated {len(frames)} frames via QuadrupedGaitAction.apply_actions()")

    target_paths_world_mm = {
        leg_name: np.stack([frame["target_world_m"][leg_idx] * MM_PER_M for frame in frames], axis=0)
        for leg_idx, leg_name in enumerate(LEG_ORDER)
    }
    fk_paths_world_mm = {
        leg_name: np.stack([frame["fk_world_m"][leg_idx, -1] * MM_PER_M for frame in frames], axis=0)
        for leg_idx, leg_name in enumerate(LEG_ORDER)
    }
    target_paths_local_mm = {
        leg_name: np.stack([frame["target_local_m"][leg_idx] * MM_PER_M for frame in frames], axis=0)
        for leg_idx, leg_name in enumerate(LEG_ORDER)
    }
    fk_paths_local_mm = {
        leg_name: np.stack([frame["fk_local_m"][leg_idx, -1] * MM_PER_M for frame in frames], axis=0)
        for leg_idx, leg_name in enumerate(LEG_ORDER)
    }

    fig = plt.figure(figsize=(15, 7))
    ax3d = fig.add_subplot(121, projection="3d")
    ax2d = fig.add_subplot(122)

    leg_lines = {}
    target_dots = {}
    fk_dots = {}
    local_target_dots = {}
    local_fk_dots = {}
    artists = []

    for leg_idx, leg_name in enumerate(LEG_ORDER):
        color = LEG_CONFIGS[leg_name]["color"]
        frame0 = frames[0]
        leg_points_world_mm = frame0["fk_world_m"][leg_idx] * MM_PER_M
        target_world_mm = frame0["target_world_m"][leg_idx] * MM_PER_M
        fk_world_mm = frame0["fk_world_m"][leg_idx, -1] * MM_PER_M

        leg_line, = ax3d.plot(
            leg_points_world_mm[:, 0],
            leg_points_world_mm[:, 1],
            leg_points_world_mm[:, 2],
            "-o",
            linewidth=3,
            color=color,
            markersize=5,
            label=f"{leg_name} leg",
        )
        target_dot = ax3d.scatter([target_world_mm[0]], [target_world_mm[1]], [target_world_mm[2]], color=color, s=45, marker="x")
        fk_dot = ax3d.scatter([fk_world_mm[0]], [fk_world_mm[1]], [fk_world_mm[2]], color=color, s=25, marker="o")

        ax3d.plot(
            target_paths_world_mm[leg_name][:, 0],
            target_paths_world_mm[leg_name][:, 1],
            target_paths_world_mm[leg_name][:, 2],
            color=color,
            alpha=0.35,
            linewidth=1.2,
            linestyle="--",
            label=f"{leg_name} target",
        )
        ax3d.plot(
            fk_paths_world_mm[leg_name][:, 0],
            fk_paths_world_mm[leg_name][:, 1],
            fk_paths_world_mm[leg_name][:, 2],
            color=color,
            alpha=0.6,
            linewidth=1.0,
            linestyle="-",
            label=f"{leg_name} FK",
        )

        ax2d.plot(
            target_paths_local_mm[leg_name][:, 0],
            target_paths_local_mm[leg_name][:, 2],
            color=color,
            linestyle="--",
            linewidth=1.4,
            label=f"{leg_name} target",
        )
        ax2d.plot(
            fk_paths_local_mm[leg_name][:, 0],
            fk_paths_local_mm[leg_name][:, 2],
            color=color,
            linestyle="-",
            linewidth=1.2,
            label=f"{leg_name} FK",
        )
        local_target_dot, = ax2d.plot([target_paths_local_mm[leg_name][0, 0]], [target_paths_local_mm[leg_name][0, 2]], marker="x", color=color, linestyle="None")
        local_fk_dot, = ax2d.plot([fk_paths_local_mm[leg_name][0, 0]], [fk_paths_local_mm[leg_name][0, 2]], marker="o", color=color, linestyle="None")

        leg_lines[leg_name] = leg_line
        target_dots[leg_name] = target_dot
        fk_dots[leg_name] = fk_dot
        local_target_dots[leg_name] = local_target_dot
        local_fk_dots[leg_name] = local_fk_dot
        artists.extend([leg_line, target_dot, fk_dot, local_target_dot, local_fk_dot])

    add_body(ax3d)
    vis_chunks_mm = [hip_world_m * MM_PER_M]
    for leg_name in LEG_ORDER:
        vis_chunks_mm.append(target_paths_world_mm[leg_name])
        vis_chunks_mm.append(fk_paths_world_mm[leg_name])
    vis_points_mm = np.vstack(vis_chunks_mm)
    x_lim, y_lim, z_lim = set_equal_axes(ax3d, vis_points_mm)
    add_ground_plane(ax3d, x_lim, y_lim, cfg.ground_height * MM_PER_M)

    ax3d.set_xlabel("X (mm, forward)")
    ax3d.set_ylabel("Y (mm, left)")
    ax3d.set_zlabel("Z (mm, up)")
    ax3d.set_title("QuadrupedGaitAction Planner on Simplified Model")
    ax3d.legend(loc="upper left", fontsize=8)
    ax3d.view_init(elev=24, azim=-48)

    local_points_mm = np.vstack([target_paths_local_mm[name][:, [0, 2]] for name in LEG_ORDER] + [fk_paths_local_mm[name][:, [0, 2]] for name in LEG_ORDER])
    local_min = local_points_mm.min(axis=0) - 20.0
    local_max = local_points_mm.max(axis=0) + 20.0
    ax2d.set_xlim(local_min[0], local_max[0])
    ax2d.set_ylim(local_min[1], local_max[1])
    ax2d.set_aspect("equal", adjustable="box")
    ax2d.grid(True, alpha=0.3)
    ax2d.set_xlabel("Local X (mm)")
    ax2d.set_ylabel("Local Z (mm)")
    ax2d.set_title("Target vs FK Foot Path (local x-z)")
    ax2d.legend(loc="best", fontsize=8, ncol=2)

    info_text = ax2d.text(
        0.02,
        0.98,
        "",
        transform=ax2d.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    artists.append(info_text)

    def update(frame_idx):
        frame = frames[frame_idx % len(frames)]
        info_lines = [
            f"frame={frame_idx:03d}",
            f"cmd=({COMMAND[0]:+.3f},{COMMAND[1]:+.3f},{COMMAND[2]:+.3f})",
        ]
        for leg_idx, leg_name in enumerate(LEG_ORDER):
            leg_points_world_mm = frame["fk_world_m"][leg_idx] * MM_PER_M
            target_world_mm = frame["target_world_m"][leg_idx] * MM_PER_M
            fk_world_mm = frame["fk_world_m"][leg_idx, -1] * MM_PER_M
            target_local_mm = frame["target_local_m"][leg_idx] * MM_PER_M
            fk_local_mm = frame["fk_local_m"][leg_idx, -1] * MM_PER_M

            leg_lines[leg_name].set_data(leg_points_world_mm[:, 0], leg_points_world_mm[:, 1])
            leg_lines[leg_name].set_3d_properties(leg_points_world_mm[:, 2])
            target_dots[leg_name]._offsets3d = ([target_world_mm[0]], [target_world_mm[1]], [target_world_mm[2]])
            fk_dots[leg_name]._offsets3d = ([fk_world_mm[0]], [fk_world_mm[1]], [fk_world_mm[2]])
            local_target_dots[leg_name].set_data([target_local_mm[0]], [target_local_mm[2]])
            local_fk_dots[leg_name].set_data([fk_local_mm[0]], [fk_local_mm[2]])

            phase_deg = frame["phase_deg"][leg_idx]
            info_lines.append(
                f"{leg_name} phase={phase_deg:6.1f} "
                f"target=({target_local_mm[0]:+6.1f},{target_local_mm[2]:+6.1f}) "
                f"fk=({fk_local_mm[0]:+6.1f},{fk_local_mm[2]:+6.1f})"
            )
        info_text.set_text("\n".join(info_lines))
        return artists

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=60, blit=False)

    backend = plt.get_backend().lower()
    if "agg" in backend:
        print(f"Non-interactive backend detected: {plt.get_backend()}")
        try:
            writer = animation.PillowWriter(fps=20)
            ani.save(HEADLESS_OUTPUT_GIF, writer=writer)
            print(f"Saved animation to {HEADLESS_OUTPUT_GIF}")
        except Exception as exc:
            print(f"GIF export failed: {exc}")
            update(0)
            fig.savefig(HEADLESS_OUTPUT_PNG, dpi=160, bbox_inches="tight")
            print(f"Saved first frame to {HEADLESS_OUTPUT_PNG}")
        finally:
            plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    run_animation()
