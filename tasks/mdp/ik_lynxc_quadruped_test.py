from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path
import os

import matplotlib


def _configure_matplotlib_backend() -> None:
    if os.environ.get("MPLBACKEND"):
        return

    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
    if has_display or session_type in {"x11", "wayland"}:
        for candidate in ("TkAgg", "QtAgg", "Qt5Agg"):
            try:
                matplotlib.use(candidate, force=True)
                return
            except Exception:
                continue

    matplotlib.use("Agg", force=True)


_configure_matplotlib_backend()

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import CheckButtons, Slider
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

MM_PER_M = 1000.0
L_COXA_M = 0.075
L_FEMUR_M = math.sqrt(0.0602**2 + 0.22**2)
L_TIBIA_M = math.sqrt(0.303431**2 + 0.0455**2 + 0.03**2)

LEG_ORDER = ("FL", "FR", "RL", "RR")
LEG_CONFIGS = {
    "FL": {"hip": np.array([0.126157, 0.075, 0.0]), "color": "tab:blue"},
    "FR": {"hip": np.array([0.126157, -0.075, 0.0]), "color": "tab:orange"},
    "RL": {"hip": np.array([-0.133843, 0.075, 0.0]), "color": "tab:green"},
    "RR": {"hip": np.array([-0.133843, -0.075, 0.0]), "color": "tab:red"},
}
JOINT_NAME_ORDER = [
    "fl0", "fl1", "fl2",
    "fr0", "fr1", "fr2",
    "rl0", "rl1", "rl2",
    "rr0", "rr1", "rr2",
]
JOINT0_SLIDER_LEGS = ("FL", "FR", "RL", "RR")
GAIT_TYPES = ("trot", "walk")
GAIT_LINE_STYLES = {"trot": "-", "walk": "--"}
GAIT_MARKERS = {"trot": "o", "walk": "^"}
GAIT_ALPHAS = {"trot": 0.95, "walk": 0.55}
HEADLESS_OUTPUT_PNG = Path("ik_lynxc_quadruped_preview.png")
COMMAND = np.array([0.45, 0.0, 0.0], dtype=np.float64)
DT = 1.0 / 60.0
HISTORY_LEN = 240


class _NullAsset:
    def __init__(self, num_joints: int):
        self.num_joints = num_joints
        self.last_target = None

    def set_joint_position_target(self, target: torch.Tensor) -> None:
        self.last_target = target.clone()


class LynxcGaitPlannerAdapter:
    _resolve_zero_angle_deg = QuadrupedGaitAction._resolve_zero_angle_deg
    _resolve_standing_joint_targets = QuadrupedGaitAction._resolve_standing_joint_targets
    _compute_turn_step = QuadrupedGaitAction._compute_turn_step
    _phase_offset_from_config = QuadrupedGaitAction._phase_offset_from_config

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
            phase_offset = self._phase_offset_from_config(leg_name, leg_conf)
            hip_xy = tuple(float(v) for v in leg_conf.get("hip_xy", (0.0, side_sign * 0.075)))
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
            knee_direction_signs=[-1.0 if name.upper().startswith("F") else 1.0 for name in leg_names],
            gait_type=cfg.gait_type,
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
        self._standing_joint_targets = self._resolve_inner_knee_standing_joint_targets()
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
        self._manual_haa_sim_offsets_rad = torch.zeros(self._leg_count, device=self.device, dtype=torch.float64)

    def _get_command(self) -> torch.Tensor:
        return self._command

    def _apply_startup_standing_blend(self, sim_joint_targets: torch.Tensor) -> torch.Tensor:
        return sim_joint_targets

    def _resolve_inner_knee_standing_joint_targets(self) -> torch.Tensor:
        foot_targets = torch.stack(
            (
                torch.full_like(self._leg_side_signs, float(self.cfg.center_offset)),
                self._leg_side_signs * float(self.cfg.l_coxa),
                torch.full_like(self._leg_side_signs, float(self.cfg.ground_height)),
            ),
            dim=-1,
        )
        standing_joint_targets, valid_ik = self._generator.solve_ik(foot_targets, self._leg_side_signs)
        if not bool(valid_ik.all()):
            raise RuntimeError("Failed to derive a valid inner-knee Lynxc standing pose from default foot targets.")
        return standing_joint_targets.to(dtype=torch.float64)

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
        }

    def set_manual_haa_offsets_deg(self, offsets_deg_by_leg: dict[str, float]) -> None:
        values = []
        for leg in self.legs:
            leg_name = leg["name"]
            sim_offset_rad = math.radians(float(offsets_deg_by_leg.get(leg_name, 0.0)))
            planner_offset_rad = sim_offset_rad * leg["joint_signs"][0]
            values.append(planner_offset_rad)
        self._manual_haa_sim_offsets_rad = torch.tensor(values, device=self.device, dtype=torch.float64)

    def reset_gait_cycle(self) -> None:
        self._step_counter.zero_()
        self._leg_phases[0] = self._initial_leg_phases.clone()

    def step(self, gait_enabled: bool) -> dict[str, np.ndarray]:
        if gait_enabled:
            QuadrupedGaitAction.apply_actions(self)
            if self._last_debug is None:
                raise RuntimeError("QuadrupedGaitAction.apply_actions did not produce debug tensors.")
            planner_targets = self._last_debug["planner_joint_targets"][0].to(dtype=torch.float64).clone()
            target_local = self._last_debug["foot_targets"][0].to(dtype=torch.float64)
            valid_ik = self._last_debug["valid_ik"][0].detach().cpu().numpy()
            phase_deg = np.degrees(self._last_debug["phases"][0].numpy())
        else:
            planner_targets = self._standing_joint_targets.clone()
            valid_ik = np.ones(self._leg_count, dtype=bool)
            phase_deg = np.degrees(self._leg_phases[0].detach().cpu().numpy())
            target_local = None

        planner_targets[:, 0] += self._manual_haa_sim_offsets_rad
        fk_points_local = self._generator.forward_kinematics(planner_targets, self._leg_side_signs)
        if target_local is None:
            target_local = fk_points_local[..., -1, :]

        sim_targets = planner_targets * self._leg_joint_signs
        return {
            "phase_deg": phase_deg,
            "target_local_m": target_local.detach().cpu().numpy(),
            "planner_q_deg": np.degrees(planner_targets.detach().cpu().numpy()),
            "sim_q_deg": np.degrees(sim_targets.detach().cpu().numpy()),
            "valid_ik": valid_ik,
            "fk_local_m": fk_points_local.detach().cpu().numpy(),
        }

    def all_joint_zero_pose(self) -> dict[str, np.ndarray]:
        planner_targets = torch.zeros_like(self._standing_joint_targets)
        fk_points_local = self._generator.forward_kinematics(planner_targets, self._leg_side_signs)
        return {
            "planner_q_deg": np.degrees(planner_targets.detach().cpu().numpy()),
            "sim_q_deg": np.degrees((planner_targets * self._leg_joint_signs).detach().cpu().numpy()),
            "target_local_m": fk_points_local[..., -1, :].detach().cpu().numpy(),
            "fk_local_m": fk_points_local.detach().cpu().numpy(),
        }


def hip_world_positions_m() -> np.ndarray:
    return np.stack([LEG_CONFIGS[leg_name]["hip"] for leg_name in LEG_ORDER], axis=0)


def resolve_default_standing_pose_deg() -> tuple[float, float, float]:
    geometry = QuadrupedGeometry(
        l_coxa=L_COXA_M,
        l_femur=L_FEMUR_M,
        l_tibia=L_TIBIA_M,
        femur_zero_angle_global=math.radians(90.0),
        tibia_zero_angle_relative=math.radians(180.0),
    )
    side_signs = torch.tensor([+1.0, -1.0, +1.0, -1.0], dtype=torch.float64)
    generator = QuadrupedGaitGenerator(
        geometry=geometry,
        leg_order=LEG_ORDER,
        side_signs=side_signs,
        phase_offsets=[0.0, math.pi, math.pi, 0.0],
        knee_direction_signs=(-1.0, -1.0, 1.0, 1.0),
        device="cpu",
        dtype=torch.float64,
    )
    foot_targets = torch.tensor(
        [
            [0.020, +L_COXA_M, -0.300],
            [0.020, -L_COXA_M, -0.300],
            [0.020, +L_COXA_M, -0.300],
            [0.020, -L_COXA_M, -0.300],
        ],
        dtype=torch.float64,
    )
    standing_targets, valid_ik = generator.solve_ik(foot_targets, side_signs)
    if not bool(valid_ik.all()):
        raise RuntimeError("Failed to derive a valid Lynxc standing pose from default foot targets.")
    standing_deg = torch.rad2deg(standing_targets[0]).tolist()
    return float(standing_deg[0]), float(standing_deg[1]), float(standing_deg[2])


def build_action_cfg(gait_type: str = "trot") -> QuadrupedGaitActionCfg:
    standing_haa_deg, standing_hfe_deg, standing_kfe_deg = resolve_default_standing_pose_deg()

    cfg = QuadrupedGaitActionCfg()
    cfg.gait_type = gait_type
    cfg.l_coxa = L_COXA_M
    cfg.l_femur = L_FEMUR_M
    cfg.l_tibia = L_TIBIA_M
    cfg.femur_zero_angle_global_deg = 90.0
    cfg.tibia_zero_angle_relative_deg = 180.0
    cfg.default_knee_direction = "inward"
    cfg.derive_standing_pose_from_ik = True
    cfg.standing_haa_deg = standing_haa_deg
    cfg.standing_hfe_deg = standing_hfe_deg
    cfg.standing_kfe_deg = standing_kfe_deg
    cfg.body_length = 0.260
    cfg.body_width = 0.150
    cfg.step_height = 0.040
    cfg.step_length = 0.090
    cfg.step_frequency = 1.8
    cfg.step_direction = 1.0
    cfg.center_offset = 0.020
    cfg.ground_height = -0.300
    cfg.stand_when_command_zero = True
    cfg.default_forward_command = 0.0
    cfg.command_speed_to_step_length = 0.020
    cfg.command_speed_to_frequency = 0.100
    cfg.command_ang_vel_to_turn_rate = 0.250
    cfg.command_min_step_length = 0.020
    cfg.command_lin_speed_deadband = 1.0e-3
    cfg.yaw_step_length_max = 0.020
    cfg.step_height_min = 0.0
    cfg.step_height_max = 0.080
    cfg.step_length_min = 0.0
    cfg.step_length_max = 0.140
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
            "coxa": "fl0",
            "femur": "fl1",
            "tibia": "fl2",
            "phase_offset_deg": 0.0,
            "side": "left",
            "haa_sign": +1.0,
            "hfe_sign": +1.0,
            "kfe_sign": +1.0,
            "hip_xy": (0.126157, 0.075),
        },
        "FR": {
            "coxa": "fr0",
            "femur": "fr1",
            "tibia": "fr2",
            "phase_offset_deg": 180.0,
            "side": "right",
            "haa_sign": -1.0,
            "hfe_sign": -1.0,
            "kfe_sign": -1.0,
            "hip_xy": (0.126157, -0.075),
        },
        "RL": {
            "coxa": "rl0",
            "femur": "rl1",
            "tibia": "rl2",
            "phase_offset_deg": 180.0,
            "side": "left",
            "haa_sign": +1.0,
            "hfe_sign": +1.0,
            "kfe_sign": +1.0,
            "hip_xy": (-0.133843, 0.075),
        },
        "RR": {
            "coxa": "rr0",
            "femur": "rr1",
            "tibia": "rr2",
            "phase_offset_deg": 0.0,
            "side": "right",
            "haa_sign": -1.0,
            "hfe_sign": -1.0,
            "kfe_sign": -1.0,
            "hip_xy": (-0.133843, -0.075),
        },
    }
    return cfg


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
    hip_points = hip_world_positions_m() * MM_PER_M
    x_front = hip_points[:, 0].max() + 25.0
    x_rear = hip_points[:, 0].min() - 25.0
    y_left = hip_points[:, 1].max() + 18.0
    y_right = hip_points[:, 1].min() - 18.0
    z = 0.0

    body_vertices = [[
        [x_front, y_left, z],
        [x_front, y_right, z],
        [x_rear, y_right, z],
        [x_rear, y_left, z],
    ]]
    body = Poly3DCollection(body_vertices, facecolors="lightgray", edgecolors="black", linewidths=1.5, alpha=0.45)
    ax.add_collection3d(body)

    ax.scatter(hip_points[:, 0], hip_points[:, 1], hip_points[:, 2], color="black", s=25, label="joint0 origins")
    for leg_name, leg_cfg in LEG_CONFIGS.items():
        text_pos = leg_cfg["hip"] * MM_PER_M + np.array([0.0, 0.0, 22.0])
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


def _init_leg_histories() -> dict[str, dict[str, list[np.ndarray]]]:
    histories = {}
    for leg_name in LEG_ORDER:
        histories[leg_name] = {
            "target_world": [],
            "fk_world": [],
            "target_local": [],
            "fk_local": [],
        }
    return histories


def _append_history(series: list[np.ndarray], value: np.ndarray) -> np.ndarray:
    series.append(np.asarray(value, dtype=np.float64))
    if len(series) > HISTORY_LEN:
        series.pop(0)
    return np.stack(series, axis=0)


def _is_headless_backend() -> bool:
    backend = matplotlib.get_backend().lower()
    interactive_backends = {
        "gtk3agg",
        "gtk3cairo",
        "gtk4agg",
        "gtk4cairo",
        "macosx",
        "nbagg",
        "notebook",
        "qtagg",
        "qtcairo",
        "qt5agg",
        "qt5cairo",
        "tkagg",
        "tkcairo",
        "webagg",
        "wx",
        "wxagg",
        "wxcairo",
    }
    return backend not in interactive_backends


def run_animation() -> None:
    cfgs = {gait_type: build_action_cfg(gait_type) for gait_type in GAIT_TYPES}
    planners = {gait_type: LynxcGaitPlannerAdapter(cfg, COMMAND) for gait_type, cfg in cfgs.items()}
    reference_cfg = cfgs[GAIT_TYPES[0]]

    print(
        f"[ik_lynxc] backend={matplotlib.get_backend()} "
        f"DISPLAY={os.environ.get('DISPLAY')} "
        f"WAYLAND_DISPLAY={os.environ.get('WAYLAND_DISPLAY')} "
        f"gaits={','.join(GAIT_TYPES)}"
    )

    hip_world_m = hip_world_positions_m()

    if _is_headless_backend():
        frames = {}
        for gait_type, planner in planners.items():
            frame = planner.step(gait_enabled=True)
            frame["target_world_m"] = hip_world_m + frame["target_local_m"]
            frame["fk_world_m"] = hip_world_m[:, None, :] + frame["fk_local_m"]
            frames[gait_type] = frame

        fig = plt.figure(figsize=(12, 6))
        ax3d = fig.add_subplot(121, projection="3d")
        ax2d = fig.add_subplot(122)
        vis_points = [hip_world_m * MM_PER_M]
        for gait_type, frame in frames.items():
            linestyle = GAIT_LINE_STYLES[gait_type]
            marker = GAIT_MARKERS[gait_type]
            alpha = GAIT_ALPHAS[gait_type]
            for leg_idx, leg_name in enumerate(LEG_ORDER):
                color = LEG_CONFIGS[leg_name]["color"]
                leg_points_world_mm = frame["fk_world_m"][leg_idx] * MM_PER_M
                target_world_mm = frame["target_world_m"][leg_idx] * MM_PER_M
                ax3d.plot(
                    leg_points_world_mm[:, 0],
                    leg_points_world_mm[:, 1],
                    leg_points_world_mm[:, 2],
                    linestyle=linestyle,
                    marker=marker,
                    color=color,
                    alpha=alpha,
                    label=f"{gait_type} {leg_name}" if leg_idx == 0 else None,
                )
                ax3d.scatter([target_world_mm[0]], [target_world_mm[1]], [target_world_mm[2]], color=color, marker="x", alpha=alpha)
                local_target_mm = frame["target_local_m"][leg_idx] * MM_PER_M
                local_fk_mm = frame["fk_local_m"][leg_idx, -1] * MM_PER_M
                ax2d.scatter([local_target_mm[0]], [local_target_mm[2]], color=color, marker="x", alpha=alpha)
                ax2d.scatter([local_fk_mm[0]], [local_fk_mm[2]], color=color, marker=marker, alpha=alpha)
            vis_points.append(frame["target_world_m"].reshape(-1, 3) * MM_PER_M)
            vis_points.append(frame["fk_world_m"].reshape(-1, 3) * MM_PER_M)
        add_body(ax3d)
        x_lim, y_lim, _ = set_equal_axes(ax3d, np.vstack(vis_points))
        add_ground_plane(ax3d, x_lim, y_lim, reference_cfg.ground_height * MM_PER_M)
        ax3d.set_title("Lynxc gait preview: trot solid / walk dashed")
        ax3d.legend(loc="upper left", fontsize=8)
        ax2d.set_title("Local x-z preview")
        ax2d.set_aspect("equal", adjustable="box")
        ax2d.grid(True, alpha=0.3)
        fig.savefig(HEADLESS_OUTPUT_PNG, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved headless preview to {HEADLESS_OUTPUT_PNG}")
        return

    fig = plt.figure(figsize=(16, 9))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.26, wspace=0.20)
    ax3d = fig.add_subplot(121, projection="3d")
    ax2d = fig.add_subplot(122)

    leg_lines = {}
    target_dots = {}
    fk_dots = {}
    target_world_paths = {}
    fk_world_paths = {}
    local_target_dots = {}
    local_fk_dots = {}
    local_target_paths = {}
    local_fk_paths = {}
    zero_pose_lines = {}
    zero_pose_world_dots = {}
    zero_pose_local_dots = {}
    artists = []
    histories = {gait_type: _init_leg_histories() for gait_type in GAIT_TYPES}
    history_reset_needed = {"value": True}

    vis_seed = [hip_world_m * MM_PER_M]
    for _leg_name in LEG_ORDER:
        vis_seed.append((hip_world_m + np.array([[0.0, 0.0, reference_cfg.ground_height]])).reshape(-1, 3) * MM_PER_M)
    x_lim, y_lim, _z_lim = set_equal_axes(ax3d, np.vstack(vis_seed), margin_mm=120.0)

    for gait_type in GAIT_TYPES:
        linestyle = GAIT_LINE_STYLES[gait_type]
        marker = GAIT_MARKERS[gait_type]
        alpha = GAIT_ALPHAS[gait_type]
        for leg_name in LEG_ORDER:
            key = (gait_type, leg_name)
            color = LEG_CONFIGS[leg_name]["color"]
            leg_line, = ax3d.plot(
                [], [], [],
                linestyle=linestyle,
                marker=marker,
                linewidth=3,
                color=color,
                alpha=alpha,
                markersize=5,
                label=f"{gait_type} {leg_name}",
            )
            target_dot = ax3d.scatter([], [], [], color=color, s=45, marker="x", alpha=alpha)
            fk_dot = ax3d.scatter([], [], [], color=color, s=25, marker=marker, alpha=alpha)
            target_path, = ax3d.plot([], [], [], color=color, alpha=0.25 * alpha, linewidth=1.2, linestyle=linestyle)
            fk_path, = ax3d.plot([], [], [], color=color, alpha=0.75 * alpha, linewidth=1.2, linestyle=linestyle)
            local_target_path, = ax2d.plot([], [], color=color, alpha=0.35 * alpha, linestyle=linestyle, linewidth=1.4)
            local_fk_path, = ax2d.plot([], [], color=color, alpha=0.85 * alpha, linestyle=linestyle, linewidth=1.2, label=f"{gait_type} {leg_name}")
            local_target_dot, = ax2d.plot([], [], marker="x", color=color, alpha=alpha, linestyle="None")
            local_fk_dot, = ax2d.plot([], [], marker=marker, color=color, alpha=alpha, linestyle="None")

            leg_lines[key] = leg_line
            target_dots[key] = target_dot
            fk_dots[key] = fk_dot
            target_world_paths[key] = target_path
            fk_world_paths[key] = fk_path
            local_target_paths[key] = local_target_path
            local_fk_paths[key] = local_fk_path
            local_target_dots[key] = local_target_dot
            local_fk_dots[key] = local_fk_dot
            artists.extend([
                leg_line,
                target_dot,
                fk_dot,
                target_path,
                fk_path,
                local_target_path,
                local_fk_path,
                local_target_dot,
                local_fk_dot,
            ])

    for leg_name in LEG_ORDER:
        color = LEG_CONFIGS[leg_name]["color"]
        zero_pose_line, = ax3d.plot([], [], [], ":", linewidth=2, color=color, alpha=0.35, visible=False)
        zero_pose_world_dot = ax3d.scatter([], [], [], color=color, s=35, marker="s", alpha=0.35, visible=False)
        zero_pose_local_dot, = ax2d.plot([], [], marker="s", color=color, linestyle="None", alpha=0.35, visible=False)
        zero_pose_lines[leg_name] = zero_pose_line
        zero_pose_world_dots[leg_name] = zero_pose_world_dot
        zero_pose_local_dots[leg_name] = zero_pose_local_dot
        artists.extend([zero_pose_line, zero_pose_world_dot, zero_pose_local_dot])

    add_body(ax3d)
    add_ground_plane(ax3d, x_lim, y_lim, reference_cfg.ground_height * MM_PER_M)
    ax3d.set_xlabel("X (mm, forward)")
    ax3d.set_ylabel("Y (mm, left)")
    ax3d.set_zlabel("Z (mm, up)")
    ax3d.set_title("Lynxc Gait Comparison: trot solid / walk dashed")
    ax3d.legend(loc="upper left", fontsize=7, ncol=2)
    ax3d.view_init(elev=24, azim=-48)

    ax2d.set_aspect("equal", adjustable="box")
    ax2d.grid(True, alpha=0.3)
    ax2d.set_xlim(-260.0, 260.0)
    ax2d.set_ylim(-340.0, 120.0)
    ax2d.set_xlabel("Local X (mm)")
    ax2d.set_ylabel("Local Z (mm)")
    ax2d.set_title("Target vs FK Foot Path (local x-z)")
    ax2d.legend(loc="best", fontsize=7, ncol=2)

    info_text = ax2d.text(
        0.02,
        0.98,
        "",
        transform=ax2d.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    artists.append(info_text)

    slider_axes = {}
    sliders = {}
    slider_y_positions = {
        "FL": 0.18,
        "FR": 0.14,
        "RL": 0.10,
        "RR": 0.06,
    }
    for leg_name in JOINT0_SLIDER_LEGS:
        slider_axes[leg_name] = fig.add_axes([0.08, slider_y_positions[leg_name], 0.56, 0.025])
        sliders[leg_name] = Slider(
            ax=slider_axes[leg_name],
            label=f"{leg_name.lower()}0 (deg)",
            valmin=-45.0,
            valmax=45.0,
            valinit=0.0,
            valstep=1.0,
        )

    check_ax = fig.add_axes([0.74, 0.06, 0.22, 0.14])
    toggles = CheckButtons(check_ax, ["Enable gait", "Show all-joint zero pose"], [True, False])
    check_ax.set_title("Animation")

    def current_slider_values() -> dict[str, float]:
        return {leg_name: float(sliders[leg_name].val) for leg_name in JOINT0_SLIDER_LEGS}

    gait_state = {"enabled": True}
    zero_pose_state = {"visible": False}
    zero_pose_frame = planners[GAIT_TYPES[0]].all_joint_zero_pose()
    zero_pose_frame["target_world_m"] = hip_world_m + zero_pose_frame["target_local_m"]
    zero_pose_frame["fk_world_m"] = hip_world_m[:, None, :] + zero_pose_frame["fk_local_m"]

    def reset_histories() -> None:
        for gait_histories in histories.values():
            for leg_history in gait_histories.values():
                for key in leg_history:
                    leg_history[key].clear()

    def on_slider_change(_value) -> None:
        slider_values = current_slider_values()
        for planner in planners.values():
            planner.set_manual_haa_offsets_deg(slider_values)
        history_reset_needed["value"] = True
        fig.canvas.draw_idle()

    def on_toggle(_label) -> None:
        toggle_states = toggles.get_status()
        was_enabled = gait_state["enabled"]
        gait_state["enabled"] = toggle_states[0]
        zero_pose_state["visible"] = toggle_states[1]
        if gait_state["enabled"] and not was_enabled:
            for planner in planners.values():
                planner.reset_gait_cycle()
        history_reset_needed["value"] = True
        fig.canvas.draw_idle()

    for slider in sliders.values():
        slider.on_changed(on_slider_change)
    toggles.on_clicked(on_toggle)
    for planner in planners.values():
        planner.set_manual_haa_offsets_deg(current_slider_values())

    def update(frame_idx):
        if history_reset_needed["value"]:
            reset_histories()
            history_reset_needed["value"] = False

        frames = {}
        for gait_type, planner in planners.items():
            frame = planner.step(gait_enabled=gait_state["enabled"])
            frame["target_world_m"] = hip_world_m + frame["target_local_m"]
            frame["fk_world_m"] = hip_world_m[:, None, :] + frame["fk_local_m"]
            frames[gait_type] = frame

        info_lines = [
            f"frame={frame_idx:03d}",
            f"gait={'ON' if gait_state['enabled'] else 'OFF'} zero_pose={'ON' if zero_pose_state['visible'] else 'OFF'} cmd=({COMMAND[0]:+.3f},{COMMAND[1]:+.3f},{COMMAND[2]:+.3f})",
            "style: trot=solid/o walk=dashed/^",
            "joint0 sim offsets=" + ", ".join(f"{leg}={sliders[leg].val:+.0f}" for leg in JOINT0_SLIDER_LEGS),
        ]

        for leg_idx, leg_name in enumerate(LEG_ORDER):
            zero_leg_points_world_mm = zero_pose_frame["fk_world_m"][leg_idx] * MM_PER_M
            zero_target_world_mm = zero_pose_frame["target_world_m"][leg_idx] * MM_PER_M
            zero_target_local_mm = zero_pose_frame["target_local_m"][leg_idx] * MM_PER_M
            zero_pose_lines[leg_name].set_visible(zero_pose_state["visible"])
            zero_pose_world_dots[leg_name].set_visible(zero_pose_state["visible"])
            zero_pose_local_dots[leg_name].set_visible(zero_pose_state["visible"])
            zero_pose_lines[leg_name].set_data(zero_leg_points_world_mm[:, 0], zero_leg_points_world_mm[:, 1])
            zero_pose_lines[leg_name].set_3d_properties(zero_leg_points_world_mm[:, 2])
            zero_pose_world_dots[leg_name]._offsets3d = ([zero_target_world_mm[0]], [zero_target_world_mm[1]], [zero_target_world_mm[2]])
            zero_pose_local_dots[leg_name].set_data([zero_target_local_mm[0]], [zero_target_local_mm[2]])

        for gait_type, frame in frames.items():
            phase_summary = []
            knee_x = frame["fk_local_m"][:, 2, 0]
            inner_knee_ok = bool((knee_x[:2] < 0.0).all() and (knee_x[2:] > 0.0).all())
            for leg_idx, leg_name in enumerate(LEG_ORDER):
                key = (gait_type, leg_name)
                leg_points_world_mm = frame["fk_world_m"][leg_idx] * MM_PER_M
                target_world_mm = frame["target_world_m"][leg_idx] * MM_PER_M
                fk_world_mm = frame["fk_world_m"][leg_idx, -1] * MM_PER_M
                target_local_mm = frame["target_local_m"][leg_idx] * MM_PER_M
                fk_local_mm = frame["fk_local_m"][leg_idx, -1] * MM_PER_M

                leg_lines[key].set_data(leg_points_world_mm[:, 0], leg_points_world_mm[:, 1])
                leg_lines[key].set_3d_properties(leg_points_world_mm[:, 2])
                target_dots[key]._offsets3d = ([target_world_mm[0]], [target_world_mm[1]], [target_world_mm[2]])
                fk_dots[key]._offsets3d = ([fk_world_mm[0]], [fk_world_mm[1]], [fk_world_mm[2]])

                gait_histories = histories[gait_type][leg_name]
                target_world_hist = _append_history(gait_histories["target_world"], target_world_mm)
                fk_world_hist = _append_history(gait_histories["fk_world"], fk_world_mm)
                target_local_hist = _append_history(gait_histories["target_local"], target_local_mm[[0, 2]])
                fk_local_hist = _append_history(gait_histories["fk_local"], fk_local_mm[[0, 2]])

                target_world_paths[key].set_data(target_world_hist[:, 0], target_world_hist[:, 1])
                target_world_paths[key].set_3d_properties(target_world_hist[:, 2])
                fk_world_paths[key].set_data(fk_world_hist[:, 0], fk_world_hist[:, 1])
                fk_world_paths[key].set_3d_properties(fk_world_hist[:, 2])
                local_target_paths[key].set_data(target_local_hist[:, 0], target_local_hist[:, 1])
                local_fk_paths[key].set_data(fk_local_hist[:, 0], fk_local_hist[:, 1])
                local_target_dots[key].set_data([target_local_mm[0]], [target_local_mm[2]])
                local_fk_dots[key].set_data([fk_local_mm[0]], [fk_local_mm[2]])

                phase_summary.append(f"{leg_name}:{frame['phase_deg'][leg_idx]:5.0f}")
            info_lines.append(f"{gait_type:<4} inner_knee={'Y' if inner_knee_ok else 'N'} phases " + " ".join(phase_summary))
        info_text.set_text("\n".join(info_lines))
        return artists

    ani = animation.FuncAnimation(fig, update, interval=60, blit=False, cache_frame_data=False)
    fig._lynxc_animation = ani
    plt.show()


if __name__ == "__main__":
    run_animation()
