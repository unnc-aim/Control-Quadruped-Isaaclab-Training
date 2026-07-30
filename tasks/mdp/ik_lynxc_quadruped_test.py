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

    generator_name = "tasks.mdp.lynx_gait_generator"
    generator_path = MDP_DIR / "lynx_gait_generator.py"
    generator_spec = importlib.util.spec_from_file_location(generator_name, generator_path)
    if generator_spec is None or generator_spec.loader is None:
        raise RuntimeError(f"Failed to load {generator_name} from {generator_path}")
    generator_module = importlib.util.module_from_spec(generator_spec)
    sys.modules[generator_name] = generator_module
    generator_spec.loader.exec_module(generator_module)

    action_name = "tasks.mdp.lynx_gait_action"
    action_path = MDP_DIR / "lynx_gait_action.py"
    action_spec = importlib.util.spec_from_file_location(action_name, action_path)
    if action_spec is None or action_spec.loader is None:
        raise RuntimeError(f"Failed to load {action_name} from {action_path}")
    action_module = importlib.util.module_from_spec(action_spec)
    sys.modules[action_name] = action_module
    action_spec.loader.exec_module(action_module)
    return generator_module, action_module


_generator_module, _action_module = _load_mdp_modules()
LynxGaitGenerator = _generator_module.LynxGaitGenerator
LynxGeometry = _generator_module.LynxGeometry
LynxGaitAction = _action_module.LynxGaitAction
LynxGaitActionCfg = _action_module.LynxGaitActionCfg

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
    def __init__(self, cfg: LynxGaitActionCfg, command_xyz: np.ndarray):
        self.cfg = cfg
        self.device = torch.device("cpu")
        self._dt = DT
        self._command = torch.tensor(command_xyz, dtype=torch.float64).unsqueeze(0)
        self.legs = list(cfg.legs_config)
        self._generator = LynxGaitGenerator(
            geometry=LynxGeometry(cfg.l_coxa, cfg.l_femur, cfg.l_tibia),
            leg_order=tuple(self.legs),
            gait_type=cfg.gait_type,
            device=self.device,
            dtype=torch.float64,
        )
        self._leg_count = len(self.legs)
        self._leg_phases = self._generator.phase_offsets.unsqueeze(0)
        self._initial_leg_phases = self._generator.phase_offsets.clone()
        self._leg_hip_xy = torch.tensor(
            [cfg.legs_config[name]["hip_xy"] for name in self.legs], dtype=torch.float64
        )
        self._standing_foot_targets = self._generator.standing_foot_targets(cfg.center_x, cfg.ground_z)
        self._standing_planner_targets, valid = self._generator.solve_ik(self._standing_foot_targets)
        if not bool(valid.all()):
            raise RuntimeError("Failed to derive a valid Lynxc standing pose.")
        self._manual_haa_offsets = torch.zeros(self._leg_count, dtype=torch.float64)
        self.set_rl_actions(np.zeros(self._leg_count * LynxGaitAction.ACTIONS_PER_LEG))

    def set_rl_actions(self, actions: np.ndarray) -> None:
        action = torch.as_tensor(actions, dtype=torch.float64).reshape(self._leg_count, 3).clamp(-1.0, 1.0)
        self._length_residual = action[:, 0] * self.cfg.step_length_residual_scale
        self._height_residual = action[:, 1] * self.cfg.step_height_residual_scale
        self._trajectory_z = torch.clamp(
            action[:, 2] * self.cfg.trajectory_z_residual_scale,
            self.cfg.trajectory_z_min,
            self.cfg.trajectory_z_max,
        )

    def set_manual_haa_offsets_deg(self, offsets_deg_by_leg: dict[str, float]) -> None:
        offsets = torch.tensor(
            [math.radians(float(offsets_deg_by_leg.get(name, 0.0))) for name in self.legs],
            dtype=torch.float64,
        )
        self._manual_haa_offsets = offsets * self._generator.joint_direction_signs[:, 0]

    def reset_gait_cycle(self) -> None:
        self._leg_phases[0] = self._initial_leg_phases

    def step(self, gait_enabled: bool) -> dict[str, np.ndarray]:
        if gait_enabled:
            cmd_x, cmd_y, cmd_yaw = self._command[0]
            linear_speed = torch.sqrt(cmd_x**2 + cmd_y**2)
            has_linear = bool(linear_speed > self.cfg.command_lin_speed_deadband)
            if has_linear:
                direction = torch.stack((cmd_x, cmd_y)) / linear_speed
                base_length = max(
                    self.cfg.step_length + float(linear_speed) * self.cfg.command_speed_to_step_length,
                    self.cfg.command_min_step_length,
                )
            else:
                direction = torch.tensor([1.0, 0.0], dtype=torch.float64)
                base_length = self.cfg.step_length
            lengths = torch.clamp(
                base_length + self._length_residual, self.cfg.step_length_min, self.cfg.step_length_max
            )
            heights = torch.clamp(
                self.cfg.step_height + self._height_residual,
                self.cfg.step_height_min,
                self.cfg.step_height_max,
            )
            frequency = min(
                max(self.cfg.step_frequency + float(linear_speed) * self.cfg.command_speed_to_frequency,
                    self.cfg.step_frequency_min),
                self.cfg.step_frequency_max,
            )
            self._leg_phases = torch.remainder(
                self._leg_phases + 2.0 * math.pi * frequency * self._dt, 2.0 * math.pi
            )
            active_lengths = lengths if has_linear or not self.cfg.stand_when_command_zero else torch.zeros_like(lengths)
            step_vectors = self.cfg.step_direction * active_lengths.unsqueeze(-1) * direction
            tangent = torch.stack((-self._leg_hip_xy[:, 1], self._leg_hip_xy[:, 0]), dim=-1)
            tangent /= torch.clamp_min(torch.linalg.norm(tangent, dim=-1, keepdim=True), 1.0e-9)
            turn_rate = float(torch.clamp(cmd_yaw * self.cfg.command_ang_vel_to_turn_rate, -1.0, 1.0))
            step_vectors += self.cfg.yaw_step_length_max * turn_rate * tangent
            has_step_motion = torch.linalg.norm(step_vectors, dim=-1) > self.cfg.command_lin_speed_deadband
            gait_heights = torch.where(has_step_motion, heights, torch.zeros_like(heights))
            planner_targets, target_local, valid = self._generator.joint_targets_from_phase(
                self._leg_phases,
                step_vectors.unsqueeze(0),
                gait_heights.unsqueeze(0),
                self._standing_foot_targets.unsqueeze(0),
                trajectory_z_offsets=self._trajectory_z.unsqueeze(0),
            )
            planner_targets = planner_targets[0]
            target_local = target_local[0]
            valid = valid[0]
            planner_targets = torch.where(
                valid.unsqueeze(-1), planner_targets, self._standing_planner_targets
            )
            phase_deg = np.degrees(self._leg_phases[0].numpy())
        else:
            planner_targets = self._standing_planner_targets.clone()
            target_local = self._standing_foot_targets.clone()
            valid = torch.ones(self._leg_count, dtype=torch.bool)
            phase_deg = np.degrees(self._leg_phases[0].numpy())

        planner_targets[:, 0] += self._manual_haa_offsets
        fk_points_local = self._generator.forward_kinematics(planner_targets)
        joint_targets = self._generator.planner_to_joint(planner_targets)
        return {
            "phase_deg": phase_deg,
            "target_local_m": target_local.numpy(),
            "planner_q_deg": np.degrees(planner_targets.numpy()),
            "sim_q_deg": np.degrees(joint_targets.numpy()),
            "valid_ik": valid.numpy(),
            "fk_local_m": fk_points_local.numpy(),
        }

    def all_joint_zero_pose(self) -> dict[str, np.ndarray]:
        planner_targets = torch.zeros_like(self._standing_planner_targets)
        fk_points_local = self._generator.forward_kinematics(planner_targets)
        return {
            "planner_q_deg": np.degrees(planner_targets.numpy()),
            "sim_q_deg": np.degrees(self._generator.planner_to_joint(planner_targets).numpy()),
            "target_local_m": fk_points_local[..., -1, :].numpy(),
            "fk_local_m": fk_points_local.numpy(),
        }


def hip_world_positions_m() -> np.ndarray:
    return np.stack([LEG_CONFIGS[leg_name]["hip"] for leg_name in LEG_ORDER], axis=0)


def build_action_cfg(gait_type: str = "trot") -> LynxGaitActionCfg:
    cfg = LynxGaitActionCfg()
    cfg.gait_type = gait_type
    cfg.l_coxa = L_COXA_M
    cfg.l_femur = L_FEMUR_M
    cfg.l_tibia = L_TIBIA_M
    cfg.step_height = 0.040
    cfg.step_length = 0.090
    cfg.step_frequency = 1.8
    cfg.step_direction = 1.0
    cfg.center_x = 0.020
    cfg.ground_z = -0.300
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
    cfg.trajectory_z_residual_scale = 0.10
    cfg.trajectory_z_min = -0.05
    cfg.trajectory_z_max = 0.10
    cfg.debug_print_enabled = False
    cfg.clip_joint_targets = False
    cfg.lock_base_in_air = False
    cfg.legs_config = {
        "FL": {"coxa": "fl0", "femur": "fl1", "tibia": "fl2", "hip_xy": (0.126157, 0.075)},
        "FR": {"coxa": "fr0", "femur": "fr1", "tibia": "fr2", "hip_xy": (0.126157, -0.075)},
        "RL": {"coxa": "rl0", "femur": "rl1", "tibia": "rl2", "hip_xy": (-0.133843, 0.075)},
        "RR": {"coxa": "rr0", "femur": "rr1", "tibia": "rr2", "hip_xy": (-0.133843, -0.075)},
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
        f"gaits={','.join(GAIT_TYPES)} action_dim={len(LEG_ORDER) * LynxGaitAction.ACTIONS_PER_LEG}"
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
        add_ground_plane(ax3d, x_lim, y_lim, reference_cfg.ground_z * MM_PER_M)
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
        vis_seed.append((hip_world_m + np.array([[0.0, 0.0, reference_cfg.ground_z]])).reshape(-1, 3) * MM_PER_M)
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
    add_ground_plane(ax3d, x_lim, y_lim, reference_cfg.ground_z * MM_PER_M)
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
            inner_knee_ok = bool((knee_x[:2] > 0.0).all() and (knee_x[2:] < 0.0).all())
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
