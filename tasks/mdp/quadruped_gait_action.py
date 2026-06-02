from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .quadruped_gait_generator import QuadrupedGaitGenerator, QuadrupedGeometry

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class QuadrupedGaitAction(ActionTerm):
    """
    Command-driven quadruped gait action for Mastiff.

    The gait generator uses USD/body coordinates directly:
    +X is forward, +Y is left, and all four leg target frames share those axes.
    Targets are expressed relative to each leg's HAA origin.
    """

    cfg: QuadrupedGaitActionCfg
    _asset: Articulation

    def __init__(self, cfg: QuadrupedGaitActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_joints = len(self._joint_ids)

        self._env = env
        self._dt = env.physics_dt
        self._step_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        self._lock_base_in_air = bool(cfg.lock_base_in_air)
        self._locked_root_pose = None
        self._locked_root_vel = None
        if self._lock_base_in_air:
            default_root_state = self._asset.data.default_root_state
            self._locked_root_pose = default_root_state[:, :7].clone()
            env_origins = getattr(self._env.scene, "env_origins", None)
            if isinstance(env_origins, torch.Tensor) and env_origins.shape == self._locked_root_pose[:, :3].shape:
                self._locked_root_pose[:, :3] += env_origins
            if cfg.lock_base_height is not None:
                self._locked_root_pose[:, 2] = cfg.lock_base_height
            self._locked_root_vel = torch.zeros_like(default_root_state[:, 7:])

        geometry = QuadrupedGeometry(
            l_coxa=cfg.l_coxa,
            l_femur=cfg.l_femur,
            l_tibia=cfg.l_tibia,
            femur_rest_angle_global=math.radians(cfg.femur_rest_angle_global_deg),
            tibia_rest_angle_relative=math.radians(cfg.tibia_rest_angle_relative_deg),
        )

        self.legs: list[dict] = []
        leg_names: list[str] = []
        leg_side_signs: list[float] = []
        leg_phase_offsets: list[float] = []
        leg_hip_xy: list[tuple[float, float]] = []
        leg_joint_signs: list[tuple[float, float, float]] = []

        enabled_leg_names = set(cfg.enabled_leg_names) if cfg.enabled_leg_names is not None else None
        for leg_name, leg_conf in cfg.legs_config.items():
            if enabled_leg_names is not None and leg_name not in enabled_leg_names:
                continue

            c_ids, _ = self._asset.find_joints([leg_conf["coxa"]])
            f_ids, _ = self._asset.find_joints([leg_conf["femur"]])
            t_ids, _ = self._asset.find_joints([leg_conf["tibia"]])
            if len(c_ids) == 0 or len(f_ids) == 0 or len(t_ids) == 0:
                print(f"[QuadrupedGaitAction] Warning: could not find joints for leg {leg_name}")
                continue

            side_sign = self._side_sign_from_config(leg_conf)
            hip_xy = self._hip_xy_from_config(leg_name, leg_conf, side_sign)
            phase_offset = math.radians(float(leg_conf.get("phase_offset_deg", 0.0)))
            joint_signs = (
                float(leg_conf.get("haa_sign", 1.0)),
                float(leg_conf.get("hfe_sign", 1.0)),
                float(leg_conf.get("kfe_sign", 1.0)),
            )

            self.legs.append(
                {
                    "name": leg_name,
                    "coxa_idx": c_ids[0],
                    "femur_idx": f_ids[0],
                    "tibia_idx": t_ids[0],
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
            dtype=torch.float32,
        )

        if self._leg_count > 0:
            initial_phases = torch.tensor(leg_phase_offsets, device=self.device, dtype=torch.float32)
            self._leg_phases = initial_phases.unsqueeze(0).repeat(self.num_envs, 1)
            self._initial_leg_phases = initial_phases
            self._leg_side_signs = torch.tensor(leg_side_signs, device=self.device, dtype=torch.float32)
            self._leg_hip_xy = torch.tensor(leg_hip_xy, device=self.device, dtype=torch.float32)
            self._leg_joint_signs = torch.tensor(leg_joint_signs, device=self.device, dtype=torch.float32)
            self._leg_coxa_indices = torch.tensor([leg["coxa_idx"] for leg in self.legs], device=self.device)
            self._leg_femur_indices = torch.tensor([leg["femur_idx"] for leg in self.legs], device=self.device)
            self._leg_tibia_indices = torch.tensor([leg["tibia_idx"] for leg in self.legs], device=self.device)
        else:
            self._leg_phases = torch.zeros(self.num_envs, 0, device=self.device, dtype=torch.float32)
            self._initial_leg_phases = torch.zeros(0, device=self.device, dtype=torch.float32)
            self._leg_side_signs = torch.zeros(0, device=self.device, dtype=torch.float32)
            self._leg_hip_xy = torch.zeros(0, 2, device=self.device, dtype=torch.float32)
            self._leg_joint_signs = torch.zeros(0, 3, device=self.device, dtype=torch.float32)
            self._leg_coxa_indices = torch.zeros(0, device=self.device, dtype=torch.long)
            self._leg_femur_indices = torch.zeros(0, device=self.device, dtype=torch.long)
            self._leg_tibia_indices = torch.zeros(0, device=self.device, dtype=torch.long)

        self._rl_action_dim = self._leg_count * 4
        self._raw_actions = torch.zeros(self.num_envs, self._rl_action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._asset.num_joints, device=self.device)

        residual_shape = (self.num_envs, self._leg_count)
        self._step_height_residual = torch.zeros(residual_shape, device=self.device)
        self._step_length_residual = torch.zeros(residual_shape, device=self.device)
        self._frequency_residual = torch.zeros(residual_shape, device=self.device)
        self._turn_rate_residual = torch.zeros(residual_shape, device=self.device)

        print(
            f"[QuadrupedGaitAction] Initialized with {self._leg_count} legs; "
            "frame=(+X forward, +Y left, +Z up)"
        )

    @property
    def action_dim(self) -> int:
        return getattr(self, "_rl_action_dim", 0)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        """Map RL actions to residual gait parameters."""
        if actions.ndim != 2 or actions.shape[0] != self.num_envs:
            raise RuntimeError(f"Expected actions with shape ({self.num_envs}, {self._rl_action_dim}), got {tuple(actions.shape)}.")
        if actions.shape[1] != self._rl_action_dim:
            raise RuntimeError(f"Expected action dim {self._rl_action_dim}, got {actions.shape[1]}.")

        self._raw_actions[:] = actions
        if self._leg_count == 0:
            return

        action_by_leg = torch.clamp(actions, -1.0, 1.0).reshape(self.num_envs, self._leg_count, 4)
        self._step_height_residual = action_by_leg[:, :, 0] * self.cfg.step_height_residual_scale
        self._step_length_residual = action_by_leg[:, :, 1] * self.cfg.step_length_residual_scale
        self._frequency_residual = action_by_leg[:, :, 2] * self.cfg.step_frequency_residual_scale
        self._turn_rate_residual = action_by_leg[:, :, 3] * self.cfg.turn_rate_residual_scale

    def apply_actions(self) -> None:
        """Advance the gait generator and write joint position targets."""
        self._step_counter += 1

        if self._lock_base_in_air and self._locked_root_pose is not None and self._locked_root_vel is not None:
            self._asset.write_root_pose_to_sim(self._locked_root_pose)
            self._asset.write_root_velocity_to_sim(self._locked_root_vel)

        if self._leg_count == 0:
            return

        command = self._get_command()
        cmd_lin_x = command[:, 0]
        cmd_lin_y = command[:, 1]
        cmd_ang_z = command[:, 2]

        cmd_lin_speed = torch.sqrt(cmd_lin_x**2 + cmd_lin_y**2)
        has_lin_cmd = cmd_lin_speed > self.cfg.command_lin_speed_deadband
        safe_lin_speed = torch.clamp_min(cmd_lin_speed, 1.0e-9)
        cmd_dir_x = torch.where(has_lin_cmd, cmd_lin_x / safe_lin_speed, torch.ones_like(cmd_lin_x))
        cmd_dir_y = torch.where(has_lin_cmd, cmd_lin_y / safe_lin_speed, torch.zeros_like(cmd_lin_y))

        h_min, h_max = self.cfg.step_height_min, self.cfg.step_height_max
        l_min, l_max = self.cfg.step_length_min, self.cfg.step_length_max
        f_min, f_max = self.cfg.step_frequency_min, self.cfg.step_frequency_max

        base_step_length = self.cfg.step_length + cmd_lin_speed * self.cfg.command_speed_to_step_length
        base_step_length = torch.where(
            has_lin_cmd,
            torch.clamp(base_step_length, min=self.cfg.command_min_step_length),
            torch.full_like(base_step_length, self.cfg.step_length),
        )
        base_step_length = torch.clamp(base_step_length, min=l_min, max=l_max).unsqueeze(1)

        base_frequency = torch.clamp(
            self.cfg.step_frequency + cmd_lin_speed * self.cfg.command_speed_to_frequency,
            min=f_min,
            max=f_max,
        ).unsqueeze(1)
        base_turn_rate = torch.clamp(
            cmd_ang_z * self.cfg.command_ang_vel_to_turn_rate,
            min=-1.0,
            max=1.0,
        ).unsqueeze(1)

        step_heights = torch.clamp(self.cfg.step_height + self._step_height_residual, min=h_min, max=h_max)
        step_lengths = torch.clamp(base_step_length + self._step_length_residual, min=l_min, max=l_max)
        frequencies = torch.clamp(base_frequency + self._frequency_residual, min=f_min, max=f_max)
        turn_rates = torch.clamp(base_turn_rate + self._turn_rate_residual, min=-1.0, max=1.0)

        self._leg_phases = (self._leg_phases + (2.0 * torch.pi) * frequencies * self._dt) % (2.0 * torch.pi)

        motion_direction = 0.0 if self.cfg.step_direction == 0.0 else (1.0 if self.cfg.step_direction > 0.0 else -1.0)
        lin_active = has_lin_cmd | (not self.cfg.stand_when_command_zero)
        lin_step_lengths = torch.where(lin_active.unsqueeze(1), step_lengths, torch.zeros_like(step_lengths))
        lin_step_x = motion_direction * lin_step_lengths * cmd_dir_x.unsqueeze(1)
        lin_step_y = motion_direction * lin_step_lengths * cmd_dir_y.unsqueeze(1)

        turn_step = self._compute_turn_step(turn_rates)
        step_vectors = torch.stack((lin_step_x, lin_step_y), dim=-1) + turn_step
        step_span = torch.linalg.norm(step_vectors, dim=-1)
        active_mask = (step_span > self.cfg.command_lin_speed_deadband) & (frequencies > 1.0e-6)

        joint_targets, foot_targets, valid_ik = self._generator.joint_targets_from_phase(
            self._leg_phases,
            step_vectors,
            step_heights,
            self.cfg.ground_height,
            center_offsets=self.cfg.center_offset,
            side_signs=self._leg_side_signs,
        )
        joint_targets = torch.where(active_mask.unsqueeze(-1) & valid_ik.unsqueeze(-1), joint_targets, torch.zeros_like(joint_targets))
        joint_targets = joint_targets * self._leg_joint_signs.unsqueeze(0)

        if self.cfg.clip_joint_targets:
            joint_targets = self._clip_leg_joint_targets(joint_targets)

        self._processed_actions.zero_()
        self._processed_actions.scatter_(1, self._leg_coxa_indices.unsqueeze(0).expand(self.num_envs, -1), joint_targets[:, :, 0])
        self._processed_actions.scatter_(1, self._leg_femur_indices.unsqueeze(0).expand(self.num_envs, -1), joint_targets[:, :, 1])
        self._processed_actions.scatter_(1, self._leg_tibia_indices.unsqueeze(0).expand(self.num_envs, -1), joint_targets[:, :, 2])

        self._asset.set_joint_position_target(self._processed_actions)
        self._maybe_log_debug(command, step_vectors, foot_targets, joint_targets, active_mask, valid_ik)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._step_counter[env_ids] = 0.0
        self._step_height_residual[env_ids] = 0.0
        self._step_length_residual[env_ids] = 0.0
        self._frequency_residual[env_ids] = 0.0
        self._turn_rate_residual[env_ids] = 0.0
        if self._leg_count > 0:
            self._leg_phases[env_ids] = self._initial_leg_phases

    def _get_command(self) -> torch.Tensor:
        if hasattr(self._env, "command_manager"):
            command = self._env.command_manager.get_command(self.cfg.command_name)
            if command.ndim == 2 and command.shape[1] >= 3:
                return command[:, :3]
        command = torch.zeros(self.num_envs, 3, device=self.device)
        command[:, 0] = float(self.cfg.default_forward_command)
        return command

    def _compute_turn_step(self, turn_rates: torch.Tensor) -> torch.Tensor:
        if self.cfg.yaw_step_length_max <= 0.0:
            return torch.zeros(self.num_envs, self._leg_count, 2, device=self.device)

        tangent = torch.stack((-self._leg_hip_xy[:, 1], self._leg_hip_xy[:, 0]), dim=-1)
        tangent_norm = torch.clamp_min(torch.linalg.norm(tangent, dim=-1, keepdim=True), 1.0e-9)
        tangent_unit = tangent / tangent_norm
        return self.cfg.yaw_step_length_max * turn_rates.unsqueeze(-1) * tangent_unit.unsqueeze(0)

    def _clip_leg_joint_targets(self, joint_targets: torch.Tensor) -> torch.Tensor:
        limits = self._get_joint_limits_tensor()
        if limits is None:
            return joint_targets

        lower_all, upper_all = limits
        lower = torch.stack(
            (
                lower_all[:, self._leg_coxa_indices],
                lower_all[:, self._leg_femur_indices],
                lower_all[:, self._leg_tibia_indices],
            ),
            dim=-1,
        )
        upper = torch.stack(
            (
                upper_all[:, self._leg_coxa_indices],
                upper_all[:, self._leg_femur_indices],
                upper_all[:, self._leg_tibia_indices],
            ),
            dim=-1,
        )
        return torch.clamp(joint_targets, min=lower, max=upper)

    def _get_joint_limits_tensor(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        joint_limits = getattr(self._asset.data, "soft_joint_pos_limits", None)
        if joint_limits is None:
            joint_limits = getattr(self._asset.data, "joint_pos_limits", None)
        if joint_limits is None:
            return None
        if joint_limits.ndim == 2:
            joint_limits = joint_limits.unsqueeze(0).expand(self.num_envs, -1, -1)
        if joint_limits.ndim != 3 or joint_limits.shape[-1] != 2:
            return None
        return joint_limits[..., 0], joint_limits[..., 1]

    def _maybe_log_debug(
        self,
        command: torch.Tensor,
        step_vectors: torch.Tensor,
        foot_targets: torch.Tensor,
        joint_targets: torch.Tensor,
        active_mask: torch.Tensor,
        valid_ik: torch.Tensor,
    ) -> None:
        if not self.cfg.debug_print_enabled or self._leg_count == 0:
            return
        step_idx = int(self._step_counter[0].item())
        if step_idx % max(1, int(self.cfg.debug_print_interval)) != 0:
            return

        env_idx = min(max(int(self.cfg.debug_env_index), 0), self.num_envs - 1)
        cmd = command[env_idx]
        print(
            f"[QuadrupedGaitDebug] step={step_idx} env={env_idx} "
            f"cmd=({cmd[0].item():+.3f},{cmd[1].item():+.3f},{cmd[2].item():+.3f}) "
            f"active={int(active_mask[env_idx].sum().item())}/{self._leg_count} "
            f"ik={int(valid_ik[env_idx].sum().item())}/{self._leg_count}"
        )
        for leg_idx, leg in enumerate(self.legs):
            phase_deg = math.degrees(float(self._leg_phases[env_idx, leg_idx].item()))
            step_vec = step_vectors[env_idx, leg_idx]
            target = foot_targets[env_idx, leg_idx]
            joints = joint_targets[env_idx, leg_idx]
            print(
                f"[QuadrupedGaitDebug][{leg['name']}] "
                f"phase={phase_deg:6.1f} "
                f"step=({step_vec[0].item():+.4f},{step_vec[1].item():+.4f}) "
                f"target=({target[0].item():+.4f},{target[1].item():+.4f},{target[2].item():+.4f}) "
                f"q=({joints[0].item():+.4f},{joints[1].item():+.4f},{joints[2].item():+.4f}) "
                f"active={'Y' if bool(active_mask[env_idx, leg_idx].item()) else 'N'} "
                f"ik={'Y' if bool(valid_ik[env_idx, leg_idx].item()) else 'N'}"
            )

    def _side_sign_from_config(self, leg_conf: dict) -> float:
        if "side_sign" in leg_conf:
            return 1.0 if float(leg_conf["side_sign"]) >= 0.0 else -1.0
        side = str(leg_conf.get("side", "left")).lower()
        return 1.0 if side == "left" else -1.0

    def _hip_xy_from_config(self, leg_name: str, leg_conf: dict, side_sign: float) -> tuple[float, float]:
        if "hip_xy" in leg_conf:
            hip_xy = leg_conf["hip_xy"]
            return float(hip_xy[0]), float(hip_xy[1])

        name_upper = leg_name.upper()
        is_front = name_upper.startswith("F") or "FRONT" in str(leg_conf.get("coxa", "")).upper()
        x = 0.5 * self.cfg.body_length if is_front else -0.5 * self.cfg.body_length
        y = side_sign * 0.5 * self.cfg.body_width
        return x, y


@configclass
class QuadrupedGaitActionCfg(ActionTermCfg):
    """Configuration for :class:`QuadrupedGaitAction`."""

    class_type: type[ActionTerm] = QuadrupedGaitAction

    joint_names: list[str] = [".*"]
    enabled_leg_names: list[str] | None = None

    # Kinematics, in meters/degrees.
    l_coxa: float = 0.12005
    l_femur: float = 0.260
    l_tibia: float = 0.300
    femur_rest_angle_global_deg: float = -40.0
    tibia_rest_angle_relative_deg: float = -100.0

    # Nominal body dimensions used only for yaw/turn step vectors.
    body_length: float = 0.520
    body_width: float = 0.180

    # Baseline gait parameters.
    step_height: float = 0.03
    step_length: float = 0.12
    step_frequency: float = 2.0
    step_direction: float = 1.0
    center_offset: float | None = -0.0269
    ground_height: float = -0.35
    stand_when_command_zero: bool = True
    default_forward_command: float = 0.0

    # Compatibility fields kept so existing task cfgs can switch classes with small edits.
    gait_type: str = "trot"
    swing_vel_limits: tuple[float, float] = (0.0, 0.0)
    stance_depth: float = 0.0
    swap_haa_hfe_targets: bool = False
    turn_rate: float = 0.0

    # Bounds.
    step_height_min: float = 0.0
    step_height_max: float = 0.08
    step_length_min: float = 0.0
    step_length_max: float = 0.12
    step_frequency_min: float = 0.0
    step_frequency_max: float = 3.0

    # Command -> gait mapping.
    command_name: str = "base_velocity"
    command_speed_to_step_length: float = 0.0
    command_speed_to_frequency: float = 0.0
    command_ang_vel_to_turn_rate: float = 1.0
    command_min_step_length: float = 0.0
    command_lin_speed_deadband: float = 1.0e-3
    yaw_step_length_max: float = 0.04

    # RL residual action: per leg [height, length, frequency, turn].
    step_height_residual_scale: float = 0.02
    step_length_residual_scale: float = 0.03
    step_frequency_residual_scale: float = 0.5
    turn_rate_residual_scale: float = 0.25

    # Debug/testing.
    debug_print_enabled: bool = False
    debug_print_interval: int = 100
    debug_env_index: int = 0
    clip_joint_targets: bool = True
    lock_base_in_air: bool = False
    lock_base_height: float | None = None

    legs_config: dict = {
        "FL": {
            "coxa": "HAA_FRONT_LEFT",
            "femur": "HFE_FRONT_LEFT",
            "tibia": "KFE_FRONT_LEFT",
            "side": "left",
            "phase_offset_deg": 0.0,
            "hip_xy": (0.260, 0.090),
        },
        "FR": {
            "coxa": "HAA_FRONT_RIGHT",
            "femur": "HFE_FRONT_RIGHT",
            "tibia": "KFE_FRONT_RIGHT",
            "side": "right",
            "phase_offset_deg": 180.0,
            "hip_xy": (0.260, -0.090),
        },
        "RL": {
            "coxa": "HAA_REAR_LEFT",
            "femur": "HFE_REAR_LEFT",
            "tibia": "KFE_REAR_LEFT",
            "side": "left",
            "phase_offset_deg": 180.0,
            "hip_xy": (-0.260, 0.090),
        },
        "RR": {
            "coxa": "HAA_REAR_RIGHT",
            "femur": "HFE_REAR_RIGHT",
            "tibia": "KFE_REAR_RIGHT",
            "side": "right",
            "phase_offset_deg": 0.0,
            "hip_xy": (-0.260, -0.090),
        },
    }
