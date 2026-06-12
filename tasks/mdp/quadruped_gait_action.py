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
    The generator produces planner-space joint targets; per-leg ``joint_signs``
    convert those planner targets into simulator joint directions.
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
        self._startup_standing_blend_duration_s = max(float(cfg.startup_standing_blend_duration_s), 0.0)

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

        femur_zero_angle_global_deg = self._resolve_zero_angle_deg(
            cfg.femur_zero_angle_global_deg,
            cfg.femur_rest_angle_global_deg,
            "femur_zero_angle_global_deg",
            "femur_rest_angle_global_deg",
        )
        tibia_zero_angle_relative_deg = self._resolve_zero_angle_deg(
            cfg.tibia_zero_angle_relative_deg,
            cfg.tibia_rest_angle_relative_deg,
            "tibia_zero_angle_relative_deg",
            "tibia_rest_angle_relative_deg",
        )
        geometry = QuadrupedGeometry(
            l_coxa=cfg.l_coxa,
            l_femur=cfg.l_femur,
            l_tibia=cfg.l_tibia,
            femur_zero_angle_global=math.radians(femur_zero_angle_global_deg),
            tibia_zero_angle_relative=math.radians(tibia_zero_angle_relative_deg),
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
            self._standing_joint_targets = self._resolve_standing_joint_targets()
            self._warn_if_degenerate_joint_limits()
            if self._standing_joint_targets is not None:
                self._nominal_home_positions = self._generator.nominal_standing_foot_positions(
                    self._standing_joint_targets,
                    self._leg_side_signs,
                )
                self._standing_sim_joint_targets = self._standing_joint_targets * self._leg_joint_signs
            else:
                self._nominal_home_positions = None
                self._standing_sim_joint_targets = torch.zeros(self._leg_count, 3, device=self.device, dtype=torch.float32)
            self._startup_blend_start_targets = torch.zeros(self.num_envs, self._leg_count, 3, device=self.device, dtype=torch.float32)
            self._startup_blend_elapsed = torch.full((self.num_envs,), self._startup_standing_blend_duration_s, device=self.device, dtype=torch.float32)
        else:
            self._leg_phases = torch.zeros(self.num_envs, 0, device=self.device, dtype=torch.float32)
            self._initial_leg_phases = torch.zeros(0, device=self.device, dtype=torch.float32)
            self._leg_side_signs = torch.zeros(0, device=self.device, dtype=torch.float32)
            self._leg_hip_xy = torch.zeros(0, 2, device=self.device, dtype=torch.float32)
            self._leg_joint_signs = torch.zeros(0, 3, device=self.device, dtype=torch.float32)
            self._leg_coxa_indices = torch.zeros(0, device=self.device, dtype=torch.long)
            self._leg_femur_indices = torch.zeros(0, device=self.device, dtype=torch.long)
            self._leg_tibia_indices = torch.zeros(0, device=self.device, dtype=torch.long)
            self._standing_joint_targets = None
            self._nominal_home_positions = None
            self._standing_sim_joint_targets = torch.zeros(0, 3, device=self.device, dtype=torch.float32)
            self._startup_blend_start_targets = torch.zeros(self.num_envs, 0, 3, device=self.device, dtype=torch.float32)
            self._startup_blend_elapsed = torch.full((self.num_envs,), self._startup_standing_blend_duration_s, device=self.device, dtype=torch.float32)

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
            home_positions=self._nominal_home_positions,
        )
        raw_joint_targets = joint_targets.clone()
        default_joint_targets = torch.zeros_like(joint_targets)
        if self._standing_joint_targets is not None:
            default_joint_targets = self._standing_joint_targets.unsqueeze(0).expand(self.num_envs, -1, -1)
        planner_joint_targets = torch.where(active_mask.unsqueeze(-1), joint_targets, default_joint_targets)
        planner_joint_targets = torch.where(valid_ik.unsqueeze(-1), planner_joint_targets, default_joint_targets)
        sim_joint_targets = planner_joint_targets * self._leg_joint_signs.unsqueeze(0)

        sim_joint_targets = self._apply_startup_standing_blend(sim_joint_targets)

        if self.cfg.clip_joint_targets:
            sim_joint_targets = self._clip_leg_joint_targets(sim_joint_targets)

        self._processed_actions.zero_()
        self._processed_actions.scatter_(1, self._leg_coxa_indices.unsqueeze(0).expand(self.num_envs, -1), sim_joint_targets[:, :, 0])
        self._processed_actions.scatter_(1, self._leg_femur_indices.unsqueeze(0).expand(self.num_envs, -1), sim_joint_targets[:, :, 1])
        self._processed_actions.scatter_(1, self._leg_tibia_indices.unsqueeze(0).expand(self.num_envs, -1), sim_joint_targets[:, :, 2])

        self._asset.set_joint_position_target(self._processed_actions)
        self._maybe_log_debug(
            command,
            step_vectors,
            foot_targets,
            raw_joint_targets,
            planner_joint_targets,
            sim_joint_targets,
            active_mask,
            valid_ik,
        )

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
            joint_source = getattr(self._asset.data, "joint_pos", None)
            if not isinstance(joint_source, torch.Tensor) or joint_source.shape[0] != self.num_envs:
                joint_source = self._asset.data.default_joint_pos
            self._startup_blend_start_targets[env_ids, :, 0] = joint_source[env_ids][:, self._leg_coxa_indices]
            self._startup_blend_start_targets[env_ids, :, 1] = joint_source[env_ids][:, self._leg_femur_indices]
            self._startup_blend_start_targets[env_ids, :, 2] = joint_source[env_ids][:, self._leg_tibia_indices]
            self._startup_blend_elapsed[env_ids] = 0.0 if self._startup_standing_blend_duration_s > 0.0 else self._startup_standing_blend_duration_s

    def _apply_startup_standing_blend(self, sim_joint_targets: torch.Tensor) -> torch.Tensor:
        if self._leg_count == 0 or self._startup_standing_blend_duration_s <= 0.0:
            return sim_joint_targets

        blend_mask = self._startup_blend_elapsed < self._startup_standing_blend_duration_s
        if not bool(blend_mask.any()):
            return sim_joint_targets

        next_elapsed = torch.clamp(self._startup_blend_elapsed + self._dt, max=self._startup_standing_blend_duration_s)
        blend_alpha = torch.clamp(next_elapsed / self._startup_standing_blend_duration_s, min=0.0, max=1.0)
        standing_targets = self._standing_sim_joint_targets.unsqueeze(0).expand(self.num_envs, -1, -1)
        blended_targets = torch.lerp(
            self._startup_blend_start_targets,
            standing_targets,
            blend_alpha.view(self.num_envs, 1, 1),
        )
        sim_joint_targets = torch.where(blend_mask.view(self.num_envs, 1, 1), blended_targets, sim_joint_targets)
        self._startup_blend_elapsed = torch.where(blend_mask, next_elapsed, self._startup_blend_elapsed)
        return sim_joint_targets

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

    def _resolve_zero_angle_deg(
        self,
        zero_angle_deg: float | None,
        deprecated_angle_deg: float | None,
        zero_field_name: str,
        deprecated_field_name: str,
    ) -> float:
        if zero_angle_deg is not None:
            return float(zero_angle_deg)
        if deprecated_angle_deg is not None:
            return float(deprecated_angle_deg)
        raise ValueError(
            f"QuadrupedGaitAction requires {zero_field_name} or deprecated {deprecated_field_name} to be set."
        )

    def _resolve_standing_joint_targets(self) -> torch.Tensor | None:
        standing_fields = (
            self.cfg.standing_haa_deg,
            self.cfg.standing_hfe_deg,
            self.cfg.standing_kfe_deg,
        )
        if all(value is None for value in standing_fields):
            return None
        if any(value is None for value in standing_fields):
            raise ValueError(
                "QuadrupedGaitAction standing pose requires standing_haa_deg, standing_hfe_deg, and standing_kfe_deg."
            )
        standing_joint_targets = torch.tensor(
            [
                math.radians(float(self.cfg.standing_haa_deg)),
                math.radians(float(self.cfg.standing_hfe_deg)),
                math.radians(float(self.cfg.standing_kfe_deg)),
            ],
            device=self.device,
            dtype=torch.float32,
        )
        return standing_joint_targets.unsqueeze(0).repeat(self._leg_count, 1)

    def _warn_if_degenerate_joint_limits(self) -> None:
        limits = self._get_joint_limits_tensor()
        if limits is None or self._leg_count == 0:
            return

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
        spans = upper - lower
        collapsed = spans[0] <= 1.0e-5
        if not bool(collapsed.any()):
            return

        details: list[str] = []
        for leg_idx, leg in enumerate(self.legs):
            for axis_idx, joint_idx in enumerate((leg["coxa_idx"], leg["femur_idx"], leg["tibia_idx"])):
                if bool(collapsed[leg_idx, axis_idx]):
                    joint_name = self._asset.joint_names[joint_idx]
                    low = lower[0, leg_idx, axis_idx].item()
                    high = upper[0, leg_idx, axis_idx].item()
                    details.append(f"{joint_name}=[{low:+.5f}, {high:+.5f}]")

        print(
            "[QuadrupedGaitAction] Warning: some controlled joints have near-zero position limits; "
            "targets may be clipped before they reach the simulator: "
            + ", ".join(details)
        )

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
        raw_planner_joint_targets: torch.Tensor,
        planner_joint_targets: torch.Tensor,
        sim_joint_targets: torch.Tensor,
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
        lab_joint_pos = getattr(self._asset.data, "joint_pos", None)
        joint_limits = self._get_joint_limits_tensor()
        selected_target_tensor = ", ".join(
            f"{name}={self._processed_actions[env_idx, joint_id].item():+.4f}"
            for joint_id, name in zip(self._joint_ids, self._joint_names, strict=False)
        )
        if isinstance(lab_joint_pos, torch.Tensor):
            selected_joint_pos_tensor = ", ".join(
                f"{name}={lab_joint_pos[env_idx, joint_id].item():+.4f}"
                for joint_id, name in zip(self._joint_ids, self._joint_names, strict=False)
            )
        else:
            selected_joint_pos_tensor = "unavailable"
        if joint_limits is not None:
            lower_all, upper_all = joint_limits
            selected_joint_limit_tensor = ", ".join(
                f"{name}=[{lower_all[env_idx, joint_id].item():+.4f},{upper_all[env_idx, joint_id].item():+.4f}]"
                for joint_id, name in zip(self._joint_ids, self._joint_names, strict=False)
            )
        else:
            selected_joint_limit_tensor = "unavailable"

        print(
            f"[QuadrupedGaitDebug] step={step_idx} env={env_idx} "
            f"cmd=({cmd[0].item():+.3f},{cmd[1].item():+.3f},{cmd[2].item():+.3f}) "
            f"active={int(active_mask[env_idx].sum().item())}/{self._leg_count} "
            f"ik={int(valid_ik[env_idx].sum().item())}/{self._leg_count}"
        )
        print(f"[QuadrupedGaitDebug][LabTargetTensor] {selected_target_tensor}")
        print(f"[QuadrupedGaitDebug][LabJointPosTensor] {selected_joint_pos_tensor}")
        print(f"[QuadrupedGaitDebug][JointLimitTensor] {selected_joint_limit_tensor}")
        for leg_idx, leg in enumerate(self.legs):
            phase_deg = math.degrees(float(self._leg_phases[env_idx, leg_idx].item()))
            step_vec = step_vectors[env_idx, leg_idx]
            target = foot_targets[env_idx, leg_idx]
            raw_planner_joints = raw_planner_joint_targets[env_idx, leg_idx]
            planner_joints = planner_joint_targets[env_idx, leg_idx]
            sim_joints = sim_joint_targets[env_idx, leg_idx]
            coxa_idx = int(self._leg_coxa_indices[leg_idx].item())
            femur_idx = int(self._leg_femur_indices[leg_idx].item())
            tibia_idx = int(self._leg_tibia_indices[leg_idx].item())
            lab_target_joints = self._processed_actions[env_idx, [coxa_idx, femur_idx, tibia_idx]]
            if isinstance(lab_joint_pos, torch.Tensor):
                lab_joint_joints = lab_joint_pos[env_idx, [coxa_idx, femur_idx, tibia_idx]]
                lab_joint_str = (
                    f"lab_q=(HAA={lab_joint_joints[0].item():+.4f},"
                    f"HFE={lab_joint_joints[1].item():+.4f},"
                    f"KFE={lab_joint_joints[2].item():+.4f}) "
                )
            else:
                lab_joint_str = "lab_q=(unavailable) "
            if joint_limits is not None:
                leg_lower = torch.stack((
                    lower_all[env_idx, coxa_idx],
                    lower_all[env_idx, femur_idx],
                    lower_all[env_idx, tibia_idx],
                ))
                leg_upper = torch.stack((
                    upper_all[env_idx, coxa_idx],
                    upper_all[env_idx, femur_idx],
                    upper_all[env_idx, tibia_idx],
                ))
                limit_str = (
                    f"limits=(HAA=[{leg_lower[0].item():+.4f},{leg_upper[0].item():+.4f}],"
                    f"HFE=[{leg_lower[1].item():+.4f},{leg_upper[1].item():+.4f}],"
                    f"KFE=[{leg_lower[2].item():+.4f},{leg_upper[2].item():+.4f}]) "
                )
            else:
                limit_str = "limits=(unavailable) "
            print(
                f"[QuadrupedGaitDebug][{leg['name']}] "
                f"phase={phase_deg:6.1f} "
                f"step=({step_vec[0].item():+.4f},{step_vec[1].item():+.4f}) "
                f"target=({target[0].item():+.4f},{target[1].item():+.4f},{target[2].item():+.4f}) "
                f"raw_ik_q=(HAA={raw_planner_joints[0].item():+.4f},"
                f"HFE={raw_planner_joints[1].item():+.4f},"
                f"KFE={raw_planner_joints[2].item():+.4f}) "
                f"planner_q=(HAA={planner_joints[0].item():+.4f},"
                f"HFE={planner_joints[1].item():+.4f},"
                f"KFE={planner_joints[2].item():+.4f}) "
                f"sim_q=(HAA={sim_joints[0].item():+.4f},"
                f"HFE={sim_joints[1].item():+.4f},"
                f"KFE={sim_joints[2].item():+.4f}) "
                f"target_q=(HAA={lab_target_joints[0].item():+.4f},"
                f"HFE={lab_target_joints[1].item():+.4f},"
                f"KFE={lab_target_joints[2].item():+.4f}) "
                f"{lab_joint_str}"
                f"{limit_str}"
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
    femur_zero_angle_global_deg: float | None = None
    tibia_zero_angle_relative_deg: float | None = None

    # Deprecated compatibility fallbacks for zero-angle mapping.
    femur_rest_angle_global_deg: float | None = -40.0
    tibia_rest_angle_relative_deg: float | None = -100.0

    # Optional standing pose in planner joint space, before per-leg joint_signs.
    standing_haa_deg: float | None = None
    standing_hfe_deg: float | None = None
    standing_kfe_deg: float | None = None

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
    lock_base_height: float | None = 2
    startup_standing_blend_duration_s: float = 0.0

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
