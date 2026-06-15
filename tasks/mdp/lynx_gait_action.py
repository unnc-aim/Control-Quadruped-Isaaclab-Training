from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .lynx_gait_generator import LynxGaitGenerator, LynxGeometry

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class LynxGaitAction(ActionTerm):
    """LynxC-only command gait action using the verified inner-knee IK convention."""

    cfg: LynxGaitActionCfg
    _asset: Articulation

    def __init__(self, cfg: LynxGaitActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._env = env
        self._dt = env.physics_dt
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names)
        self._step_counter = torch.zeros(self.num_envs, device=self.device)

        self.legs: list[dict] = []
        leg_names: list[str] = []
        hip_xy: list[tuple[float, float]] = []
        for leg_name, leg_cfg in cfg.legs_config.items():
            name = leg_name.upper()
            coxa_ids, _ = self._asset.find_joints([leg_cfg["coxa"]])
            femur_ids, _ = self._asset.find_joints([leg_cfg["femur"]])
            tibia_ids, _ = self._asset.find_joints([leg_cfg["tibia"]])
            if len(coxa_ids) == 0 or len(femur_ids) == 0 or len(tibia_ids) == 0:
                raise ValueError(f"Could not resolve all Lynx joints for leg {name}: {leg_cfg}")
            leg_hip_xy = tuple(float(v) for v in leg_cfg["hip_xy"])
            self.legs.append(
                {
                    "name": name,
                    "coxa_idx": coxa_ids[0],
                    "femur_idx": femur_ids[0],
                    "tibia_idx": tibia_ids[0],
                    "hip_xy": leg_hip_xy,
                }
            )
            leg_names.append(name)
            hip_xy.append(leg_hip_xy)

        self._leg_count = len(self.legs)
        if self._leg_count != 4:
            raise ValueError(f"LynxGaitAction requires all four legs, resolved {self._leg_count}.")

        self._generator = LynxGaitGenerator(
            geometry=LynxGeometry(cfg.l_coxa, cfg.l_femur, cfg.l_tibia),
            leg_order=tuple(leg_names),
            gait_type=cfg.gait_type,
            device=self.device,
            dtype=torch.float32,
        )
        self._leg_phases = self._generator.phase_offsets.unsqueeze(0).repeat(self.num_envs, 1)
        self._initial_leg_phases = self._generator.phase_offsets.clone()
        self._leg_hip_xy = torch.tensor(hip_xy, device=self.device, dtype=torch.float32)
        self._leg_coxa_indices = torch.tensor([leg["coxa_idx"] for leg in self.legs], device=self.device)
        self._leg_femur_indices = torch.tensor([leg["femur_idx"] for leg in self.legs], device=self.device)
        self._leg_tibia_indices = torch.tensor([leg["tibia_idx"] for leg in self.legs], device=self.device)

        self._standing_foot_targets = self._generator.standing_foot_targets(cfg.center_x, cfg.ground_z)
        self._standing_planner_targets, valid = self._generator.solve_ik(self._standing_foot_targets)
        if not bool(valid.all()):
            invalid = [self.legs[idx]["name"] for idx in range(self._leg_count) if not bool(valid[idx])]
            raise ValueError(f"Invalid Lynx standing IK for legs {invalid}; check center_x, ground_z, and geometry.")
        self._standing_joint_targets = self._generator.planner_to_joint(self._standing_planner_targets)

        self._lock_base_in_air = bool(cfg.lock_base_in_air)
        self._locked_root_pose = None
        self._locked_root_vel = None
        if self._lock_base_in_air:
            default_root_state = self._asset.data.default_root_state
            self._locked_root_pose = default_root_state[:, :7].clone()
            env_origins = getattr(env.scene, "env_origins", None)
            if isinstance(env_origins, torch.Tensor) and env_origins.shape == self._locked_root_pose[:, :3].shape:
                self._locked_root_pose[:, :3] += env_origins
            if cfg.lock_base_height is not None:
                self._locked_root_pose[:, 2] = float(cfg.lock_base_height)
            self._locked_root_vel = torch.zeros_like(default_root_state[:, 7:])

        self._startup_blend_duration = max(float(cfg.startup_standing_blend_duration_s), 0.0)
        self._startup_blend_elapsed = torch.full(
            (self.num_envs,), self._startup_blend_duration, device=self.device, dtype=torch.float32
        )
        self._startup_blend_start = torch.zeros(
            self.num_envs, self._leg_count, 3, device=self.device, dtype=torch.float32
        )

        self._rl_action_dim = self._leg_count * 4
        self._raw_actions = torch.zeros(self.num_envs, self._rl_action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._asset.num_joints, device=self.device)
        residual_shape = (self.num_envs, self._leg_count)
        self._height_residual = torch.zeros(residual_shape, device=self.device)
        self._length_residual = torch.zeros(residual_shape, device=self.device)
        self._frequency_residual = torch.zeros(residual_shape, device=self.device)
        self._turn_residual = torch.zeros(residual_shape, device=self.device)

        print(
            f"[LynxGaitAction] Initialized gait={self._generator.gait_type}; "
            "IK=inner-knee, zeros=(HFE=+90deg,KFE=180deg), directions=(left +++, right ---)"
        )

    @property
    def action_dim(self) -> int:
        return self._rl_action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        expected = (self.num_envs, self._rl_action_dim)
        if tuple(actions.shape) != expected:
            raise RuntimeError(f"Expected Lynx gait actions with shape {expected}, got {tuple(actions.shape)}.")
        self._raw_actions[:] = actions
        by_leg = torch.clamp(actions, -1.0, 1.0).reshape(self.num_envs, self._leg_count, 4)
        self._height_residual = by_leg[..., 0] * self.cfg.step_height_residual_scale
        self._length_residual = by_leg[..., 1] * self.cfg.step_length_residual_scale
        self._frequency_residual = by_leg[..., 2] * self.cfg.step_frequency_residual_scale
        self._turn_residual = by_leg[..., 3] * self.cfg.turn_rate_residual_scale

    def apply_actions(self) -> None:
        self._step_counter += 1
        if self._lock_base_in_air and self._locked_root_pose is not None:
            self._asset.write_root_pose_to_sim(self._locked_root_pose)
            self._asset.write_root_velocity_to_sim(self._locked_root_vel)

        command = self._get_command()
        cmd_x, cmd_y, cmd_yaw = command.unbind(dim=-1)
        linear_speed = torch.sqrt(cmd_x**2 + cmd_y**2)
        has_linear = linear_speed > self.cfg.command_lin_speed_deadband
        safe_speed = torch.clamp_min(linear_speed, 1.0e-9)
        direction_x = torch.where(has_linear, cmd_x / safe_speed, torch.ones_like(cmd_x))
        direction_y = torch.where(has_linear, cmd_y / safe_speed, torch.zeros_like(cmd_y))

        base_length = self.cfg.step_length + linear_speed * self.cfg.command_speed_to_step_length
        base_length = torch.where(
            has_linear,
            torch.clamp(base_length, min=self.cfg.command_min_step_length),
            torch.full_like(base_length, self.cfg.step_length),
        ).unsqueeze(1)
        base_frequency = (
            self.cfg.step_frequency + linear_speed * self.cfg.command_speed_to_frequency
        ).unsqueeze(1)
        base_turn = torch.clamp(cmd_yaw * self.cfg.command_ang_vel_to_turn_rate, -1.0, 1.0).unsqueeze(1)

        heights = torch.clamp(
            self.cfg.step_height + self._height_residual,
            self.cfg.step_height_min,
            self.cfg.step_height_max,
        )
        lengths = torch.clamp(
            base_length + self._length_residual,
            self.cfg.step_length_min,
            self.cfg.step_length_max,
        )
        frequencies = torch.clamp(
            base_frequency + self._frequency_residual,
            self.cfg.step_frequency_min,
            self.cfg.step_frequency_max,
        )
        turn_rates = torch.clamp(base_turn + self._turn_residual, -1.0, 1.0)
        self._leg_phases = torch.remainder(
            self._leg_phases + 2.0 * math.pi * frequencies * self._dt,
            2.0 * math.pi,
        )

        active_linear = has_linear | (not self.cfg.stand_when_command_zero)
        active_lengths = torch.where(active_linear.unsqueeze(1), lengths, torch.zeros_like(lengths))
        motion_direction = float(self.cfg.step_direction)
        linear_step = torch.stack(
            (
                motion_direction * active_lengths * direction_x.unsqueeze(1),
                motion_direction * active_lengths * direction_y.unsqueeze(1),
            ),
            dim=-1,
        )
        step_vectors = linear_step + self._compute_turn_step(turn_rates)
        active = (torch.linalg.norm(step_vectors, dim=-1) > self.cfg.command_lin_speed_deadband) & (
            frequencies > 1.0e-6
        )

        planner_targets, foot_targets, valid = self._generator.joint_targets_from_phase(
            self._leg_phases,
            step_vectors,
            heights,
            self._standing_foot_targets.unsqueeze(0),
        )
        standing = self._standing_planner_targets.unsqueeze(0).expand(self.num_envs, -1, -1)
        planner_targets = torch.where(active.unsqueeze(-1) & valid.unsqueeze(-1), planner_targets, standing)
        joint_targets = self._generator.planner_to_joint(planner_targets)
        joint_targets = self._apply_startup_blend(joint_targets)
        if self.cfg.clip_joint_targets:
            joint_targets = self._clip_joint_targets(joint_targets)

        self._processed_actions.zero_()
        env_indices = torch.arange(self.num_envs, device=self.device).unsqueeze(1)
        self._processed_actions[env_indices, self._leg_coxa_indices] = joint_targets[..., 0]
        self._processed_actions[env_indices, self._leg_femur_indices] = joint_targets[..., 1]
        self._processed_actions[env_indices, self._leg_tibia_indices] = joint_targets[..., 2]
        self._asset.set_joint_position_target(self._processed_actions)
        self._maybe_log_debug(command, foot_targets, planner_targets, joint_targets, active, valid)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._step_counter[env_ids] = 0.0
        self._height_residual[env_ids] = 0.0
        self._length_residual[env_ids] = 0.0
        self._frequency_residual[env_ids] = 0.0
        self._turn_residual[env_ids] = 0.0
        self._leg_phases[env_ids] = self._initial_leg_phases

        joint_pos = getattr(self._asset.data, "joint_pos", None)
        if not isinstance(joint_pos, torch.Tensor) or joint_pos.shape[0] != self.num_envs:
            joint_pos = self._asset.data.default_joint_pos
        self._startup_blend_start[env_ids, :, 0] = joint_pos[env_ids][:, self._leg_coxa_indices]
        self._startup_blend_start[env_ids, :, 1] = joint_pos[env_ids][:, self._leg_femur_indices]
        self._startup_blend_start[env_ids, :, 2] = joint_pos[env_ids][:, self._leg_tibia_indices]
        self._startup_blend_elapsed[env_ids] = 0.0 if self._startup_blend_duration > 0.0 else self._startup_blend_duration

    def _apply_startup_blend(self, joint_targets: torch.Tensor) -> torch.Tensor:
        if self._startup_blend_duration <= 0.0:
            return joint_targets
        blending = self._startup_blend_elapsed < self._startup_blend_duration
        if not bool(blending.any()):
            return joint_targets
        next_elapsed = torch.clamp(self._startup_blend_elapsed + self._dt, max=self._startup_blend_duration)
        alpha = next_elapsed / self._startup_blend_duration
        standing = self._standing_joint_targets.unsqueeze(0).expand(self.num_envs, -1, -1)
        blended = torch.lerp(self._startup_blend_start, standing, alpha.view(-1, 1, 1))
        self._startup_blend_elapsed = torch.where(blending, next_elapsed, self._startup_blend_elapsed)
        return torch.where(blending.view(-1, 1, 1), blended, joint_targets)

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
        tangent /= torch.clamp_min(torch.linalg.norm(tangent, dim=-1, keepdim=True), 1.0e-9)
        return self.cfg.yaw_step_length_max * turn_rates.unsqueeze(-1) * tangent.unsqueeze(0)

    def _clip_joint_targets(self, targets: torch.Tensor) -> torch.Tensor:
        limits = getattr(self._asset.data, "soft_joint_pos_limits", None)
        if limits is None:
            limits = getattr(self._asset.data, "joint_pos_limits", None)
        if limits is None:
            return targets
        if limits.ndim == 2:
            limits = limits.unsqueeze(0).expand(self.num_envs, -1, -1)
        lower = torch.stack(
            (limits[:, self._leg_coxa_indices, 0], limits[:, self._leg_femur_indices, 0], limits[:, self._leg_tibia_indices, 0]),
            dim=-1,
        )
        upper = torch.stack(
            (limits[:, self._leg_coxa_indices, 1], limits[:, self._leg_femur_indices, 1], limits[:, self._leg_tibia_indices, 1]),
            dim=-1,
        )
        return torch.clamp(targets, min=lower, max=upper)

    def _maybe_log_debug(
        self,
        command: torch.Tensor,
        foot_targets: torch.Tensor,
        planner_targets: torch.Tensor,
        joint_targets: torch.Tensor,
        active: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        if not self.cfg.debug_print_enabled:
            return
        step = int(self._step_counter[0].item())
        if step % max(int(self.cfg.debug_print_interval), 1) != 0:
            return
        env_idx = min(max(int(self.cfg.debug_env_index), 0), self.num_envs - 1)
        print(
            f"[LynxGaitDebug] step={step} cmd={command[env_idx].tolist()} "
            f"active={int(active[env_idx].sum())}/4 ik={int(valid[env_idx].sum())}/4"
        )
        for idx, leg in enumerate(self.legs):
            print(
                f"[LynxGaitDebug][{leg['name']}] target={foot_targets[env_idx, idx].tolist()} "
                f"planner_q={planner_targets[env_idx, idx].tolist()} joint_q={joint_targets[env_idx, idx].tolist()}"
            )


@configclass
class LynxGaitActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = LynxGaitAction

    joint_names: list[str] = [".*"]
    l_coxa: float = 0.075
    l_femur: float = math.sqrt(0.0602**2 + 0.22**2)
    l_tibia: float = math.sqrt(0.303431**2 + 0.0455**2 + 0.03**2)

    gait_type: str = "walk"
    center_x: float = 0.020
    ground_z: float = -0.300
    step_height: float = 0.030
    step_length: float = 0.050
    step_frequency: float = 2.5
    step_direction: float = 1.0
    stand_when_command_zero: bool = True

    step_height_min: float = 0.0
    step_height_max: float = 0.080
    step_length_min: float = 0.0
    step_length_max: float = 0.140
    step_frequency_min: float = 0.0
    step_frequency_max: float = 3.0

    command_name: str = "base_velocity"
    default_forward_command: float = 0.0
    command_speed_to_step_length: float = 0.020
    command_speed_to_frequency: float = 0.100
    command_ang_vel_to_turn_rate: float = 0.250
    command_min_step_length: float = 0.020
    command_lin_speed_deadband: float = 1.0e-3
    yaw_step_length_max: float = 0.020

    step_height_residual_scale: float = 0.008
    step_length_residual_scale: float = 0.012
    step_frequency_residual_scale: float = 0.2
    turn_rate_residual_scale: float = 0.1

    lock_base_in_air: bool = False
    lock_base_height: float | None = None
    startup_standing_blend_duration_s: float = 0.35
    clip_joint_targets: bool = True
    debug_print_enabled: bool = False
    debug_print_interval: int = 100
    debug_env_index: int = 0

    legs_config: dict = {
        "FL": {"coxa": "FL0", "femur": "FL1", "tibia": "FL2", "hip_xy": (0.126157, 0.075)},
        "FR": {"coxa": "FR0", "femur": "FR1", "tibia": "FR2", "hip_xy": (0.126157, -0.075)},
        "RL": {"coxa": "RL0", "femur": "RL1", "tibia": "RL2", "hip_xy": (-0.133843, 0.075)},
        "RR": {"coxa": "RR0", "femur": "RR1", "tibia": "RR2", "hip_xy": (-0.133843, -0.075)},
    }
