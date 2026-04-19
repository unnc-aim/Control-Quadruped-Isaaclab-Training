from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTermCfg


@configclass
class CPGConfig:
    """Kinematic and zero-pose parameters used by CPG IK."""

    # Link lengths (meters)
    l_coxa: float = 0.12005
    l_femur: float = 0.260
    l_tibia: float = 0.300

    # Zero-pose absolute angles (deg), 
    femur_rest_angle_global_deg: float = -40
    # Keep this aligned with tasks/mdp/ik_test.py.
    tibia_rest_angle_relative_deg: float = -100

    # Optional legacy vector form (femur_xy=(y, x), tibia_xy=(y, x)); when provided, it overrides deg fields.
    femur_xy: tuple[float, float] | None = None
    tibia_xy: tuple[float, float] | None = None


class CPGPositionAction(ActionTerm):
    """Dummy joint action term that makes joints follow a specific hexapod leg trajectory.
    
    This action term implements a CPG-like trajectory generator and Inverse Kinematics (IK)
    for a hexapod leg, based on the logic from ik_test.py.
    """

    cfg: CPGPositionActionCfg
    """The configuration of the action term."""
    
    _asset: Articulation
    """The articulation asset on which the action term is applied."""

    def __init__(self, cfg: CPGPositionActionCfg, env: ManagerBasedEnv) -> None:
        # Initialize the action term
        super().__init__(cfg, env)
        
        # Resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_joints = len(self._joint_ids)
        
        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)
        
        # Create tensor for processed joint targets.
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._env = env
        
        # Parameter scaling ranges (for RL action mapping)
        # RL action [-1, 1] -> [0, 1] -> [min, max] for each parameter
        self._step_height_range = (cfg.step_height_min, cfg.step_height_max)
        self._step_length_range = (cfg.step_length_min, cfg.step_length_max)
        self._frequency_range = (cfg.step_frequency_min, cfg.step_frequency_max)
        
        # Step counter to track simulation steps
        self._step_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        
        # Get simulation timestep (physics_dt, NOT step_dt)
        # physics_dt = sim.dt = 0.01s (100Hz physics)
        # step_dt = decimation * physics_dt = 20 * 0.01 = 0.2s (5Hz RL update)
        # apply_actions() is called every physics step, so we use physics_dt for phase update
        self._dt = env.physics_dt

        # Debug logging controls.
        self._debug_enabled = bool(cfg.debug_print_enabled)
        self._debug_interval = max(1, int(cfg.debug_print_interval))
        self._debug_env_index = int(cfg.debug_env_index)
        self._joint_pos_min = torch.full(
            (self.num_envs, self._asset.num_joints), float("inf"), device=self.device, dtype=torch.float32
        )
        self._joint_pos_max = torch.full(
            (self.num_envs, self._asset.num_joints), float("-inf"), device=self.device, dtype=torch.float32
        )
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

        # --- CPG IK geometry and zero-pose parameters ---
        cpg_cfg = cfg.cpg_config
        self.L_COXA = cpg_cfg.l_coxa
        self.L_FEMUR = cpg_cfg.l_femur
        self.L_TIBIA = cpg_cfg.l_tibia
        if cpg_cfg.femur_xy is None:
            self.FEMUR_REST_ANGLE_GLOBAL = math.radians(cpg_cfg.femur_rest_angle_global_deg)
        else:
            self.FEMUR_REST_ANGLE_GLOBAL = math.atan2(cpg_cfg.femur_xy[0], cpg_cfg.femur_xy[1])
        if cpg_cfg.tibia_xy is None:
            self.TIBIA_REST_ANGLE_RELATIVE = math.radians(cpg_cfg.tibia_rest_angle_relative_deg)
        else:
            self.TIBIA_REST_ANGLE_RELATIVE = (
                math.atan2(cpg_cfg.tibia_xy[0], cpg_cfg.tibia_xy[1]) - self.FEMUR_REST_ANGLE_GLOBAL
            )

        # --- Trajectory Parameters ---
        self.step_length = cfg.step_length
        self.step_height = cfg.step_height
        self.ground_height = cfg.ground_height
        self.step_frequency = cfg.step_frequency
        self.step_direction = cfg.step_direction
        self.turn_rate = cfg.turn_rate  # Differential turning rate
        self.center_offset = cfg.center_offset

        # --- Parse Leg Configurations ---
        self._enabled_leg_names = None
        if cfg.enabled_leg_names is not None:
            self._enabled_leg_names = set(cfg.enabled_leg_names)
        self.legs = []
        leg_omegas: list[float] = []
        leg_initial_phases: list[float] = []
        for leg_name, leg_conf in cfg.legs_config.items():
            # Find joint indices
            # We assume the config provides exact joint names or patterns that match 1 joint
            c_ids, _ = self._asset.find_joints([leg_conf["coxa"]])
            f_ids, _ = self._asset.find_joints([leg_conf["femur"]])
            t_ids, _ = self._asset.find_joints([leg_conf["tibia"]])
            
            if len(c_ids) > 0 and len(f_ids) > 0 and len(t_ids) > 0:
                phase_offset_deg = leg_conf.get("phase_offset_deg", 0.0)
                frequency = leg_conf.get("frequency", self.step_frequency)
                step_length = leg_conf.get("step_length", self.step_length)
                step_height = leg_conf.get("step_height", self.step_height)
                ground_height = leg_conf.get("ground_height", self.ground_height)
                center_offset = leg_conf.get("center_offset", self.center_offset)
                self.legs.append({
                    "name": leg_name,
                    "coxa_idx": c_ids[0],
                    "femur_idx": f_ids[0],
                    "tibia_idx": t_ids[0],
                    "angle_rad": math.radians(leg_conf["body_angle"]),
                    "phase_offset": math.radians(phase_offset_deg),
                    "frequency": frequency,
                    "step_length": step_length,
                    "step_height": step_height,
                    "ground_height": ground_height,
                    "center_offset": center_offset,
                    "direction_multiplier": leg_conf.get("direction_multiplier", 1.0),
                    "side": leg_conf.get("side", "left"),  # "left" or "right" for differential turning
                })
                leg_omegas.append(2.0 * math.pi * frequency)
                leg_initial_phases.append(math.radians(phase_offset_deg))
            else:
                print(f"[DummyJointPositionAction] Warning: Could not find joints for leg {leg_name}")
        self._leg_count = len(self.legs)
        if self._leg_count > 0:
            self._leg_omegas = torch.tensor(leg_omegas, device=self.device, dtype=torch.float32)
            initial_phases_tensor = torch.tensor(leg_initial_phases, device=self.device, dtype=torch.float32)
            self._leg_phases = initial_phases_tensor.unsqueeze(0).repeat(self.num_envs, 1)
            self._initial_leg_phases = initial_phases_tensor
            
            # Pre-compute leg parameters as tensors to avoid Python dict access in hot loop
            # Shape: (num_legs,) for per-leg constants
            self._leg_angle_rads = torch.tensor(
                [leg["angle_rad"] for leg in self.legs], device=self.device, dtype=torch.float32
            )
            self._leg_center_offsets = torch.tensor(
                [leg["center_offset"] for leg in self.legs], device=self.device, dtype=torch.float32
            )
            self._leg_ground_heights = torch.tensor(
                [leg["ground_height"] for leg in self.legs], device=self.device, dtype=torch.float32
            )
            self._leg_direction_multipliers = torch.tensor(
                [leg.get("direction_multiplier", 1.0) for leg in self.legs], device=self.device, dtype=torch.float32
            )
            # Side: 1.0 for left, -1.0 for right (for turn rate calculation)
            self._leg_side_signs = torch.tensor(
                [1.0 if leg.get("side", "left") == "left" else -1.0 for leg in self.legs], 
                device=self.device, dtype=torch.float32
            )
            # Keep HAA near neutral by centering target y around the coxa lateral offset.
            self._leg_lateral_offsets = self.L_COXA * self._leg_side_signs
            # Joint indices as tensors for vectorized indexing
            self._leg_coxa_indices = torch.tensor(
                [leg["coxa_idx"] for leg in self.legs], device=self.device, dtype=torch.long
            )
            self._leg_femur_indices = torch.tensor(
                [leg["femur_idx"] for leg in self.legs], device=self.device, dtype=torch.long
            )
            self._leg_tibia_indices = torch.tensor(
                [leg["tibia_idx"] for leg in self.legs], device=self.device, dtype=torch.long
            )
        else:
            self._leg_omegas = torch.zeros(0, device=self.device, dtype=torch.float32)
            self._leg_phases = torch.zeros(self.num_envs, 0, device=self.device, dtype=torch.float32)
            self._initial_leg_phases = torch.zeros(0, device=self.device, dtype=torch.float32)

        # RL action is per-leg residual over command-driven CPG parameters:
        # [d_step_height, d_step_length, d_frequency, d_turn_rate] for each leg.
        self._rl_action_dim = self._leg_count * 4
        self._raw_actions = torch.zeros(self.num_envs, self._rl_action_dim, device=self.device)
        leg_param_shape = (self.num_envs, self._leg_count)
        # Final CPG parameters applied each step (command baseline + RL residual).
        self._rl_step_height = torch.full(leg_param_shape, cfg.step_height, device=self.device)
        self._rl_step_length = torch.full(leg_param_shape, cfg.step_length, device=self.device)
        self._rl_frequency = torch.full(leg_param_shape, cfg.step_frequency, device=self.device)
        self._rl_turn_rate = torch.zeros(leg_param_shape, device=self.device)
        # RL residuals are set in process_actions().
        self._rl_step_height_residual = torch.zeros(leg_param_shape, device=self.device)
        self._rl_step_length_residual = torch.zeros(leg_param_shape, device=self.device)
        self._rl_frequency_residual = torch.zeros(leg_param_shape, device=self.device)
        self._rl_turn_rate_residual = torch.zeros(leg_param_shape, device=self.device)

        print(f"[CPGPositionAction] Initialized with {self._leg_count} legs configured")
        print(f"[CPGPositionAction] Default Freq={self.step_frequency}, Height={self.step_height}, Length={self.step_length}")
        print(
            f"[CPGPositionAction] RL Action Dim={self._rl_action_dim} "
            f"({self._leg_count} legs x 4): [d_height, d_length, d_freq, d_turn] per leg"
        )

    @property
    def action_dim(self) -> int:
        """RL action dimension: num_legs * [d_step_height, d_step_length, d_frequency, d_turn_rate]."""
        return self._rl_action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        """
        Process RL actions as per-leg residuals on top of command-driven CPG parameters.

        Action format: flatten(num_legs, [d_step_height, d_step_length, d_frequency, d_turn_rate]) in [-1, 1].
        Zero action means "no residual", i.e. pure command->CPG mapping.
        """
        if actions.ndim != 2 or actions.shape[0] != self.num_envs:
            raise RuntimeError(
                f"Expected actions with shape ({self.num_envs}, {self._rl_action_dim}), got {tuple(actions.shape)}."
            )
        if actions.shape[1] != self._rl_action_dim:
            raise RuntimeError(
                f"Expected action dim {self._rl_action_dim} (num_legs*4), got {actions.shape[1]}."
            )

        self._raw_actions[:] = actions
        if self._leg_count == 0:
            return

        action_clamped = torch.clamp(actions, -1.0, 1.0)
        action_by_leg = action_clamped.reshape(self.num_envs, self._leg_count, 4)
        self._rl_step_height_residual = action_by_leg[:, :, 0] * self.cfg.step_height_residual_scale
        self._rl_step_length_residual = action_by_leg[:, :, 1] * self.cfg.step_length_residual_scale
        self._rl_frequency_residual = action_by_leg[:, :, 2] * self.cfg.step_frequency_residual_scale
        self._rl_turn_rate_residual = action_by_leg[:, :, 3] * self.cfg.turn_rate_residual_scale

    def _get_joint_limits_for_env(self, env_idx: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Get joint lower/upper limits for a specific environment, if available."""
        joint_limits = getattr(self._asset.data, "soft_joint_pos_limits", None)
        if joint_limits is None:
            joint_limits = getattr(self._asset.data, "joint_pos_limits", None)
        if joint_limits is None:
            return None
        if joint_limits.ndim == 3:
            joint_limits = joint_limits[env_idx]
        if joint_limits.ndim != 2 or joint_limits.shape[-1] != 2:
            return None
        return joint_limits[:, 0], joint_limits[:, 1]

    def _get_joint_limits_tensor(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Get joint lower/upper limits for all environments, if available."""
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
        phi_shifted: torch.Tensor,
        phi_contact: torch.Tensor,
        step_span: torch.Tensor,
        step_x_local: torch.Tensor,
        step_y_local: torch.Tensor,
        target_x: torch.Tensor,
        target_y: torch.Tensor,
        target_z: torch.Tensor,
        valid_ik: torch.Tensor,
        active_mask: torch.Tensor,
        use_alt_branch: torch.Tensor | None = None,
        clamped_mask: torch.Tensor | None = None,
    ) -> None:
        """Print periodic CPG debug snapshot for one environment."""
        if not self._debug_enabled or self._leg_count == 0:
            return
        step_idx = int(self._step_counter[0].item())
        if step_idx % self._debug_interval != 0:
            return

        env_idx = min(max(self._debug_env_index, 0), self.num_envs - 1)
        cmd = command[env_idx]
        yaw_rate = self._get_base_yaw_rate()[env_idx].item()
        ik_valid_count = int(valid_ik[env_idx].sum().item())
        active_count = int(active_mask[env_idx].sum().item())
        alt_branch_count = int(use_alt_branch[env_idx].sum().item()) if use_alt_branch is not None else 0
        clamped_count = int(clamped_mask[env_idx].sum().item()) if clamped_mask is not None else 0

        print(
            f"[CPGDebug] step={step_idx} env={env_idx} "
            f"cmd(vx,vy,wz)=({cmd[0].item():+.3f},{cmd[1].item():+.3f},{cmd[2].item():+.3f}) "
            f"base_wz={yaw_rate:+.3f} "
            f"cpg_mean(h,l,f,turn)=({self._rl_step_height[env_idx].mean().item():.4f},"
            f"{self._rl_step_length[env_idx].mean().item():.4f},"
            f"{self._rl_frequency[env_idx].mean().item():.4f},"
            f"{self._rl_turn_rate[env_idx].mean().item():+.4f}) "
            f"ik_valid={ik_valid_count}/{self._leg_count} active={active_count}/{self._leg_count} "
            f"alt_branch={alt_branch_count}/{self._leg_count} clamped={clamped_count}/{self._leg_count}"
        )

        joint_pos_env = self._asset.data.joint_pos[env_idx]
        target_joint_pos_env = self._processed_actions[env_idx]
        target_buf = None
        target_buf_name = None
        for attr_name in ("joint_pos_target", "joint_position_target", "joint_targets"):
            buf = getattr(self._asset.data, attr_name, None)
            if isinstance(buf, torch.Tensor):
                target_buf = buf
                target_buf_name = attr_name
                break
        target_buf_env = target_buf[env_idx] if target_buf is not None else None
        limit_data = self._get_joint_limits_for_env(env_idx)
        limit_eps = 1.0e-3

        def _joint_limit_flag(joint_idx: int) -> str:
            if limit_data is None:
                return "n/a"
            lower, upper = limit_data
            cur = joint_pos_env[joint_idx]
            lo = lower[joint_idx]
            hi = upper[joint_idx]
            if cur <= lo + limit_eps:
                return "LOW"
            if cur >= hi - limit_eps:
                return "HIGH"
            return "-"

        for leg_idx, leg in enumerate(self.legs):
            coxa_idx = int(self._leg_coxa_indices[leg_idx].item())
            femur_idx = int(self._leg_femur_indices[leg_idx].item())
            tibia_idx = int(self._leg_tibia_indices[leg_idx].item())
            phase_deg = math.degrees(phi_shifted[env_idx, leg_idx].item())
            gait_phase = "STANCE" if phase_deg < 180.0 else "SWING"
            contact_phase_deg = math.degrees(phi_contact[env_idx, leg_idx].item())
            contact_phase = "STANCE" if contact_phase_deg < 180.0 else "SWING"
            q_set_coxa = (
                target_buf_env[coxa_idx].item() if target_buf_env is not None else target_joint_pos_env[coxa_idx].item()
            )
            q_set_femur = (
                target_buf_env[femur_idx].item() if target_buf_env is not None else target_joint_pos_env[femur_idx].item()
            )
            q_set_tibia = (
                target_buf_env[tibia_idx].item() if target_buf_env is not None else target_joint_pos_env[tibia_idx].item()
            )
            q_span_coxa = (self._joint_pos_max[env_idx, coxa_idx] - self._joint_pos_min[env_idx, coxa_idx]).item()
            q_span_femur = (self._joint_pos_max[env_idx, femur_idx] - self._joint_pos_min[env_idx, femur_idx]).item()
            q_span_tibia = (self._joint_pos_max[env_idx, tibia_idx] - self._joint_pos_min[env_idx, tibia_idx]).item()
            print(
                f"[CPGDebug][{leg['name']}] "
                f"phase={phase_deg:6.1f}deg({gait_phase}) "
                f"contact_phase={contact_phase_deg:6.1f}deg({contact_phase}) "
                f"span={step_span[env_idx, leg_idx].item():.4f} "
                f"cpg=({self._rl_step_height[env_idx, leg_idx].item():.4f},"
                f"{self._rl_step_length[env_idx, leg_idx].item():.4f},"
                f"{self._rl_frequency[env_idx, leg_idx].item():.4f},"
                f"{self._rl_turn_rate[env_idx, leg_idx].item():+.4f}) "
                f"step_vec=({step_x_local[env_idx, leg_idx].item():+.4f},"
                f"{step_y_local[env_idx, leg_idx].item():+.4f}) "
                f"target=({target_x[env_idx, leg_idx].item():+.4f},"
                f"{target_y[env_idx, leg_idx].item():+.4f},"
                f"{target_z[env_idx, leg_idx].item():+.4f}) "
                f"ik={'Y' if bool(valid_ik[env_idx, leg_idx].item()) else 'N'} "
                f"active={'Y' if bool(active_mask[env_idx, leg_idx].item()) else 'N'} "
                f"branch={'ALT' if (use_alt_branch is not None and bool(use_alt_branch[env_idx, leg_idx].item())) else 'IKT'} "
                f"clamp={'Y' if (clamped_mask is not None and bool(clamped_mask[env_idx, leg_idx].item())) else 'N'} "
                f"q_cmd=({target_joint_pos_env[coxa_idx].item():+.4f},"
                f"{target_joint_pos_env[femur_idx].item():+.4f},"
                f"{target_joint_pos_env[tibia_idx].item():+.4f}) "
                f"q_set=({q_set_coxa:+.4f},"
                f"{q_set_femur:+.4f},"
                f"{q_set_tibia:+.4f}) "
                f"q_cur=({joint_pos_env[coxa_idx].item():+.4f},"
                f"{joint_pos_env[femur_idx].item():+.4f},"
                f"{joint_pos_env[tibia_idx].item():+.4f}) "
                f"q_err=({(q_set_coxa - joint_pos_env[coxa_idx].item()):+.4f},"
                f"{(q_set_femur - joint_pos_env[femur_idx].item()):+.4f},"
                f"{(q_set_tibia - joint_pos_env[tibia_idx].item()):+.4f}) "
                f"q_span=({q_span_coxa:+.4f},{q_span_femur:+.4f},{q_span_tibia:+.4f}) "
                f"limit=({_joint_limit_flag(coxa_idx)},"
                f"{_joint_limit_flag(femur_idx)},"
                f"{_joint_limit_flag(tibia_idx)})"
            )
        if target_buf_name is not None:
            print(f"[CPGDebug] target_buffer={target_buf_name}")
            joint_pairs = ", ".join(
                f"{name}={target_buf_env[j].item():+.4f}" for j, name in enumerate(self._asset.joint_names)
            )
            print(f"[CPGDebug] target_dump[{env_idx}] {joint_pairs}")
        else:
            print("[CPGDebug] target_buffer=processed_actions(fallback)")
            joint_pairs = ", ".join(
                f"{name}={target_joint_pos_env[j].item():+.4f}" for j, name in enumerate(self._asset.joint_names)
            )
            print(f"[CPGDebug] target_dump[{env_idx}] {joint_pairs}")

    def _compute_trajectory(
        self,
        phase: torch.Tensor,
        angle_rad: float,
        step_length: float,
        step_height: float,
        center_offset: float,
        ground_height: float,
        direction: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute target (x, y, z) in the leg's LOCAL coordinate frame.
        """
        # 与 ik_test 对齐：相位零点放在支撑相中点（phi=pi/2）
        phase_zero_shift = math.pi / 2.0
        # Reverse phase sampling order to preserve gait travel direction
        # after switching swing arc to the upper semi-ellipse.
        phi = (phase_zero_shift - phase) % (2 * math.pi)

        # Initialize trajectory displacement in the walking direction
        d_walk = torch.zeros_like(phi)  # displacement along walking direction
        z_loc = torch.zeros_like(phi)   # vertical displacement

        # 与 ik_test 对齐：先支撑后摆动（支撑相脚向后蹬，摆动相脚向前伸）
        d_walk = direction * ((step_length / 2.0) * torch.cos(phi))
        stance_mask = phi < math.pi
        swing_mask = ~stance_mask
        z_loc[stance_mask] = 0.0
        # Align with ik_test: swing phase lifts the foot toward larger Z.
        z_loc[swing_mask] = step_height * torch.sin(phi[swing_mask] - math.pi)
        
        # --- Transform to Leg Local Coordinate Frame ---
        # The walking displacement d_walk is in a direction rotated by angle_rad from the leg's X-axis
        dx = d_walk * math.cos(angle_rad)
        dy = d_walk * math.sin(angle_rad)
        
        # Final position in leg local frame
        X = center_offset + dx
        Y = dy
        Z = ground_height + z_loc
        
        return X, Y, Z

    def _solve_ik(
        self,
        target_x: torch.Tensor,
        target_y: torch.Tensor,
        target_z: torch.Tensor,
        side_sign: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Solve Inverse Kinematics for a batch of targets in the leg's local coordinate frame.
        This follows the current ik_test left-leg geometry:
        HAA rotates around local X, coxa offset is along local +Y.
        """
        # --- 1. Solve HAA (theta1) from YZ plane constraint ---
        signed_l_coxa = self.L_COXA * side_sign
        r_yz = torch.sqrt(target_y**2 + target_z**2)
        safe_r_yz = torch.clamp_min(r_yz, 1.0e-9)
        phi_yz = torch.atan2(target_z, target_y)
        theta1 = phi_yz + torch.acos(torch.clamp(signed_l_coxa / safe_r_yz, -1.0, 1.0))

        # --- 2. Transform to femur-tibia plane ---
        c1 = torch.cos(theta1)
        s1 = torch.sin(theta1)
        w = target_x
        h = -target_y * s1 + target_z * c1
        L_virtual = torch.sqrt(w**2 + h**2)

        # --- 3. Law of cosines for femur/tibia ---
        cos_beta = (self.L_FEMUR**2 + self.L_TIBIA**2 - L_virtual**2) / (2 * self.L_FEMUR * self.L_TIBIA)
        cos_beta = torch.clamp(cos_beta, -1.0, 1.0)
        beta = torch.acos(cos_beta)

        safe_l_virtual = torch.clamp_min(L_virtual, 1.0e-9)
        cos_alpha = (self.L_FEMUR**2 + safe_l_virtual**2 - self.L_TIBIA**2) / (2 * self.L_FEMUR * safe_l_virtual)
        cos_alpha = torch.clamp(cos_alpha, -1.0, 1.0)
        alpha = torch.acos(cos_alpha)

        gamma = torch.atan2(h, w)

        # Dog-like knee-back branch (same as ik_test)
        theta2_absolute = gamma - alpha
        theta3_relative = math.pi - beta

        # --- 5. Convert to Control Deltas ---
        d_theta1 = theta1
        d_theta2 = theta2_absolute - self.FEMUR_REST_ANGLE_GLOBAL
        d_theta3 = theta3_relative - self.TIBIA_REST_ANGLE_RELATIVE
        
        return d_theta1, d_theta2, d_theta3

    def _get_base_yaw_rate(self) -> torch.Tensor:
        """Get base yaw rate (rad/s) in whichever frame is available."""
        for attr_name in ("root_ang_vel_b", "root_ang_vel_w", "root_ang_vel"):
            ang_vel = getattr(self._asset.data, attr_name, None)
            if isinstance(ang_vel, torch.Tensor) and ang_vel.ndim == 2 and ang_vel.shape[1] >= 3:
                return ang_vel[:, 2]
        return torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

    def apply_actions(self):
        """Apply the CPG trajectory to the joints using RL-controlled parameters.
        
        This is a fully vectorized implementation to avoid Python loops in the hot path.
        """
        # Increment step counter
        self._step_counter += 1

        if self._lock_base_in_air and self._locked_root_pose is not None and self._locked_root_vel is not None:
            self._asset.write_root_pose_to_sim(self._locked_root_pose)
            self._asset.write_root_velocity_to_sim(self._locked_root_vel)
        
        if self._leg_count == 0:
            return
        
        if not hasattr(self._env, "command_manager"):
            raise RuntimeError("CPGPositionAction requires env.command_manager for command-driven gait.")
        command = self._env.command_manager.get_command(self.cfg.command_name)
        if command.ndim != 2 or command.shape[1] < 3:
            raise RuntimeError(
                f"Command '{self.cfg.command_name}' must provide at least [lin_x, lin_y, ang_z], "
                f"but got shape {tuple(command.shape)}."
            )

        cmd_lin_x = command[:, 0]
        cmd_lin_y = command[:, 1]
        cmd_ang_z = command[:, 2]
        cmd_lin_speed = torch.sqrt(cmd_lin_x**2 + cmd_lin_y**2)
        has_cmd_lin = cmd_lin_speed > self.cfg.command_lin_speed_deadband
        safe_cmd_lin_speed = torch.clamp_min(cmd_lin_speed, 1.0e-9)
        cmd_dir_x = torch.where(has_cmd_lin, cmd_lin_x / safe_cmd_lin_speed, torch.zeros_like(cmd_lin_x))
        cmd_dir_y = torch.where(has_cmd_lin, cmd_lin_y / safe_cmd_lin_speed, torch.zeros_like(cmd_lin_y))

        h_min, h_max = self._step_height_range
        l_min, l_max = self._step_length_range
        f_min, f_max = self._frequency_range

        # Temporarily disable command-speed correction on step length.
        base_step_length = torch.full_like(cmd_lin_speed, self.step_length)
        base_frequency = torch.clamp(
            self.step_frequency + cmd_lin_speed * self.cfg.command_speed_to_frequency,
            min=f_min,
            max=f_max,
        )
        base_turn_rate = torch.clamp(
            cmd_ang_z * self.cfg.command_ang_vel_to_turn_rate,
            min=-1.0,
            max=1.0,
        )

        base_step_length = base_step_length.unsqueeze(1)
        base_frequency = base_frequency.unsqueeze(1)
        base_turn_rate = base_turn_rate.unsqueeze(1)

        self._rl_step_height = torch.clamp(self.step_height + self._rl_step_height_residual, h_min, h_max)
        self._rl_step_length = torch.clamp(base_step_length + self._rl_step_length_residual, l_min, l_max)
        self._rl_frequency = torch.clamp(base_frequency + self._rl_frequency_residual, f_min, f_max)
        self._rl_turn_rate = torch.clamp(base_turn_rate + self._rl_turn_rate_residual, -1.0, 1.0)

        # Compute per-leg omega from command-driven + residual frequency.
        rl_omegas = (2.0 * torch.pi) * self._rl_frequency
        
        # Update all leg phases at once.
        self._leg_phases = (self._leg_phases + rl_omegas * self._dt) % (2.0 * torch.pi)
        
        # Reset all joint actions to 0.0
        self._processed_actions.zero_()
        
        # --- Vectorized computation for all legs at once ---
        # Shapes:
        #   self._leg_phases: (num_envs, num_legs)
        #   self._rl_step_length: (num_envs, num_legs)
        #   self._rl_step_height: (num_envs, num_legs)
        #   self._rl_turn_rate: (num_envs, num_legs)
        
        # Build per-leg step vector from command velocity:
        # - translation part follows (lin_x, lin_y)
        # - turning part is differential stride on body-x for left/right legs
        turn_stride = self._rl_turn_rate * self.cfg.yaw_step_length_max * self._leg_side_signs.unsqueeze(0)
        motion_direction = 0.0 if self.step_direction == 0.0 else (1.0 if self.step_direction > 0.0 else -1.0)
        lin_step_x_body = motion_direction * self._rl_step_length * cmd_dir_x.unsqueeze(1)
        lin_step_y_body = motion_direction * self._rl_step_length * cmd_dir_y.unsqueeze(1)
        step_x_body = lin_step_x_body + turn_stride
        step_y_body = lin_step_y_body

        # Rotate body-frame step vector into each leg local frame.
        leg_angle_cos = torch.cos(self._leg_angle_rads).unsqueeze(0)
        leg_angle_sin = torch.sin(self._leg_angle_rads).unsqueeze(0)
        step_x_local = step_x_body * leg_angle_cos - step_y_body * leg_angle_sin
        step_y_local = step_x_body * leg_angle_sin + step_y_body * leg_angle_cos
        step_span = torch.sqrt(step_x_local**2 + step_y_local**2)
        safe_step_span = torch.clamp_min(step_span, 1.0e-9)
        has_step = step_span > self.cfg.command_lin_speed_deadband
        gait_dir_x = torch.where(has_step, step_x_local / safe_step_span, leg_angle_cos.expand_as(step_span))
        gait_dir_y = torch.where(has_step, step_y_local / safe_step_span, leg_angle_sin.expand_as(step_span))
        
        # Compute trajectory for all legs at once
        # phase: (num_envs, num_legs)
        phi = self._leg_phases % (2.0 * torch.pi)
        
        # 与 ik_test 对齐：零相位在支撑相中点，且先支撑后摆动
        phase_zero_shift = torch.pi / 2.0
        phi_shifted = (phase_zero_shift - phi) % (2.0 * torch.pi)

        direction_mul = self._leg_direction_multipliers.unsqueeze(0)
        d_walk = direction_mul * (step_span / 2.0) * torch.cos(phi_shifted)
        stance_mask = phi_shifted < torch.pi
        z_loc = torch.zeros_like(phi_shifted)
        # Align with ik_test: swing phase lifts the foot (upper semi-ellipse).
        z_loc = torch.where(stance_mask, z_loc, self._rl_step_height * torch.sin(phi_shifted - torch.pi))
        
        # Determine if each leg should be moving.
        is_moving = (self._rl_frequency > 1e-6) & (self._rl_step_height > 1e-6) & has_step

        # Trajectory direction in leg local frame follows command velocity vector.
        dx = d_walk * gait_dir_x
        dy = d_walk * gait_dir_y
        
        # center_offset and ground_height: (num_legs,) -> (1, num_legs)
        center_offset_expanded = self._leg_center_offsets.unsqueeze(0)
        ground_height_expanded = self._leg_ground_heights.unsqueeze(0)
        lateral_offset_expanded = self._leg_lateral_offsets.unsqueeze(0)
        
        # Final target positions: (num_envs, num_legs)
        target_x = center_offset_expanded + dx
        target_y = lateral_offset_expanded + dy
        target_z = ground_height_expanded + z_loc
        
        # --- Vectorized IK for all legs (same geometry/branch as ik_test left leg) ---
        signed_l_coxa = self.L_COXA * self._leg_side_signs.unsqueeze(0)
        r_yz = torch.sqrt(target_y**2 + target_z**2)
        valid_coxa = r_yz >= torch.abs(signed_l_coxa)
        safe_r_yz = torch.clamp_min(r_yz, 1.0e-9)
        phi_yz = torch.atan2(target_z, target_y)
        theta1 = phi_yz + torch.acos(torch.clamp(signed_l_coxa / safe_r_yz, -1.0, 1.0))

        c1 = torch.cos(theta1)
        s1 = torch.sin(theta1)
        w = target_x
        h = -target_y * s1 + target_z * c1
        L_virtual = torch.sqrt(w**2 + h**2)

        valid_l_virtual = L_virtual > 1.0e-9
        valid_reach = (L_virtual <= (self.L_FEMUR + self.L_TIBIA)) & (
            L_virtual >= abs(self.L_FEMUR - self.L_TIBIA)
        )
        valid_ik = valid_coxa & valid_l_virtual & valid_reach

        cos_beta = (self.L_FEMUR**2 + self.L_TIBIA**2 - L_virtual**2) / (2 * self.L_FEMUR * self.L_TIBIA)
        cos_beta = torch.clamp(cos_beta, -1.0, 1.0)
        beta = torch.acos(cos_beta)

        safe_l_virtual = torch.clamp_min(L_virtual, 1.0e-9)
        cos_alpha = (self.L_FEMUR**2 + safe_l_virtual**2 - self.L_TIBIA**2) / (2 * self.L_FEMUR * safe_l_virtual)
        cos_alpha = torch.clamp(cos_alpha, -1.0, 1.0)
        alpha = torch.acos(cos_alpha)

        gamma = torch.atan2(h, w)

        # Branch A: same as ik_test.
        theta2_absolute = gamma - alpha
        theta3_relative = torch.pi - beta
        d_theta1_main = theta1
        d_theta2_main = theta2_absolute - self.FEMUR_REST_ANGLE_GLOBAL
        d_theta3_main = theta3_relative - self.TIBIA_REST_ANGLE_RELATIVE

        # Branch B: alternate knee branch used only when it better satisfies joint limits.
        theta2_absolute_alt = gamma + alpha
        theta3_relative_alt = beta - torch.pi
        d_theta1_alt = theta1
        d_theta2_alt = theta2_absolute_alt - self.FEMUR_REST_ANGLE_GLOBAL
        d_theta3_alt = theta3_relative_alt - self.TIBIA_REST_ANGLE_RELATIVE
        
        # For non-moving or IK-invalid targets, set joint angles to 0
        active_mask = is_moving & valid_ik
        d_theta1 = torch.where(active_mask, d_theta1_main, torch.zeros_like(d_theta1_main))
        d_theta2 = torch.where(active_mask, d_theta2_main, torch.zeros_like(d_theta2_main))
        d_theta3 = torch.where(active_mask, d_theta3_main, torch.zeros_like(d_theta3_main))

        # Joint-limit-aware branch fallback and final clamp.
        use_alt_branch = torch.zeros_like(active_mask, dtype=torch.bool)
        clamped_mask = torch.zeros_like(active_mask, dtype=torch.bool)
        limits = self._get_joint_limits_tensor()
        if limits is not None:
            lower_all, upper_all = limits  # (num_envs, num_joints)
            lower_coxa = lower_all[:, self._leg_coxa_indices]
            upper_coxa = upper_all[:, self._leg_coxa_indices]
            lower_femur = lower_all[:, self._leg_femur_indices]
            upper_femur = upper_all[:, self._leg_femur_indices]
            lower_tibia = lower_all[:, self._leg_tibia_indices]
            upper_tibia = upper_all[:, self._leg_tibia_indices]

            def _violation(a: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
                return torch.relu(lo - a) + torch.relu(a - hi)

            main_violation = (
                _violation(d_theta1_main, lower_coxa, upper_coxa)
                + _violation(d_theta2_main, lower_femur, upper_femur)
                + _violation(d_theta3_main, lower_tibia, upper_tibia)
            )
            alt_violation = (
                _violation(d_theta1_alt, lower_coxa, upper_coxa)
                + _violation(d_theta2_alt, lower_femur, upper_femur)
                + _violation(d_theta3_alt, lower_tibia, upper_tibia)
            )

            use_alt_branch = active_mask & (alt_violation + 1.0e-6 < main_violation)
            d_theta1 = torch.where(use_alt_branch, d_theta1_alt, d_theta1)
            d_theta2 = torch.where(use_alt_branch, d_theta2_alt, d_theta2)
            d_theta3 = torch.where(use_alt_branch, d_theta3_alt, d_theta3)

            pre_clamp1 = d_theta1
            pre_clamp2 = d_theta2
            pre_clamp3 = d_theta3
            d_theta1 = torch.where(active_mask, torch.clamp(d_theta1, min=lower_coxa, max=upper_coxa), d_theta1)
            d_theta2 = torch.where(active_mask, torch.clamp(d_theta2, min=lower_femur, max=upper_femur), d_theta2)
            d_theta3 = torch.where(active_mask, torch.clamp(d_theta3, min=lower_tibia, max=upper_tibia), d_theta3)
            clamped_mask = active_mask & (
                (torch.abs(d_theta1 - pre_clamp1) > 1.0e-6)
                | (torch.abs(d_theta2 - pre_clamp2) > 1.0e-6)
                | (torch.abs(d_theta3 - pre_clamp3) > 1.0e-6)
            )
        
        # Scatter results to processed_actions using advanced indexing
        # This avoids the Python for loop entirely
        self._processed_actions.scatter_(1, self._leg_coxa_indices.unsqueeze(0).expand(self.num_envs, -1), d_theta1)
        self._processed_actions.scatter_(1, self._leg_femur_indices.unsqueeze(0).expand(self.num_envs, -1), d_theta2)
        self._processed_actions.scatter_(1, self._leg_tibia_indices.unsqueeze(0).expand(self.num_envs, -1), d_theta3)

        # Set position targets
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)
        joint_pos_now = self._asset.data.joint_pos
        self._joint_pos_min = torch.minimum(self._joint_pos_min, joint_pos_now)
        self._joint_pos_max = torch.maximum(self._joint_pos_max, joint_pos_now)

        self._maybe_log_debug(
            command=command,
            phi_shifted=phi_shifted,
            phi_contact=phi_shifted,
            step_span=step_span,
            step_x_local=step_x_local,
            step_y_local=step_y_local,
            target_x=target_x,
            target_y=target_y,
            target_z=target_z,
            valid_ik=valid_ik,
            active_mask=active_mask,
            use_alt_branch=use_alt_branch,
            clamped_mask=clamped_mask,
        )
    
    def _compute_trajectory_batched(
        self,
        phase: torch.Tensor,
        angle_rad: float,
        step_length: torch.Tensor,  # Now per-environment: (num_envs,)
        step_height: torch.Tensor,  # Now per-environment: (num_envs,)
        center_offset: float,
        ground_height: float,
        direction: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute target (x, y, z) with per-environment step_length and step_height.
        
        This is a batched version that handles different parameters per environment.
        """
        # 与 ik_test 对齐：相位零点放在支撑相中点（phi=pi/2）
        phase_zero_shift = math.pi / 2.0
        phi = (phase_zero_shift - phase) % (2 * math.pi)

        # Initialize trajectory displacement
        d_walk = torch.zeros_like(phi)
        z_loc = torch.zeros_like(phi)
        
        # 与 ik_test 对齐：先支撑后摆动（支撑相向后蹬，摆动相向前伸）
        d_walk = direction * ((step_length / 2.0) * torch.cos(phi))
        stance_mask = phi < math.pi
        swing_mask = ~stance_mask
        z_loc[stance_mask] = 0.0
        z_loc[swing_mask] = step_height[swing_mask] * torch.sin(phi[swing_mask] - math.pi)
        
        # Transform to Leg Local Coordinate Frame
        dx = d_walk * math.cos(angle_rad)
        dy = d_walk * math.sin(angle_rad)
        
        # Final position in leg local frame
        X = center_offset + dx
        Y = dy
        Z = ground_height + z_loc
        
        return X, Y, Z

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the action term for the specified environments."""
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._step_counter[env_ids] = 0.0
        
        # Reset RL-controlled parameters to defaults
        self._rl_step_height[env_ids] = self.step_height
        self._rl_step_length[env_ids] = self.step_length
        self._rl_frequency[env_ids] = self.step_frequency
        self._rl_turn_rate[env_ids] = 0.0
        self._rl_step_height_residual[env_ids] = 0.0
        self._rl_step_length_residual[env_ids] = 0.0
        self._rl_frequency_residual[env_ids] = 0.0
        self._rl_turn_rate_residual[env_ids] = 0.0
        joint_pos_now = self._asset.data.joint_pos
        self._joint_pos_min[env_ids] = joint_pos_now[env_ids]
        self._joint_pos_max[env_ids] = joint_pos_now[env_ids]
        
        if self._leg_count > 0:
            self._leg_phases[env_ids] = self._initial_leg_phases


@configclass
class CPGPositionActionCfg(ActionTermCfg):
    """
    Configuration for the CPG position action term with RL interface.

    CPG baseline is driven by command ``command_name`` and RL only outputs residuals:
    RL Action Format: flatten(num_legs, [d_step_height, d_step_length, d_frequency, d_turn_rate]) in [-1, 1].
    Zero action means no residual and therefore pure command-driven CPG.
    """

    class_type: type[ActionTerm] = CPGPositionAction

    joint_names: list[str] = [".*"]
    
    # Optional: List of leg names to enable (None = all legs enabled)
    enabled_leg_names: list[str] | None = None
    
    # Default CPG Parameters (used as initial values and for reset)
    step_height: float = 0.03      # Default step height (m)
    step_length: float = 0.05      # Default step length (m)
    step_frequency: float = 2    # Default frequency (Hz)
    step_direction: float = 1.0    # 1.0 = Forward, -1.0 = Backward (fixed, not RL controlled)
    turn_rate: float = 0.0         # Default turn rate (kept for compatibility)
    gait_type: str = "trot"        # Compatibility field (selected in task cfg)
    swing_vel_limits: tuple[float, float] = (0.0, 0.0)  # Compatibility field
    stance_depth: float = 0.0      # Compatibility field
    swap_haa_hfe_targets: bool = False  # Compatibility field

    # CPG parameter bounds
    step_height_min: float = 0.0    # Minimum step height (m) - 0 means no lift
    step_height_max: float = 0.08   # Maximum step height (m) - 80mm
    step_length_min: float = 0.0    # Minimum step length (m) - 0 means no forward motion
    step_length_max: float = 0.12   # Maximum step length (m) - 120mm
    step_frequency_min: float = 0.0 # Minimum frequency (Hz) - 0 means stationary
    step_frequency_max: float = 3.0 # Maximum frequency (Hz) - 3 steps per second

    # Command -> CPG mapping
    command_name: str = "base_velocity"
    command_speed_to_step_length: float = 0.12
    command_speed_to_frequency: float = 0.0
    command_ang_vel_to_turn_rate: float = 1.0
    command_min_step_length: float = 0.0
    command_lin_speed_deadband: float = 1.0e-3
    yaw_step_length_max: float = 0.08

    # RL residual scales
    step_height_residual_scale: float = 0.02
    step_length_residual_scale: float = 0.03
    step_frequency_residual_scale: float = 0.5
    turn_rate_residual_scale: float = 0.25
    debug_print_enabled: bool = False
    debug_print_interval: int = 100
    debug_env_index: int = 0
    lock_base_in_air: bool = False
    lock_base_height: float | None = None
    
    # CPG IK configuration
    cpg_config: CPGConfig = CPGConfig()

    # Trajectory geometry parameters
    center_offset: float = -0.0269  # ik_test 零位足端在腿局部 X 的默认偏置（m）
    ground_height: float = -0.35    # 站立高度 350mm -> 足端相对髋关节高度（m）
    
    # Leg Configuration: Name -> {Joint Names, Body Angle, Phase Offset, Side}
    # 
    # body_angle: Rotation angle of the walking trajectory relative to the leg's radial direction.
    #   - 0° means walking direction is along the leg's radial axis (outward/inward)
    #   - 90° means walking direction is tangential (sideways)
    #   - For forward walking, this should be set so the trajectory aligns with body X-axis
    #
    # phase_offset_deg: Phase offset for tripod gait coordination
    #   - Tripod Gait: Two groups of 3 legs alternate
    #   - Group A (phase=0°): FL, MR, RL - swing together
    #   - Group B (phase=180°): FR, ML, RR - swing together
    #
    # side: Which side of the body the leg is on ("left" or "right")
    #   - Used for differential turning control
    #
    # For a hexapod with legs arranged around the body:
    #   - FL/FR at ±45° from body X-axis, so body_angle = ∓45° to align with body X
    #   - ML/MR at ±90° from body X-axis, so body_angle = ∓90° to align with body X  
    #   - RL/RR at ±135° from body X-axis, so body_angle = ∓135° to align with body X
    #
    legs_config: dict = {
        # Group A: FL, MR, RL (phase = 0°)
        "FL": {"coxa": "coxa_FL", "femur": "femur_FL", "tibia": "tibia_FL", "body_angle": -90.0, "phase_offset_deg": 0.0, "side": "left"},
        "MR": {"coxa": "coxa_MR", "femur": "femur_MR", "tibia": "tibia_MR", "body_angle": 90.0, "phase_offset_deg": 0.0, "side": "right"},
        "RL": {"coxa": "coxa_RL", "femur": "femur_RL", "tibia": "tibia_RL", "body_angle": -90.0, "phase_offset_deg": 0.0, "side": "left"},
        # Group B: FR, ML, RR (phase = 180°)
        "FR": {"coxa": "coxa_FR", "femur": "femur_FR", "tibia": "tibia_FR", "body_angle": 90.0, "phase_offset_deg": 180.0, "side": "right"},
        "ML": {"coxa": "coxa_ML", "femur": "femur_ML", "tibia": "tibia_ML", "body_angle": -90.0, "phase_offset_deg": 180.0, "side": "left"},
        "RR": {"coxa": "coxa_RR", "femur": "femur_RR", "tibia": "tibia_RR", "body_angle": 90.0, "phase_offset_deg": 180.0, "side": "right"},
    }
