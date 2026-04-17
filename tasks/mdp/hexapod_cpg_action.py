from __future__ import annotations

import math
import re
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
    """Kinematic parameters for quadruped HAA-HFE-KFE leg IK."""

    # Leg geometry (meters)
    # l_coxa is interpreted as the lateral HAA->HFE chain offset magnitude.
    l_coxa: float = 0.1205
    l_femur: float = 0.260
    l_tibia: float = 0.300

    # Joint rest angles (radians) in robot control convention.
    hip_rest_angle: float = 0.0
    femur_rest_angle = None
    tibia_rest_angle = None
    # Legacy fallback: if rest-angle fields are None, derive from these vectors.
    # femur angle: atan2(femur_xy[0], femur_xy[1])
    femur_xy: tuple[float, float] | None = (-220.1, 138.5)
    # tibia relative angle: atan2(tibia_xy[0], tibia_xy[1]) - femur_rest_angle
    tibia_xy: tuple[float, float] | None = (-348.49, 87.58)


class CPGPositionAction(ActionTerm):
    """Joint action term for command-driven quadruped gait generation.

    The locomotion logic is adapted from wb-mpc-locoman gait scheduling concepts
    (contact/swing sequencing and swing profile shaping), while keeping IsaacLab's
    vectorized action-term interface.
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
        
        # Create tensors for raw and processed actions.
        # RL action is residual over command-driven CPG parameters:
        # [d_step_height, d_step_length, d_frequency, d_turn_rate]
        self._rl_action_dim = 4
        self._raw_actions = torch.zeros(self.num_envs, self._rl_action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._env = env
        self._debug_print_enabled = cfg.debug_print_enabled
        self._debug_print_interval = max(1, int(cfg.debug_print_interval))
        self._debug_env_index = max(0, int(cfg.debug_env_index))
        
        # Final CPG parameters applied each step (command baseline + RL residual).
        self._rl_step_height = torch.full((self.num_envs,), cfg.step_height, device=self.device)
        self._rl_step_length = torch.full((self.num_envs,), cfg.step_length, device=self.device)
        self._rl_frequency = torch.full((self.num_envs,), cfg.step_frequency, device=self.device)
        self._rl_turn_rate = torch.zeros(self.num_envs, device=self.device)
        # RL residuals are set in process_actions().
        self._rl_step_height_residual = torch.zeros(self.num_envs, device=self.device)
        self._rl_step_length_residual = torch.zeros(self.num_envs, device=self.device)
        self._rl_frequency_residual = torch.zeros(self.num_envs, device=self.device)
        self._rl_turn_rate_residual = torch.zeros(self.num_envs, device=self.device)
        
        # CPG parameter bounds
        self._step_height_range = (cfg.step_height_min, cfg.step_height_max)
        self._step_length_range = (cfg.step_length_min, cfg.step_length_max)
        self._frequency_range = (cfg.step_frequency_min, cfg.step_frequency_max)
        
        # Step counter to track simulation steps
        self._step_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._gait_type = cfg.gait_type.lower()
        if self._gait_type not in {"trot", "walk", "stand"}:
            raise ValueError(f"Unsupported gait_type: {cfg.gait_type}. Expected one of: trot, walk, stand.")
        if self._gait_type == "trot":
            self._swing_ratio = 0.5
        elif self._gait_type == "walk":
            self._swing_ratio = 0.25
        else:
            self._swing_ratio = 0.0
        
        # Get simulation timestep (physics_dt, NOT step_dt)
        # physics_dt = sim.dt = 0.01s (100Hz physics)
        # step_dt = decimation * physics_dt = 20 * 0.01 = 0.2s (5Hz RL update)
        # apply_actions() is called every physics step, so we use physics_dt for phase update
        self._dt = env.physics_dt

        # --- CPG IK geometry and zero-pose parameters ---
        cpg_cfg = cfg.cpg_config
        self.L_HIP = cpg_cfg.l_coxa
        self.L_FEMUR = cpg_cfg.l_femur
        self.L_TIBIA = cpg_cfg.l_tibia
        self.HIP_REST_ANGLE = cpg_cfg.hip_rest_angle
        if cpg_cfg.femur_rest_angle is not None and cpg_cfg.tibia_rest_angle is not None:
            self.FEMUR_REST_ANGLE_GLOBAL = cpg_cfg.femur_rest_angle
            self.TIBIA_REST_ANGLE_RELATIVE = cpg_cfg.tibia_rest_angle
        elif cpg_cfg.femur_xy is not None and cpg_cfg.tibia_xy is not None:
            self.FEMUR_REST_ANGLE_GLOBAL = math.atan2(cpg_cfg.femur_xy[0], cpg_cfg.femur_xy[1])
            self.TIBIA_REST_ANGLE_RELATIVE = (
                math.atan2(cpg_cfg.tibia_xy[0], cpg_cfg.tibia_xy[1]) - self.FEMUR_REST_ANGLE_GLOBAL
            )
        else:
            raise ValueError(
                "CPGConfig must provide either femur_rest_angle+tibia_rest_angle or femur_xy+tibia_xy."
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
            # Use anchored patterns to avoid accidental partial regex matches.
            coxa_pattern = f"^{re.escape(leg_conf['coxa'])}$"
            femur_pattern = f"^{re.escape(leg_conf['femur'])}$"
            tibia_pattern = f"^{re.escape(leg_conf['tibia'])}$"
            c_ids, c_names = self._asset.find_joints([coxa_pattern])
            f_ids, f_names = self._asset.find_joints([femur_pattern])
            t_ids, t_names = self._asset.find_joints([tibia_pattern])
            
            if len(c_ids) > 0 and len(f_ids) > 0 and len(t_ids) > 0:
                phase_offset_deg = leg_conf.get("phase_offset_deg", 0.0)
                frequency = leg_conf.get("frequency", self.step_frequency)
                step_length = leg_conf.get("step_length", self.step_length)
                step_height = leg_conf.get("step_height", self.step_height)
                ground_height = leg_conf.get("ground_height", self.ground_height)
                center_offset = leg_conf.get("center_offset", self.center_offset)
                hip_offset = abs(leg_conf.get("hip_offset", self.L_HIP))
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
                    "hip_offset": hip_offset,
                    "direction_multiplier": leg_conf.get("direction_multiplier", 1.0),
                    "side": leg_conf.get("side", "left"),  # "left" or "right" for differential turning
                })
                leg_omegas.append(2.0 * math.pi * frequency)
                leg_initial_phases.append(math.radians(phase_offset_deg))
            else:
                print(f"[DummyJointPositionAction] Warning: Could not find joints for leg {leg_name}")
                if self._debug_print_enabled:
                    print(
                        f"[CPGDebug] leg={leg_name} joint lookup "
                        f"coxa({coxa_pattern})={list(c_names)} "
                        f"femur({femur_pattern})={list(f_names)} "
                        f"tibia({tibia_pattern})={list(t_names)}"
                    )
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
            self._leg_hip_offsets = torch.tensor(
                [leg["hip_offset"] for leg in self.legs], device=self.device, dtype=torch.float32
            )
            self._leg_direction_multipliers = torch.tensor(
                [leg.get("direction_multiplier", 1.0) for leg in self.legs], device=self.device, dtype=torch.float32
            )
            # Side: 1.0 for left, -1.0 for right (for turn rate calculation)
            self._leg_side_signs = torch.tensor(
                [1.0 if leg.get("side", "left") == "left" else -1.0 for leg in self.legs], 
                device=self.device, dtype=torch.float32
            )
            self._leg_signed_hip_offsets = self._leg_hip_offsets * self._leg_side_signs
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

        print(f"[CPGPositionAction] Initialized with {self._leg_count} legs configured")
        print(f"[CPGPositionAction] Default Freq={self.step_frequency}, Height={self.step_height}, Length={self.step_length}")
        print(
            f"[CPGPositionAction] RL Action Dim={self._rl_action_dim} "
            "[d_height, d_length, d_freq, d_turn] (residual over command)"
        )
        if self._debug_print_enabled and self._leg_count > 0:
            all_joint_ids, all_joint_names = self._asset.find_joints([".*"])
            id_to_name = {int(j_id): j_name for j_id, j_name in zip(all_joint_ids, all_joint_names)}
            for leg in self.legs:
                c_idx = int(leg["coxa_idx"])
                f_idx = int(leg["femur_idx"])
                t_idx = int(leg["tibia_idx"])
                print(
                    f"[CPGDebug] leg={leg['name']} indices "
                    f"coxa={c_idx}:{id_to_name.get(c_idx, '?')} "
                    f"femur={f_idx}:{id_to_name.get(f_idx, '?')} "
                    f"tibia={t_idx}:{id_to_name.get(t_idx, '?')}"
                )

    @property
    def action_dim(self) -> int:
        """RL action dimension: [d_step_height, d_step_length, d_frequency, d_turn_rate]."""
        return self._rl_action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        """
        Process RL actions as residuals on top of command-driven CPG parameters.

        Action format: [d_step_height, d_step_length, d_frequency, d_turn_rate] in [-1, 1].
        Zero action means "no residual", i.e. pure command->CPG mapping.
        """
        self._raw_actions[:] = actions

        action_clamped = torch.clamp(actions, -1.0, 1.0)
        self._rl_step_height_residual = action_clamped[:, 0] * self.cfg.step_height_residual_scale
        self._rl_step_length_residual = action_clamped[:, 1] * self.cfg.step_length_residual_scale
        self._rl_frequency_residual = action_clamped[:, 2] * self.cfg.step_frequency_residual_scale
        self._rl_turn_rate_residual = action_clamped[:, 3] * self.cfg.turn_rate_residual_scale

    @staticmethod
    def _cubic_hermite(
        u: torch.Tensor,
        p0: torch.Tensor,
        p1: torch.Tensor,
        v0: torch.Tensor,
        v1: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """Cubic Hermite spline segment used by the wb-mpc swing profile."""
        h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
        h10 = u**3 - 2.0 * u**2 + u
        h01 = -2.0 * u**3 + 3.0 * u**2
        h11 = u**3 - u**2
        return h00 * p0 + h10 * dt * v0 + h01 * p1 + h11 * dt * v1

    def _compute_gait_schedule(self, phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return swing mask and normalized swing phase [0, 1] for each leg."""
        if self._gait_type == "stand" or self._swing_ratio <= 0.0:
            swing_mask = torch.zeros_like(phi, dtype=torch.bool)
            swing_phase = torch.zeros_like(phi)
            return swing_mask, swing_phase

        leg_phase = (phi / (2.0 * torch.pi)) % 1.0
        swing_mask = leg_phase < self._swing_ratio
        swing_phase = torch.where(swing_mask, leg_phase / self._swing_ratio, torch.zeros_like(leg_phase))
        return swing_mask, swing_phase

    def _compute_swing_height(
        self,
        swing_phase: torch.Tensor,
        swing_mask: torch.Tensor,
        step_height_expanded: torch.Tensor,
        frequency: torch.Tensor,
    ) -> torch.Tensor:
        """Compute swing-foot z profile using wb-mpc style cubic spline segments."""
        if self._gait_type == "stand" or self._swing_ratio <= 0.0:
            return torch.zeros_like(swing_phase)

        freq_safe = torch.clamp(frequency, min=1.0e-6).unsqueeze(1)
        swing_period = self._swing_ratio / freq_safe
        half_period = 0.5 * swing_period
        v_liftoff = torch.full_like(swing_phase, self.cfg.swing_vel_limits[0])
        v_touchdown = torch.full_like(swing_phase, self.cfg.swing_vel_limits[1])

        first_half = swing_phase < 0.5
        u1 = torch.clamp(2.0 * swing_phase, 0.0, 1.0)
        z1 = self._cubic_hermite(
            u=u1,
            p0=torch.zeros_like(swing_phase),
            p1=step_height_expanded,
            v0=v_liftoff,
            v1=torch.zeros_like(swing_phase),
            dt=half_period,
        )

        u2 = torch.clamp(2.0 * (swing_phase - 0.5), 0.0, 1.0)
        z2 = self._cubic_hermite(
            u=u2,
            p0=step_height_expanded,
            p1=torch.zeros_like(swing_phase),
            v0=torch.zeros_like(swing_phase),
            v1=v_touchdown,
            dt=half_period,
        )

        z_swing = torch.where(first_half, z1, z2)
        return torch.where(swing_mask, z_swing, torch.zeros_like(z_swing))

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
        
        The leg local frame is defined as:
        - Origin: at the coxa joint
        - X-axis: pointing radially outward from body center (along the leg)
        - Y-axis: perpendicular to X in the horizontal plane
        - Z-axis: vertical (up is positive)
        
        The trajectory is a D-shaped path in a plane that can be rotated by angle_rad
        around the Z-axis relative to the leg's radial direction.
        
        Args:
            phase: Current phase angle [0, 2*pi) for each environment
            angle_rad: Rotation angle of the trajectory plane relative to leg's radial direction
                       (0 = radial/forward-backward, pi/2 = tangential/sideway)
            step_length: Total length of one step in the walking direction
            step_height: Maximum height of leg lift during swing phase
            center_offset: Radial distance from coxa joint to foot (in leg local X direction)
            ground_height: Z position of ground relative to coxa joint (typically negative)
            direction: Walking direction multiplier (+1 forward, -1 backward, 0 stationary)
                      This controls which direction the body moves by changing where the
                      foot lands during swing phase.
        
        Returns:
            X, Y, Z: Target foot position in leg local coordinate frame
        """
        # Wrap phase to [0, 2*pi)
        phi = phase % (2 * math.pi)

        # Initialize trajectory displacement in the walking direction
        d_walk = torch.zeros_like(phi)  # displacement along walking direction
        z_loc = torch.zeros_like(phi)   # vertical displacement
        
        # The key insight for direction control:
        # - During STANCE phase (foot on ground), the foot pushes backward relative to body
        #   to propel the body forward. The foot position goes from FRONT to REAR.
        # - During SWING phase (foot in air), the foot moves forward to prepare for next stance.
        #   The foot position goes from REAR to FRONT.
        #
        # For FORWARD walking (direction > 0):
        #   Swing (0->pi): foot moves from rear (-) to front (+) in air
        #   Stance (pi->2pi): foot moves from front (+) to rear (-) on ground, pushing body forward
        #
        # For BACKWARD walking (direction < 0):
        #   We flip the foot positions: swing goes front to rear, stance goes rear to front
        #   This makes the stance phase push the body backward
        
        # Base trajectory: -cos(phi) goes from -1 (at phi=0) to +1 (at phi=pi) to -1 (at phi=2pi)
        # Multiplied by step_length/2 gives position from -step_length/2 to +step_length/2
        
        # --- Swing Phase (0 <= phi <= pi) ---
        swing_mask = phi <= math.pi
        # direction multiplier flips which end is "front" vs "rear"
        d_walk[swing_mask] = direction * (-(step_length / 2.0) * torch.cos(phi[swing_mask]))
        z_loc[swing_mask] = step_height * torch.sin(phi[swing_mask])
        
        # --- Stance Phase (pi < phi <= 2*pi) ---
        stance_mask = ~swing_mask
        d_walk[stance_mask] = direction * (-(step_length / 2.0) * torch.cos(phi[stance_mask]))
        z_loc[stance_mask] = 0.0
        
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
        hip_offset_signed: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Solve batched IK for a quadruped leg with HAA(x)-HFE(y)-KFE(y) axes.

        Coordinates are in each leg-local frame:
        - X: forward/backward in sagittal direction
        - Y: lateral (left positive)
        - Z: up positive
        """
        if hip_offset_signed is None:
            y_offset = torch.full_like(target_x, self.L_HIP)
        elif isinstance(hip_offset_signed, torch.Tensor):
            y_offset = hip_offset_signed
        else:
            y_offset = torch.full_like(target_x, float(hip_offset_signed))

        # 1) HAA angle around x-axis.
        # Choose branch that keeps theta1 close to zero near nominal standing posture.
        yz_radius = torch.sqrt(target_y**2 + target_z**2).clamp_min(1.0e-6)
        cos_arg = torch.clamp(y_offset / yz_radius, -1.0, 1.0)
        theta1_absolute = -torch.acos(cos_arg) - torch.atan2(target_z, target_y)

        # Rotate target into HFE-KFE sagittal plane.
        sin_t1 = torch.sin(theta1_absolute)
        cos_t1 = torch.cos(theta1_absolute)
        z_plane = target_y * sin_t1 + target_z * cos_t1
        x_plane = target_x
        L_virtual = torch.sqrt(x_plane**2 + z_plane**2).clamp_min(1.0e-6)

        # 2) HFE/KFE two-link IK in x-z plane.
        cos_beta = (self.L_FEMUR**2 + self.L_TIBIA**2 - L_virtual**2) / (2 * self.L_FEMUR * self.L_TIBIA)
        cos_beta = torch.clamp(cos_beta, -1.0, 1.0)
        beta = torch.acos(cos_beta)

        cos_alpha = (self.L_FEMUR**2 + L_virtual**2 - self.L_TIBIA**2) / (2 * self.L_FEMUR * L_virtual)
        cos_alpha = torch.clamp(cos_alpha, -1.0, 1.0)
        alpha = torch.acos(cos_alpha)

        gamma = torch.atan2(z_plane, x_plane)
        theta2_absolute = gamma + alpha
        theta3_relative = beta - math.pi

        # 3) Convert to control deltas around configured rest angles.
        d_theta1 = theta1_absolute - self.HIP_REST_ANGLE
        d_theta2 = theta2_absolute - self.FEMUR_REST_ANGLE_GLOBAL
        d_theta3 = theta3_relative - self.TIBIA_REST_ANGLE_RELATIVE

        return d_theta1, d_theta2, d_theta3

    def apply_actions(self):
        """Apply the CPG trajectory to the joints using RL-controlled parameters.
        
        This is a fully vectorized implementation to avoid Python loops in the hot path.
        """
        # Increment step counter
        self._step_counter += 1
        
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
        cmd_heading = torch.atan2(cmd_lin_y, cmd_lin_x)
        cmd_heading = torch.where(
            cmd_lin_speed > self.cfg.command_lin_speed_deadband,
            cmd_heading,
            torch.zeros_like(cmd_heading),
        )

        h_min, h_max = self._step_height_range
        l_min, l_max = self._step_length_range
        f_min, f_max = self._frequency_range

        base_step_length = torch.clamp(
            cmd_lin_speed * self.cfg.command_speed_to_step_length,
            min=l_min,
            max=l_max,
        )
        if self.cfg.command_min_step_length > 0.0:
            moving_cmd = cmd_lin_speed > self.cfg.command_lin_speed_deadband
            min_stride = torch.full_like(base_step_length, self.cfg.command_min_step_length)
            base_step_length = torch.where(moving_cmd, torch.maximum(base_step_length, min_stride), base_step_length)
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

        self._rl_step_height = torch.clamp(self.step_height + self._rl_step_height_residual, h_min, h_max)
        self._rl_step_length = torch.clamp(base_step_length + self._rl_step_length_residual, l_min, l_max)
        self._rl_frequency = torch.clamp(base_frequency + self._rl_frequency_residual, f_min, f_max)
        self._rl_turn_rate = torch.clamp(base_turn_rate + self._rl_turn_rate_residual, -1.0, 1.0)

        # Compute per-environment omega from command-driven + residual frequency.
        rl_omegas = (2.0 * torch.pi) * self._rl_frequency
        
        # Update leg phases for each environment
        # Expand rl_omegas to (num_envs, num_legs) and update all phases at once
        omega_expanded = rl_omegas.unsqueeze(1)  # (num_envs, 1)
        self._leg_phases = (self._leg_phases + omega_expanded * self._dt) % (2.0 * torch.pi)
        
        # Reset all joint actions to 0.0
        self._processed_actions.zero_()
        
        # --- Vectorized computation for all legs at once ---
        # Shapes:
        #   self._leg_phases: (num_envs, num_legs)
        #   self._rl_step_length: (num_envs,)
        #   self._rl_step_height: (num_envs,)
        #   self._rl_turn_rate: (num_envs,)
        
        # Turning is implemented by adding opposite signed stride to left/right legs.
        turn_stride = self._rl_turn_rate.unsqueeze(1) * self.cfg.yaw_step_length_max * self._leg_side_signs.unsqueeze(0)
        effective_step_length = self._rl_step_length.unsqueeze(1) + turn_stride
        combined_direction = (self.step_direction * self._leg_direction_multipliers).unsqueeze(0)
        effective_step_length = effective_step_length * combined_direction
        
        # Compute trajectory for all legs at once
        # phase: (num_envs, num_legs)
        phi = self._leg_phases % (2.0 * torch.pi)
        swing_mask, swing_phase = self._compute_gait_schedule(phi)

        # Step height expanded: (num_envs, 1) -> broadcast to (num_envs, num_legs)
        step_height_expanded = self._rl_step_height.unsqueeze(1)

        # wb-mpc style phase-based horizontal progression:
        # swing: rear -> front, stance: front -> rear
        if self._gait_type == "stand" or self._swing_ratio <= 0.0:
            d_walk = torch.zeros_like(phi)
            z_loc = torch.zeros_like(phi)
        else:
            leg_phase = (phi / (2.0 * torch.pi)) % 1.0
            swing_phase_linear = torch.clamp(leg_phase / self._swing_ratio, 0.0, 1.0)
            stance_den = max(1.0 - self._swing_ratio, 1.0e-6)
            stance_phase_linear = torch.clamp((leg_phase - self._swing_ratio) / stance_den, 0.0, 1.0)

            d_swing = -0.5 * effective_step_length + effective_step_length * swing_phase_linear
            d_stance = 0.5 * effective_step_length - effective_step_length * stance_phase_linear
            d_walk = torch.where(swing_mask, d_swing, d_stance)

            z_loc = self._compute_swing_height(
                swing_phase=swing_phase,
                swing_mask=swing_mask,
                step_height_expanded=step_height_expanded,
                frequency=self._rl_frequency,
            )
        
        # Determine if each environment should be moving.
        has_stride = torch.any(torch.abs(effective_step_length) > self.cfg.command_lin_speed_deadband, dim=1)
        is_moving = (self._rl_frequency > 1e-6) & (self._rl_step_height > 1e-6) & has_stride

        # Transform to leg local coordinate frame.
        # Command heading rotates the gait direction for omnidirectional motion.
        walking_angle = self._leg_angle_rads.unsqueeze(0) + cmd_heading.unsqueeze(1)
        angle_cos = torch.cos(walking_angle)
        angle_sin = torch.sin(walking_angle)
        
        dx = d_walk * angle_cos
        dy = d_walk * angle_sin
        
        # center_offset and ground_height: (num_legs,) -> (1, num_legs)
        center_offset_expanded = self._leg_center_offsets.unsqueeze(0)
        ground_height_expanded = self._leg_ground_heights.unsqueeze(0)
        
        # Final target positions: (num_envs, num_legs)
        # Add signed hip lateral offset so HAA neutral pose aligns with leg workspace.
        hip_offset_expanded = self._leg_signed_hip_offsets.unsqueeze(0)
        target_x = center_offset_expanded + dx
        target_y = hip_offset_expanded + dy
        target_z = ground_height_expanded + z_loc

        # --- Vectorized IK for all legs ---
        d_theta1, d_theta2, d_theta3 = self._solve_ik(
            target_x=target_x,
            target_y=target_y,
            target_z=target_z,
            hip_offset_signed=hip_offset_expanded,
        )
        
        # For non-moving environments, set joint angles to 0
        is_moving_expanded = is_moving.unsqueeze(1)  # (num_envs, 1)
        d_theta1 = torch.where(is_moving_expanded, d_theta1, torch.zeros_like(d_theta1))
        d_theta2 = torch.where(is_moving_expanded, d_theta2, torch.zeros_like(d_theta2))
        d_theta3 = torch.where(is_moving_expanded, d_theta3, torch.zeros_like(d_theta3))
        
        # Scatter results to processed_actions using advanced indexing
        # This avoids the Python for loop entirely
        self._processed_actions.scatter_(1, self._leg_coxa_indices.unsqueeze(0).expand(self.num_envs, -1), d_theta1)
        self._processed_actions.scatter_(1, self._leg_femur_indices.unsqueeze(0).expand(self.num_envs, -1), d_theta2)
        self._processed_actions.scatter_(1, self._leg_tibia_indices.unsqueeze(0).expand(self.num_envs, -1), d_theta3)

        # Set position targets
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

        if self._debug_print_enabled and int(self._step_counter[0].item()) % self._debug_print_interval == 0:
            env_idx = min(self._debug_env_index, self.num_envs - 1)
            hfe_env = d_theta2[env_idx]
            haa_env = d_theta1[env_idx]
            kfe_env = d_theta3[env_idx]
            print(
                "[CPGDebug] "
                f"env={env_idx} "
                f"cmd=({cmd_lin_x[env_idx].item():+.3f},{cmd_lin_y[env_idx].item():+.3f},{cmd_ang_z[env_idx].item():+.3f}) "
                f"heading={math.degrees(cmd_heading[env_idx].item()):+.1f}deg "
                f"step_len={self._rl_step_length[env_idx].item():.4f} "
                f"step_h={self._rl_step_height[env_idx].item():.4f} "
                f"freq={self._rl_frequency[env_idx].item():.3f} "
                f"turn={self._rl_turn_rate[env_idx].item():+.3f} "
                f"haa_rng={float((haa_env.max() - haa_env.min()).item()):.4f} "
                f"hfe_rng={float((hfe_env.max() - hfe_env.min()).item()):.4f} "
                f"kfe_rng={float((kfe_env.max() - kfe_env.min()).item()):.4f}"
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
        # Wrap phase to [0, 2*pi)
        phi = phase % (2 * math.pi)

        # Initialize trajectory displacement
        d_walk = torch.zeros_like(phi)
        z_loc = torch.zeros_like(phi)
        
        # --- Swing Phase (0 <= phi <= pi) ---
        swing_mask = phi <= math.pi
        d_walk[swing_mask] = direction * (-(step_length[swing_mask] / 2.0) * torch.cos(phi[swing_mask]))
        z_loc[swing_mask] = step_height[swing_mask] * torch.sin(phi[swing_mask])
        
        # --- Stance Phase (pi < phi <= 2*pi) ---
        stance_mask = ~swing_mask
        d_walk[stance_mask] = direction * (-(step_length[stance_mask] / 2.0) * torch.cos(phi[stance_mask]))
        z_loc[stance_mask] = 0.0
        
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
        
        if self._leg_count > 0:
            self._leg_phases[env_ids] = self._initial_leg_phases


@configclass
class CPGPositionActionCfg(ActionTermCfg):
    """
    Configuration for the CPG position action term with RL interface.

    Quadruped chassis control baseline is driven by command ``command_name`` and RL only outputs residuals:
    RL Action Format: [d_step_height, d_step_length, d_frequency, d_turn_rate] in [-1, 1].
    Zero action means no residual and therefore pure command-driven gait controller.
    """

    class_type: type[ActionTerm] = CPGPositionAction

    joint_names: list[str] = [".*"]
    
    # Optional: List of leg names to enable (None = all legs enabled)
    enabled_leg_names: list[str] | None = None
    
    # Default CPG Parameters (used as initial values and for reset)
    step_height: float = 0.03      # Default step height (m)
    step_length: float = 0.05      # Default step length (m)
    step_frequency: float = 1.0    # Default frequency (Hz)
    step_direction: float = 1.0    # 1.0 = Forward, -1.0 = Backward (fixed, not RL controlled)
    turn_rate: float = 0.0         # Default turn rate (kept for compatibility)
    gait_type: str = "trot"        # "trot", "walk", or "stand"
    swing_vel_limits: tuple[float, float] = (0.1, -0.2)  # liftoff / touchdown z-velocity targets (m/s)
    
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

    # Optional runtime diagnostics
    debug_print_enabled: bool = False
    debug_print_interval: int = 120
    debug_env_index: int = 0
    
    # CPG IK configuration
    cpg_config: CPGConfig = CPGConfig()

    # Trajectory geometry parameters
    center_offset: float = 0.12   # default foot X position in HAA frame
    ground_height: float = -0.07  # default foot Z position in HAA frame
    
    # Leg Configuration: Name -> {Joint Names, Body Angle, Phase Offset, Side, hip_offset}
    # body_angle rotates the stride direction in leg-local XY.
    # phase_offset_deg uses a trot default: FL+RR vs FR+RL.
    # hip_offset is optional; if omitted, cpg_config.l_coxa is used.
    legs_config: dict = {
        "FL": {"coxa": "coxa_FL", "femur": "femur_FL", "tibia": "tibia_FL", "body_angle": 0.0, "phase_offset_deg": 0.0, "side": "left"},
        "FR": {"coxa": "coxa_FR", "femur": "femur_FR", "tibia": "tibia_FR", "body_angle": 0.0, "phase_offset_deg": 180.0, "side": "right"},
        "RL": {"coxa": "coxa_RL", "femur": "femur_RL", "tibia": "tibia_RL", "body_angle": 180.0, "phase_offset_deg": 180.0, "side": "left"},
        "RR": {"coxa": "coxa_RR", "femur": "femur_RR", "tibia": "tibia_RR", "body_angle": 180.0, "phase_offset_deg": 0.0, "side": "right"},
    }
