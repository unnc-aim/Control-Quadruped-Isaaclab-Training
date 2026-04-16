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
        
        # Create tensors for raw and processed actions.
        # RL action is residual over command-driven CPG parameters:
        # [d_step_height, d_step_length, d_frequency, d_turn_rate]
        self._rl_action_dim = 4
        self._raw_actions = torch.zeros(self.num_envs, self._rl_action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._env = env
        
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

        # --- Robot Geometric Parameters (in meters) ---
        # Converted from mm: L_COXA=52mm, L_FEMUR=66.03mm, L_TIBIA=128.48mm
        self.L_COXA = 0.052
        self.L_FEMUR = 0.06603
        self.L_TIBIA = 0.12848
        
        # Initial pose absolute angles
        # Femur: Up 77.23 deg, Tibia: Relative to Femur Down 154.46 deg
        # Note: These ratios are dimensionless (from mm values 64.4/14.6 and -125.3/28.4)
        self.FEMUR_REST_ANGLE_GLOBAL = math.atan2(64.4, 14.6)
        # Tibia Rest Angle (Relative to Femur in geometric sense)
        self.TIBIA_REST_ANGLE_RELATIVE = math.atan2(-125.3, 28.4) - self.FEMUR_REST_ANGLE_GLOBAL

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

    def _solve_ik(self, target_x: torch.Tensor, target_y: torch.Tensor, target_z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Solve Inverse Kinematics for a batch of targets in the leg's local coordinate frame.
        
        The leg local frame has:
        - Origin at coxa joint
        - X-axis pointing radially outward
        - Z-axis pointing up
        
        Args:
            target_x: Target X position (radial distance from coxa)
            target_y: Target Y position (lateral offset)
            target_z: Target Z position (height relative to coxa)
            
        Returns:
            d_theta1: Coxa joint angle (rotation around Z)
            d_theta2: Femur joint angle delta (relative to rest pose)
            d_theta3: Tibia joint angle delta (relative to rest pose)
        """
        # --- 1. Solve Coxa (Theta 1) ---
        theta1 = torch.atan2(target_y, target_x)
        
        # --- 2. Transform to Femur-Tibia Plane ---
        r_projection = torch.sqrt(target_x**2 + target_y**2)
        w = r_projection - self.L_COXA
        h = target_z
        
        L_virtual = torch.sqrt(w**2 + h**2)
        
        # --- 4. Law of Cosines for J2, J3 ---
        cos_beta = (self.L_FEMUR**2 + self.L_TIBIA**2 - L_virtual**2) / (2 * self.L_FEMUR * self.L_TIBIA)
        cos_beta = torch.clamp(cos_beta, -1.0, 1.0)
        beta = torch.acos(cos_beta)
        
        cos_alpha = (self.L_FEMUR**2 + L_virtual**2 - self.L_TIBIA**2) / (2 * self.L_FEMUR * L_virtual)
        cos_alpha = torch.clamp(cos_alpha, -1.0, 1.0)
        alpha = torch.acos(cos_alpha)
        
        gamma = torch.atan2(h, w)
        
        theta2_absolute = gamma + alpha
        theta3_relative = beta - math.pi
        
        # --- 5. Convert to Control Deltas ---
        d_theta1 = theta1
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
        
        # Step height expanded: (num_envs, 1) -> broadcast to (num_envs, num_legs)
        step_height_expanded = self._rl_step_height.unsqueeze(1)
        
        # Compute d_walk and z_loc for all envs and legs
        # d_walk: displacement along walking direction
        # z_loc: vertical displacement
        d_walk = -(effective_step_length / 2.0) * torch.cos(phi)
        
        # Swing phase mask: phi <= pi
        swing_mask = phi <= torch.pi
        z_loc = torch.where(swing_mask, step_height_expanded * torch.sin(phi), torch.zeros_like(phi))
        
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
        target_x = center_offset_expanded + dx
        target_y = dy
        target_z = ground_height_expanded + z_loc
        
        # --- Vectorized IK for all legs ---
        # Solve coxa (theta1)
        theta1 = torch.atan2(target_y, target_x)
        
        # Transform to femur-tibia plane
        r_projection = torch.sqrt(target_x**2 + target_y**2)
        w = r_projection - self.L_COXA
        h = target_z
        L_virtual = torch.sqrt(w**2 + h**2)
        
        # Law of cosines
        cos_beta = (self.L_FEMUR**2 + self.L_TIBIA**2 - L_virtual**2) / (2 * self.L_FEMUR * self.L_TIBIA)
        cos_beta = torch.clamp(cos_beta, -1.0, 1.0)
        beta = torch.acos(cos_beta)
        
        cos_alpha = (self.L_FEMUR**2 + L_virtual**2 - self.L_TIBIA**2) / (2 * self.L_FEMUR * L_virtual)
        cos_alpha = torch.clamp(cos_alpha, -1.0, 1.0)
        alpha = torch.acos(cos_alpha)
        
        gamma = torch.atan2(h, w)
        
        theta2_absolute = gamma + alpha
        theta3_relative = beta - torch.pi
        
        # Convert to control deltas
        d_theta1 = theta1
        d_theta2 = theta2_absolute - self.FEMUR_REST_ANGLE_GLOBAL
        d_theta3 = theta3_relative - self.TIBIA_REST_ANGLE_RELATIVE
        
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

    CPG baseline is driven by command ``command_name`` and RL only outputs residuals:
    RL Action Format: [d_step_height, d_step_length, d_frequency, d_turn_rate] in [-1, 1].
    Zero action means no residual and therefore pure command-driven CPG.
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
    command_lin_speed_deadband: float = 1.0e-3
    yaw_step_length_max: float = 0.08

    # RL residual scales
    step_height_residual_scale: float = 0.02
    step_length_residual_scale: float = 0.03
    step_frequency_residual_scale: float = 0.5
    turn_rate_residual_scale: float = 0.25
    
    # Geometric Parameters
    center_offset: float = 0.12   # 120mm -> 0.12m (radial distance from coxa to foot)
    ground_height: float = -0.07  # -70mm -> -0.07m (foot height relative to coxa)
    
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
        "FL": {"coxa": "coxa_FL", "femur": "femur_FL", "tibia": "tibia_FL", "body_angle": -45.0, "phase_offset_deg": 0.0, "side": "left"},
        "MR": {"coxa": "coxa_MR", "femur": "femur_MR", "tibia": "tibia_MR", "body_angle": 90.0, "phase_offset_deg": 0.0, "side": "right"},
        "RL": {"coxa": "coxa_RL", "femur": "femur_RL", "tibia": "tibia_RL", "body_angle": -135.0, "phase_offset_deg": 0.0, "side": "left"},
        # Group B: FR, ML, RR (phase = 180°)
        "FR": {"coxa": "coxa_FR", "femur": "femur_FR", "tibia": "tibia_FR", "body_angle": 45.0, "phase_offset_deg": 180.0, "side": "right"},
        "ML": {"coxa": "coxa_ML", "femur": "femur_ML", "tibia": "tibia_ML", "body_angle": -90.0, "phase_offset_deg": 180.0, "side": "left"},
        "RR": {"coxa": "coxa_RR", "femur": "femur_RR", "tibia": "tibia_RR", "body_angle": 135.0, "phase_offset_deg": 180.0, "side": "right"},
    }
