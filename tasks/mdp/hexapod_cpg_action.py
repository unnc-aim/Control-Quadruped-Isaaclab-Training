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
        
        # Create tensors for raw and processed actions
        # RL action dimension: [step_height_scale, step_length_scale, frequency_scale, turn_rate]
        self._rl_action_dim = 4
        self._raw_actions = torch.zeros(self.num_envs, self._rl_action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        
        # RL-controlled CPG parameters (per environment)
        # These will be updated by process_actions() based on RL output
        self._rl_step_height = torch.full((self.num_envs,), cfg.step_height, device=self.device)
        self._rl_step_length = torch.full((self.num_envs,), cfg.step_length, device=self.device)
        self._rl_frequency = torch.full((self.num_envs,), cfg.step_frequency, device=self.device)
        self._rl_turn_rate = torch.zeros(self.num_envs, device=self.device)
        
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
        print(f"[CPGPositionAction] RL Action Dim={self._rl_action_dim} [height_scale, length_scale, freq_scale, turn_rate]")

    @property
    def action_dim(self) -> int:
        """RL action dimension: [step_height_scale, step_length_scale, frequency_scale, turn_rate]"""
        return self._rl_action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        """
        Process RL actions and map them to CPG parameters.
        
        RL Policy typically outputs actions in [-1, 1] range.
        
        Action format: [step_height_action, step_length_action, frequency_action, turn_rate]
        - step_height_action: [-1, 1] -> maps to [step_height_min, step_height_max]
        - step_length_action: [-1, 1] -> maps to [step_length_min, step_length_max]
        - frequency_action: [-1, 1] -> maps to [step_frequency_min, step_frequency_max]
        - turn_rate: [-1, 1] -> used directly for differential turning
        """
        self._raw_actions[:] = actions
        
        # Clamp all actions to [-1, 1]
        height_action = torch.clamp(actions[:, 0], -1.0, 1.0)
        length_action = torch.clamp(actions[:, 1], -1.0, 1.0)
        freq_action = torch.clamp(actions[:, 2], -1.0, 1.0)
        turn_rate = torch.clamp(actions[:, 3], -1.0, 1.0)
        
        # Map [-1, 1] to [0, 1] scale
        height_scale = (height_action + 1.0) * 0.5
        length_scale = (length_action + 1.0) * 0.5
        freq_scale = (freq_action + 1.0) * 0.5
        
        # Map [0, 1] to parameter ranges
        h_min, h_max = self._step_height_range
        l_min, l_max = self._step_length_range
        f_min, f_max = self._frequency_range
        
        self._rl_step_height = h_min + height_scale * (h_max - h_min)
        self._rl_step_length = l_min + length_scale * (l_max - l_min)
        self._rl_frequency = f_min + freq_scale * (f_max - f_min)
        self._rl_turn_rate = turn_rate

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
        
        # Compute per-environment omega from RL-controlled frequency
        # omega = 2 * pi * frequency, shape: (num_envs,)
        rl_omegas = (2.0 * torch.pi) * self._rl_frequency
        
        # Update leg phases for each environment
        # Expand rl_omegas to (num_envs, num_legs) and update all phases at once
        omega_expanded = rl_omegas.unsqueeze(1)  # (num_envs, 1)
        self._leg_phases = (self._leg_phases + omega_expanded * self._dt) % (2.0 * torch.pi)
        
        # Reset all joint actions to 0.0
        self._processed_actions.zero_()
        
        # Determine if each environment should be moving, shape: (num_envs,)
        is_moving = (self._rl_frequency > 1e-6) & (self._rl_step_length > 1e-6) & (self._rl_step_height > 1e-6)
        
        # --- Vectorized computation for all legs at once ---
        # Shapes:
        #   self._leg_phases: (num_envs, num_legs)
        #   self._rl_step_length: (num_envs,)
        #   self._rl_step_height: (num_envs,)
        #   self._rl_turn_rate: (num_envs,)
        
        # Compute turn scale for all legs: left legs get (1 + turn_rate), right legs get (1 - turn_rate)
        # self._leg_side_signs: (num_legs,) with 1.0 for left, -1.0 for right
        # turn_scale: (num_envs, num_legs)
        turn_scale = 1.0 + self._rl_turn_rate.unsqueeze(1) * self._leg_side_signs.unsqueeze(0)
        turn_scale = torch.clamp(turn_scale, 0.0, 2.0)
        
        # Effective step length per leg per env: (num_envs, num_legs)
        effective_step_length = self._rl_step_length.unsqueeze(1) * turn_scale
        
        # Combined direction: global_direction * per_leg_multiplier, shape: (num_legs,)
        combined_direction = self.step_direction * self._leg_direction_multipliers
        
        # Compute trajectory for all legs at once
        # phase: (num_envs, num_legs)
        phi = self._leg_phases % (2.0 * torch.pi)
        
        # Step height expanded: (num_envs, 1) -> broadcast to (num_envs, num_legs)
        step_height_expanded = self._rl_step_height.unsqueeze(1)
        
        # direction expanded: (num_legs,) -> (1, num_legs) for broadcasting
        direction_expanded = combined_direction.unsqueeze(0)
        
        # Compute d_walk and z_loc for all envs and legs
        # d_walk: displacement along walking direction
        # z_loc: vertical displacement
        d_walk = direction_expanded * (-(effective_step_length / 2.0) * torch.cos(phi))
        
        # Swing phase mask: phi <= pi
        swing_mask = phi <= torch.pi
        z_loc = torch.where(swing_mask, step_height_expanded * torch.sin(phi), torch.zeros_like(phi))
        
        # Transform to leg local coordinate frame
        # angle_rad: (num_legs,) -> (1, num_legs)
        angle_cos = torch.cos(self._leg_angle_rads).unsqueeze(0)
        angle_sin = torch.sin(self._leg_angle_rads).unsqueeze(0)
        
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
        
        if self._leg_count > 0:
            self._leg_phases[env_ids] = self._initial_leg_phases


@configclass
class CPGPositionActionCfg(ActionTermCfg):
    """
    Configuration for the CPG position action term with RL interface.
    
    RL Action Format: [step_height, step_length, frequency, turn_rate] (all in [-1, 1])
    - step_height: [-1, 1] -> maps to [step_height_min, step_height_max]
    - step_length: [-1, 1] -> maps to [step_length_min, step_length_max]  
    - frequency: [-1, 1] -> maps to [step_frequency_min, step_frequency_max]
    - turn_rate: [-1, 1] -> differential turning (positive = turn right, negative = turn left)
    
    The default values (step_height, step_length, step_frequency) are used during reset.
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
    turn_rate: float = 0.0         # Default turn rate (not used directly, RL controls this)
    
    # RL Action Scaling Ranges
    # These define the min/max values that RL can command
    step_height_min: float = 0.0    # Minimum step height (m) - 0 means no lift
    step_height_max: float = 0.08   # Maximum step height (m) - 80mm
    step_length_min: float = 0.0    # Minimum step length (m) - 0 means no forward motion
    step_length_max: float = 0.12   # Maximum step length (m) - 120mm
    step_frequency_min: float = 0.0 # Minimum frequency (Hz) - 0 means stationary
    step_frequency_max: float = 3.0 # Maximum frequency (Hz) - 3 steps per second
    
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

