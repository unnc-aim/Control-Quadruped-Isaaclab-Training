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
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        
        # Step counter to track simulation steps (used for oscillation)
        self._step_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        
        # Get simulation timestep
        self._dt = env.physics_dt

        # --- Robot Geometric Parameters (from ik_test.py) ---
        self.L_COXA = 52.0
        self.L_FEMUR = 66.03
        self.L_TIBIA = 128.48
        
        # Initial pose absolute angles
        # Femur: Up 77.23 deg, Tibia: Relative to Femur Down 154.46 deg
        self.FEMUR_REST_ANGLE_GLOBAL = math.atan2(64.4, 14.6)
        # Tibia Rest Angle (Relative to Femur in geometric sense)
        self.TIBIA_REST_ANGLE_RELATIVE = math.atan2(-125.3, 28.4) - self.FEMUR_REST_ANGLE_GLOBAL

        # --- Trajectory Parameters ---
        self.step_length = cfg.step_length
        self.step_height = cfg.step_height
        self.ground_height = cfg.ground_height
        self.step_frequency = cfg.step_frequency
        self.step_direction = cfg.step_direction
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
        else:
            self._leg_omegas = torch.zeros(0, device=self.device, dtype=torch.float32)
            self._leg_phases = torch.zeros(self.num_envs, 0, device=self.device, dtype=torch.float32)
            self._initial_leg_phases = torch.zeros(0, device=self.device, dtype=torch.float32)

        print(f"[DummyJointPositionAction] Initialized with {self._leg_count} legs configured")
        print(f"[DummyJointPositionAction] Default Freq={self.step_frequency}, Dir={self.step_direction}")

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        """Process the input actions. For this dummy action, we ignore the input."""
        self._raw_actions[:] = actions

    def _compute_trajectory(
        self,
        phase: torch.Tensor,
        angle_rad: float,
        step_length: float,
        step_height: float,
        center_offset: float,
        ground_height: float,
        direction_sign: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute target (x, y, z) based on phase and leg angle.
        """
        # Wrap phase
        phi = phase % (2 * math.pi)
        
        dir_multiplier = 0.0 if direction_sign == 0.0 else (1.0 if direction_sign > 0.0 else -1.0)

        # Initialize local coordinates
        x_loc = torch.zeros_like(phi)
        z_loc = torch.zeros_like(phi)
        
        # --- Swing Phase (0 <= phi <= pi) ---
        swing_mask = phi <= math.pi
        x_loc[swing_mask] = dir_multiplier * (-(step_length / 2.0) * torch.cos(phi[swing_mask]))
        z_loc[swing_mask] = step_height * torch.sin(phi[swing_mask])
        
        # --- Stance Phase (pi < phi <= 2*pi) ---
        stance_mask = ~swing_mask
        x_loc[stance_mask] = dir_multiplier * (-(step_length / 2.0) * torch.cos(phi[stance_mask]))
        z_loc[stance_mask] = 0.0
        
        # --- Global Transformation (Rotation by body_angle) ---
        # x_rot = x_loc * cos(alpha)
        # y_rot = x_loc * sin(alpha)
        x_rot = x_loc * math.cos(angle_rad)
        y_rot = x_loc * math.sin(angle_rad)
        
        X = center_offset + x_rot
        Y = y_rot
        Z = ground_height + z_loc
        
        return X, Y, Z

    def _solve_ik(self, target_x: torch.Tensor, target_y: torch.Tensor, target_z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Solve Inverse Kinematics for a batch of targets.
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
        """Apply the CPG trajectory to the joints."""
        # Increment step counter
        self._step_counter += 1
        
        # Update leg phases independently
        if self._leg_count > 0:
            self._leg_phases = (self._leg_phases + self._leg_omegas * self._dt) % (2 * math.pi)
        
        # Reset all actions to 0.0
        self._processed_actions[:] = 0.0
        
        # Iterate over configured legs
        if self.step_direction > 0.0:
            direction_sign = 1.0
        elif self.step_direction < 0.0:
            direction_sign = -1.0
        else:
            direction_sign = 0.0
        for leg_idx, leg in enumerate(self.legs):
            if self._enabled_leg_names is not None and leg["name"] not in self._enabled_leg_names:
                continue
            leg_phase = self._leg_phases[:, leg_idx]
            leg_direction = direction_sign * leg.get("direction_multiplier", 1.0)
                
            # Compute target position
            target_x, target_y, target_z = self._compute_trajectory(
                leg_phase,
                leg["angle_rad"],
                leg["step_length"],
                leg["step_height"],
                leg["center_offset"],
                leg["ground_height"],
                leg_direction,
            )
            
            # Solve IK
            th1, th2, th3 = self._solve_ik(target_x, target_y, target_z)
            
            # Apply to joints
            self._processed_actions[:, leg["coxa_idx"]] = th1
            self._processed_actions[:, leg["femur_idx"]] = th2
            self._processed_actions[:, leg["tibia_idx"]] = th3

        # Set position targets
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the action term for the specified environments."""
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._step_counter[env_ids] = 0.0
        if self._leg_count > 0:
            self._leg_phases[env_ids] = self._initial_leg_phases


@configclass
class CPGPositionActionCfg(ActionTermCfg):
    """Configuration for the dummy joint position action term."""

    class_type: type[ActionTerm] = CPGPositionAction

    joint_names: list[str] = [".*"]
    
    # Default Control Parameters
    step_height: float = 50.0
    step_length: float = 70.0
    step_frequency: float = 1.0
    step_direction: float = 1.0 # 1.0 = Forward, -1.0 = Backward
    
    center_offset: float = 120.0
    ground_height: float = -70.0
    
    # Leg Configuration: Name -> {Joint Names, Body Angle}
    # Body Angle: Angle of the trajectory in the local leg frame (0=Radial, 90=Tangential)
    legs_config: dict = {
        "ML": {"coxa": "coxa_ML", "femur": "femur_ML", "tibia": "tibia_ML", "body_angle": -90.0},
        "MR": {"coxa": "coxa_MR", "femur": "femur_MR", "tibia": "tibia_MR", "body_angle": 90.0},
        "FL": {"coxa": "coxa_FL", "femur": "femur_FL", "tibia": "tibia_FL", "body_angle": 315.0},
        "FR": {"coxa": "coxa_FR", "femur": "femur_FR", "tibia": "tibia_FR", "body_angle": -315.0},
        "RL": {"coxa": "coxa_RL", "femur": "femur_RL", "tibia": "tibia_RL", "body_angle": -135.0},
        "RR": {"coxa": "coxa_RR", "femur": "femur_RR", "tibia": "tibia_RR", "body_angle": 135.0},
    }

