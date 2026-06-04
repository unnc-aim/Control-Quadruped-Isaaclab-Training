from __future__ import annotations

import math
from dataclasses import dataclass

import torch


DEFAULT_LEG_ORDER = ("FL", "FR", "RL", "RR")
DEFAULT_SIDE_SIGNS = {
    "FL": 1.0,
    "FR": -1.0,
    "RL": 1.0,
    "RR": -1.0,
}
DEFAULT_TROT_PHASE_OFFSETS = {
    "FL": 0.0,
    "FR": math.pi,
    "RL": math.pi,
    "RR": 0.0,
}


@dataclass(frozen=True)
class QuadrupedGeometry:
    """Kinematic constants for the Mastiff leg model, in meters and radians."""

    l_coxa: float = 0.12005
    l_femur: float = 0.260
    l_tibia: float = 0.300
    femur_zero_angle_global: float = math.radians(-150.0)
    tibia_zero_angle_relative: float = math.radians(15.0)


class QuadrupedGaitGenerator:
    """
    Body-frame quadruped gait generator.

    Coordinate convention:
    - +X is body forward in USD.
    - +Y is body left in USD.
    - +Z is up.
    - Every leg target is expressed in a HAA-origin frame whose axes are aligned
      with the body frame. The only left/right difference is the coxa lateral
      offset: left legs use +Y, right legs use -Y.
    """

    def __init__(
        self,
        geometry: QuadrupedGeometry | None = None,
        leg_order: tuple[str, ...] = DEFAULT_LEG_ORDER,
        side_signs: torch.Tensor | list[float] | tuple[float, ...] | None = None,
        phase_offsets: torch.Tensor | list[float] | tuple[float, ...] | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.geometry = geometry if geometry is not None else QuadrupedGeometry()
        self.leg_order = tuple(leg_order)
        self.device = torch.device(device)
        self.dtype = dtype
        if side_signs is None:
            side_signs = [DEFAULT_SIDE_SIGNS[name] for name in self.leg_order]
        if phase_offsets is None:
            phase_offsets = [DEFAULT_TROT_PHASE_OFFSETS[name] for name in self.leg_order]
        self.side_signs = torch.as_tensor(side_signs, device=self.device, dtype=self.dtype)
        self.phase_offsets = torch.as_tensor(phase_offsets, device=self.device, dtype=self.dtype)

    @property
    def num_legs(self) -> int:
        return len(self.leg_order)

    def home_foot_positions(self, side_signs: torch.Tensor | None = None) -> torch.Tensor:
        """Return zero-joint foot positions in the HAA-origin body-aligned frames."""
        if side_signs is None:
            side_signs = self.side_signs
        zeros = torch.zeros((*side_signs.shape, 3), device=side_signs.device, dtype=side_signs.dtype)
        return self.forward_kinematics(zeros, side_signs)[..., -1, :]

    def nominal_standing_foot_positions(
        self,
        standing_joint_targets: torch.Tensor,
        side_signs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return standing-pose foot positions in the HAA-origin body-aligned frames."""
        if side_signs is None:
            side_signs = self.side_signs
        return self.forward_kinematics(standing_joint_targets, side_signs)[..., -1, :]

    def phase_to_targets(
        self,
        phases: torch.Tensor,
        step_vectors_body: torch.Tensor,
        step_heights: torch.Tensor | float,
        ground_heights: torch.Tensor | float,
        center_offsets: torch.Tensor | float | None = None,
        side_signs: torch.Tensor | None = None,
        home_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Convert gait phase to foot targets.

        Args:
            phases: Shape ``(..., num_legs)``. A phase of 0 starts at stance
                mid-point after the internal ``pi/2`` shift.
            step_vectors_body: Shape ``(..., num_legs, 2)`` in body-aligned
                coordinates. Positive X means desired body-forward travel.
            step_heights: Scalar or shape ``(..., num_legs)``.
            ground_heights: Scalar or shape ``(..., num_legs)``.
            center_offsets: Optional scalar or shape ``(..., num_legs)`` for
                the nominal foot X. Defaults to the FK zero-pose X.
            side_signs: Optional shape ``(num_legs,)`` or broadcastable tensor.
            home_positions: Optional shape ``(..., num_legs, 3)`` standing or
                home pose. When provided, it overrides legacy
                ``center_offsets`` and ``ground_heights`` handling.
        """
        phases = phases.to(device=self.device, dtype=self.dtype)
        step_vectors_body = step_vectors_body.to(device=self.device, dtype=self.dtype)
        if side_signs is None:
            side_signs = self.side_signs
        side_signs = side_signs.to(device=phases.device, dtype=phases.dtype)

        step_heights_t = self._expand_param(step_heights, phases)

        phi = (phases + math.pi / 2.0) % (2.0 * math.pi)
        stance_mask = phi < math.pi

        xy_displacement = 0.5 * step_vectors_body * torch.cos(phi).unsqueeze(-1)
        z_lift = torch.where(
            stance_mask,
            torch.zeros_like(phi),
            step_heights_t * torch.sin(phi - math.pi),
        )

        if home_positions is not None:
            home_positions_t = home_positions.to(device=phases.device, dtype=phases.dtype)
            target_x = home_positions_t[..., 0] + xy_displacement[..., 0]
            target_y = home_positions_t[..., 1] + xy_displacement[..., 1]
            target_z = home_positions_t[..., 2] + z_lift
        else:
            ground_heights_t = self._expand_param(ground_heights, phases)
            if center_offsets is None:
                home_x = self.home_foot_positions(side_signs)[..., 0]
                center_offsets_t = torch.zeros_like(phases) + home_x
            else:
                center_offsets_t = self._expand_param(center_offsets, phases)
            target_x = center_offsets_t + xy_displacement[..., 0]
            target_y = self.geometry.l_coxa * side_signs + xy_displacement[..., 1]
            target_z = ground_heights_t + z_lift
        return torch.stack((target_x, target_y, target_z), dim=-1)

    def joint_targets_from_phase(
        self,
        phases: torch.Tensor,
        step_vectors_body: torch.Tensor,
        step_heights: torch.Tensor | float,
        ground_heights: torch.Tensor | float,
        center_offsets: torch.Tensor | float | None = None,
        side_signs: torch.Tensor | None = None,
        home_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(joint_targets, foot_targets, valid_ik)`` for a gait phase."""
        if side_signs is None:
            side_signs = self.side_signs
        foot_targets = self.phase_to_targets(
            phases,
            step_vectors_body,
            step_heights,
            ground_heights,
            center_offsets=center_offsets,
            side_signs=side_signs,
            home_positions=home_positions,
        )
        joint_targets, valid_ik = self.solve_ik(foot_targets, side_signs)
        return joint_targets, foot_targets, valid_ik

    def solve_ik(
        self,
        foot_targets: torch.Tensor,
        side_signs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Solve IK in the body-aligned HAA-origin frame.

        Returns:
            A tuple ``(joint_targets, valid_ik)``. ``joint_targets`` has shape
            ``(..., num_legs, 3)`` in HAA/HFE/KFE order and remains in planner
            space; asset-specific simulator direction mapping happens later.
        """
        foot_targets = foot_targets.to(device=self.device, dtype=self.dtype)
        if side_signs is None:
            side_signs = self.side_signs
        side_signs = side_signs.to(device=foot_targets.device, dtype=foot_targets.dtype)

        target_x = foot_targets[..., 0]
        target_y = foot_targets[..., 1]
        target_z = foot_targets[..., 2]

        l_coxa = self.geometry.l_coxa
        l_femur = self.geometry.l_femur
        l_tibia = self.geometry.l_tibia

        r_yz = torch.sqrt(target_y**2 + target_z**2)
        valid_coxa = r_yz >= l_coxa
        safe_r_yz = torch.clamp_min(r_yz, 1.0e-9)

        phi_yz = torch.atan2(target_z, target_y)
        delta = torch.acos(torch.clamp((side_signs * l_coxa) / safe_r_yz, -1.0, 1.0))

        theta1_a = phi_yz - delta
        theta1_b = phi_yz + delta
        h_a = -target_y * torch.sin(theta1_a) + target_z * torch.cos(theta1_a)
        h_b = -target_y * torch.sin(theta1_b) + target_z * torch.cos(theta1_b)

        # Pick the branch whose femur/tibia plane coordinate points downward.
        theta1 = torch.where(h_a < h_b, theta1_a, theta1_b)
        h = torch.minimum(h_a, h_b)
        w = target_x
        l_virtual = torch.sqrt(w**2 + h**2)

        valid_l_virtual = l_virtual > 1.0e-9
        valid_reach = (l_virtual <= (l_femur + l_tibia)) & (l_virtual >= abs(l_femur - l_tibia))
        valid_ik = valid_coxa & valid_l_virtual & valid_reach

        cos_beta = (l_femur**2 + l_tibia**2 - l_virtual**2) / (2.0 * l_femur * l_tibia)
        beta = torch.acos(torch.clamp(cos_beta, -1.0, 1.0))

        safe_l_virtual = torch.clamp_min(l_virtual, 1.0e-9)
        cos_alpha = (l_femur**2 + safe_l_virtual**2 - l_tibia**2) / (2.0 * l_femur * safe_l_virtual)
        alpha = torch.acos(torch.clamp(cos_alpha, -1.0, 1.0))

        gamma = torch.atan2(h, w)
        theta2_absolute = gamma - alpha
        theta3_relative = math.pi - beta

        d_theta1 = theta1
        d_theta2 = theta2_absolute - self.geometry.femur_zero_angle_global
        d_theta3 = theta3_relative - self.geometry.tibia_zero_angle_relative
        joint_targets = torch.stack((d_theta1, d_theta2, d_theta3), dim=-1)
        return joint_targets, valid_ik

    def forward_kinematics(
        self,
        joint_targets: torch.Tensor,
        side_signs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute FK for body-aligned leg frames.

        Args:
            joint_targets: Shape ``(..., num_legs, 3)`` or ``(..., 3)`` in
                HAA/HFE/KFE order.
            side_signs: Shape ``(num_legs,)`` or broadcastable tensor.

        Returns:
            Joint positions with shape ``(..., 4, 3)``.
        """
        joint_targets = joint_targets.to(device=self.device, dtype=self.dtype)
        if side_signs is None:
            if joint_targets.shape[-2] == self.num_legs:
                side_signs = self.side_signs
            else:
                side_signs = torch.ones(joint_targets.shape[:-1], device=joint_targets.device, dtype=joint_targets.dtype)
        side_signs = side_signs.to(device=joint_targets.device, dtype=joint_targets.dtype)

        theta1 = joint_targets[..., 0]
        theta2 = joint_targets[..., 1] + self.geometry.femur_zero_angle_global
        theta3 = joint_targets[..., 2] + self.geometry.tibia_zero_angle_relative

        c1 = torch.cos(theta1)
        s1 = torch.sin(theta1)

        p0 = torch.zeros((*theta1.shape, 3), device=joint_targets.device, dtype=joint_targets.dtype)
        p1 = torch.stack(
            (
                torch.zeros_like(theta1),
                side_signs * self.geometry.l_coxa * c1,
                side_signs * self.geometry.l_coxa * s1,
            ),
            dim=-1,
        )

        femur_z = self.geometry.l_femur * torch.sin(theta2)
        femur_vec = torch.stack(
            (
                self.geometry.l_femur * torch.cos(theta2),
                -s1 * femur_z,
                c1 * femur_z,
            ),
            dim=-1,
        )

        tibia_angle = theta2 + theta3
        tibia_z = self.geometry.l_tibia * torch.sin(tibia_angle)
        tibia_vec = torch.stack(
            (
                self.geometry.l_tibia * torch.cos(tibia_angle),
                -s1 * tibia_z,
                c1 * tibia_z,
            ),
            dim=-1,
        )

        p2 = p1 + femur_vec
        p3 = p2 + tibia_vec
        return torch.stack((p0, p1, p2, p3), dim=-2)

    def _expand_param(self, value: torch.Tensor | float, like: torch.Tensor) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(device=like.device, dtype=like.dtype) + torch.zeros_like(like)
        return torch.full_like(like, float(value))
