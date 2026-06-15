from __future__ import annotations

import math
from dataclasses import dataclass

import torch


LYNX_LEG_ORDER = ("FL", "FR", "RL", "RR")
LYNX_SIDE_SIGNS = {"FL": 1.0, "FR": -1.0, "RL": 1.0, "RR": -1.0}
# lynxc_defnitions: every right-side joint has the opposite positive direction
# from its left-side counterpart. The same sign applies to joints 0, 1, and 2.
LYNX_SIDE_JOINT_SIGN = {"FL": 1.0, "FR": -1.0, "RL": 1.0, "RR": -1.0}
TROT_PHASE_OFFSETS = {"FL": 0.0, "FR": math.pi, "RL": math.pi, "RR": 0.0}
WALK_SEQUENCE = ("FL", "RL", "RR", "FR")
WALK_SWING_FRACTION = 0.25


@dataclass(frozen=True)
class LynxGeometry:
    l_coxa: float = 0.075
    l_femur: float = math.sqrt(0.0602**2 + 0.22**2)
    l_tibia: float = math.sqrt(0.303431**2 + 0.0455**2 + 0.03**2)


class LynxGaitGenerator:
    """LynxC gait and inner-knee IK using the asset's verified joint-zero convention."""

    # lynxc_defnitions: at joint zero the thigh points +Z while the calf
    # points -Z, so the knee zero is a 180-degree relative rotation.
    FEMUR_ZERO_ANGLE = math.pi / 2.0
    TIBIA_ZERO_ANGLE = math.pi

    def __init__(
        self,
        geometry: LynxGeometry | None = None,
        leg_order: tuple[str, ...] = LYNX_LEG_ORDER,
        gait_type: str = "walk",
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.geometry = geometry if geometry is not None else LynxGeometry()
        self.leg_order = tuple(name.upper() for name in leg_order)
        unknown = set(self.leg_order) - set(LYNX_LEG_ORDER)
        if unknown:
            raise ValueError(f"Unsupported Lynx leg names: {sorted(unknown)}")
        self.gait_type = self._normalize_gait_type(gait_type)
        self.device = torch.device(device)
        self.dtype = dtype
        self.side_signs = torch.tensor(
            [LYNX_SIDE_SIGNS[name] for name in self.leg_order], device=self.device, dtype=self.dtype
        )
        side_joint_signs = torch.tensor(
            [LYNX_SIDE_JOINT_SIGN[name] for name in self.leg_order], device=self.device, dtype=self.dtype
        )
        self.joint_direction_signs = side_joint_signs.unsqueeze(-1).expand(-1, 3).clone()
        self.phase_offsets = torch.tensor(
            self.default_phase_offsets(self.gait_type, self.leg_order), device=self.device, dtype=self.dtype
        )

    @staticmethod
    def _normalize_gait_type(gait_type: str) -> str:
        normalized = str(gait_type).lower()
        if normalized not in {"trot", "walk"}:
            raise ValueError(f"Unsupported Lynx gait_type {gait_type!r}; expected 'trot' or 'walk'.")
        return normalized

    @classmethod
    def default_phase_offsets(cls, gait_type: str, leg_order: tuple[str, ...]) -> list[float]:
        normalized = cls._normalize_gait_type(gait_type)
        if normalized == "trot":
            return [TROT_PHASE_OFFSETS[name] for name in leg_order]
        slots = {name: idx for idx, name in enumerate(WALK_SEQUENCE)}
        return [2.0 * math.pi * ((-slots[name] / len(WALK_SEQUENCE)) % 1.0) for name in leg_order]

    def standing_foot_targets(self, center_x: float, ground_z: float) -> torch.Tensor:
        return torch.stack(
            (
                torch.full_like(self.side_signs, float(center_x)),
                self.side_signs * self.geometry.l_coxa,
                torch.full_like(self.side_signs, float(ground_z)),
            ),
            dim=-1,
        )

    def phase_to_targets(
        self,
        phases: torch.Tensor,
        step_vectors: torch.Tensor,
        step_heights: torch.Tensor | float,
        home_positions: torch.Tensor,
    ) -> torch.Tensor:
        phases = phases.to(device=self.device, dtype=self.dtype)
        step_vectors = step_vectors.to(device=self.device, dtype=self.dtype)
        home_positions = home_positions.to(device=self.device, dtype=self.dtype)
        heights = self._expand_param(step_heights, phases)

        if self.gait_type == "walk":
            cycle = torch.remainder(phases, 2.0 * math.pi) / (2.0 * math.pi)
            swing = cycle < WALK_SWING_FRACTION
            swing_t = torch.clamp(cycle / WALK_SWING_FRACTION, 0.0, 1.0)
            stance_t = torch.clamp((cycle - WALK_SWING_FRACTION) / (1.0 - WALK_SWING_FRACTION), 0.0, 1.0)
            stride = torch.where(swing, -0.5 + 0.5 * (1.0 - torch.cos(math.pi * swing_t)), 0.5 - stance_t)
            lift = torch.where(swing, heights * torch.sin(math.pi * swing_t), torch.zeros_like(cycle))
        else:
            phi = torch.remainder(phases + math.pi / 2.0, 2.0 * math.pi)
            stance = phi < math.pi
            stride = 0.5 * torch.cos(phi)
            lift = torch.where(stance, torch.zeros_like(phi), heights * torch.sin(phi - math.pi))

        targets = home_positions + torch.cat((step_vectors * stride.unsqueeze(-1), lift.unsqueeze(-1)), dim=-1)
        return targets

    def joint_targets_from_phase(
        self,
        phases: torch.Tensor,
        step_vectors: torch.Tensor,
        step_heights: torch.Tensor | float,
        home_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        foot_targets = self.phase_to_targets(phases, step_vectors, step_heights, home_positions)
        planner_targets, valid = self.solve_ik(foot_targets)
        return planner_targets, foot_targets, valid

    def solve_ik(self, foot_targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        foot_targets = foot_targets.to(device=self.device, dtype=self.dtype)
        side_signs = self.side_signs.to(device=foot_targets.device, dtype=foot_targets.dtype)
        x, y, z = foot_targets.unbind(dim=-1)
        g = self.geometry

        r_yz = torch.sqrt(y**2 + z**2)
        valid_coxa = r_yz >= g.l_coxa
        safe_r_yz = torch.clamp_min(r_yz, 1.0e-9)
        phi_yz = torch.atan2(z, y)
        delta = torch.acos(torch.clamp(side_signs * g.l_coxa / safe_r_yz, -1.0, 1.0))
        theta1_a = phi_yz - delta
        theta1_b = phi_yz + delta
        h_a = -y * torch.sin(theta1_a) + z * torch.cos(theta1_a)
        h_b = -y * torch.sin(theta1_b) + z * torch.cos(theta1_b)
        theta1 = torch.where(h_a < h_b, theta1_a, theta1_b)
        h = torch.minimum(h_a, h_b)

        virtual_length = torch.sqrt(x**2 + h**2)
        valid_length = virtual_length > 1.0e-9
        valid_reach = (virtual_length <= g.l_femur + g.l_tibia) & (
            virtual_length >= abs(g.l_femur - g.l_tibia)
        )
        valid = valid_coxa & valid_length & valid_reach
        safe_length = torch.clamp_min(virtual_length, 1.0e-9)

        beta = torch.acos(torch.clamp(
            (g.l_femur**2 + g.l_tibia**2 - virtual_length**2) / (2.0 * g.l_femur * g.l_tibia),
            -1.0,
            1.0,
        ))
        alpha = torch.acos(torch.clamp(
            (g.l_femur**2 + safe_length**2 - g.l_tibia**2) / (2.0 * g.l_femur * safe_length),
            -1.0,
            1.0,
        ))
        gamma = torch.atan2(h, x)

        # Match ik_lynxc_quadruped_test.py inner-knee mode: front knees
        # point toward -X and rear knees point toward +X in each hip frame.
        knee_backward = self._planner_targets(theta1, gamma - alpha, math.pi - beta)
        knee_forward = self._planner_targets(theta1, gamma + alpha, beta - math.pi)
        is_front = torch.tensor(
            [name.startswith("F") for name in self.leg_order], device=foot_targets.device, dtype=torch.bool
        )
        planner_targets = torch.where(is_front.unsqueeze(-1), knee_backward, knee_forward)
        return planner_targets, valid

    def planner_to_joint(self, planner_targets: torch.Tensor) -> torch.Tensor:
        """Convert body-frame IK angles to Lynx USD joint coordinates."""
        signs = self.joint_direction_signs.to(device=planner_targets.device, dtype=planner_targets.dtype)
        return planner_targets * signs

    def joint_to_planner(self, joint_targets: torch.Tensor) -> torch.Tensor:
        """Convert Lynx USD joint coordinates back to body-frame IK angles."""
        signs = self.joint_direction_signs.to(device=joint_targets.device, dtype=joint_targets.dtype)
        return joint_targets * signs

    # Compatibility aliases for callers created before the coordinate spaces were named.
    def model_to_sim(self, model_targets: torch.Tensor) -> torch.Tensor:
        return self.planner_to_joint(model_targets)

    def sim_to_model(self, sim_targets: torch.Tensor) -> torch.Tensor:
        return self.joint_to_planner(sim_targets)

    def forward_kinematics(self, planner_targets: torch.Tensor) -> torch.Tensor:
        planner_targets = planner_targets.to(device=self.device, dtype=self.dtype)
        side_signs = self.side_signs.to(device=planner_targets.device, dtype=planner_targets.dtype)
        theta1 = planner_targets[..., 0]
        theta2 = planner_targets[..., 1] + self.FEMUR_ZERO_ANGLE
        theta3 = planner_targets[..., 2] + self.TIBIA_ZERO_ANGLE
        c1, s1 = torch.cos(theta1), torch.sin(theta1)

        p0 = torch.zeros((*theta1.shape, 3), device=planner_targets.device, dtype=planner_targets.dtype)
        p1 = torch.stack(
            (torch.zeros_like(theta1), side_signs * self.geometry.l_coxa * c1, side_signs * self.geometry.l_coxa * s1),
            dim=-1,
        )
        femur_z = self.geometry.l_femur * torch.sin(theta2)
        femur = torch.stack((self.geometry.l_femur * torch.cos(theta2), -s1 * femur_z, c1 * femur_z), dim=-1)
        tibia_angle = theta2 + theta3
        tibia_z = self.geometry.l_tibia * torch.sin(tibia_angle)
        tibia = torch.stack((self.geometry.l_tibia * torch.cos(tibia_angle), -s1 * tibia_z, c1 * tibia_z), dim=-1)
        p2 = p1 + femur
        p3 = p2 + tibia
        return torch.stack((p0, p1, p2, p3), dim=-2)

    def _planner_targets(self, theta1: torch.Tensor, theta2: torch.Tensor, theta3: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                self._wrap_to_pi(theta1),
                self._wrap_to_pi(theta2 - self.FEMUR_ZERO_ANGLE),
                self._wrap_to_pi(theta3 - self.TIBIA_ZERO_ANGLE),
            ),
            dim=-1,
        )

    @staticmethod
    def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
        return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi

    @staticmethod
    def _expand_param(value: torch.Tensor | float, like: torch.Tensor) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(device=like.device, dtype=like.dtype) + torch.zeros_like(like)
        return torch.full_like(like, float(value))
