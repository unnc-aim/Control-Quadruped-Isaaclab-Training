import importlib.util
import math
import sys
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "tasks" / "mdp" / "quadruped_gait_generator.py"
SPEC = importlib.util.spec_from_file_location("quadruped_gait_generator", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

QuadrupedGaitGenerator = MODULE.QuadrupedGaitGenerator
QuadrupedGeometry = MODULE.QuadrupedGeometry


def test_fk_ik_roundtrip_respects_explicit_zero_angles() -> None:
    geometry = QuadrupedGeometry(
        femur_zero_angle_global=math.radians(-82.0),
        tibia_zero_angle_relative=math.radians(24.0),
    )
    generator = QuadrupedGaitGenerator(
        geometry=geometry,
        leg_order=("FL", "FR"),
        side_signs=(1.0, -1.0),
        device="cpu",
        dtype=torch.float64,
    )
    side_signs = generator.side_signs.to(dtype=torch.float64)
    standing_joint_targets = torch.tensor(
        [
            [math.radians(3.0), math.radians(-18.0), math.radians(72.0)],
            [math.radians(-3.0), math.radians(-18.0), math.radians(72.0)],
        ],
        dtype=torch.float64,
    )

    foot_targets = generator.forward_kinematics(standing_joint_targets, side_signs)[..., -1, :]
    solved_joint_targets, valid_ik = generator.solve_ik(foot_targets, side_signs)

    assert bool(valid_ik.all())
    torch.testing.assert_close(solved_joint_targets, standing_joint_targets, atol=1e-6, rtol=0.0)


def test_phase_targets_use_standing_home_positions_when_provided() -> None:
    geometry = QuadrupedGeometry(
        femur_zero_angle_global=math.radians(-82.0),
        tibia_zero_angle_relative=math.radians(24.0),
    )
    generator = QuadrupedGaitGenerator(
        geometry=geometry,
        leg_order=("FL", "FR"),
        side_signs=(1.0, -1.0),
        device="cpu",
        dtype=torch.float64,
    )
    side_signs = generator.side_signs.to(dtype=torch.float64)
    standing_joint_targets = torch.tensor(
        [
            [0.0, math.radians(-20.0), math.radians(68.0)],
            [0.0, math.radians(-20.0), math.radians(68.0)],
        ],
        dtype=torch.float64,
    )
    standing_home = generator.nominal_standing_foot_positions(standing_joint_targets, side_signs)
    zero_home = generator.home_foot_positions(side_signs).to(dtype=torch.float64)

    targets = generator.phase_to_targets(
        phases=torch.zeros((1, 2), dtype=torch.float64),
        step_vectors_body=torch.zeros((1, 2, 2), dtype=torch.float64),
        step_heights=0.05,
        ground_heights=-123.0,
        center_offsets=456.0,
        side_signs=side_signs,
        home_positions=standing_home.unsqueeze(0),
    )

    torch.testing.assert_close(targets, standing_home.unsqueeze(0), atol=1e-6, rtol=0.0)
    assert not torch.allclose(standing_home, zero_home)


def test_mastiff_joint_sign_table_maps_planner_targets_to_sim_space() -> None:
    planner_joint_targets = torch.tensor(
        [
            [0.2, 0.3, 0.4],
            [0.2, 0.3, 0.4],
            [0.2, 0.3, 0.4],
            [0.2, 0.3, 0.4],
        ],
        dtype=torch.float64,
    )
    joint_signs = torch.tensor(
        [
            [1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, -1.0],
        ],
        dtype=torch.float64,
    )
    sim_joint_targets = planner_joint_targets * joint_signs

    expected = torch.tensor(
        [
            [0.2, 0.3, -0.4],
            [0.2, -0.3, -0.4],
            [-0.2, 0.3, -0.4],
            [-0.2, -0.3, -0.4],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(sim_joint_targets, expected, atol=0.0, rtol=0.0)


def test_inward_knee_configuration_selects_opposite_front_rear_branches() -> None:
    geometry = QuadrupedGeometry(
        l_coxa=0.075,
        l_femur=math.sqrt(0.0602**2 + 0.22**2),
        l_tibia=math.sqrt(0.303431**2 + 0.0455**2 + 0.03**2),
        femur_zero_angle_global=math.radians(90.0),
        tibia_zero_angle_relative=0.0,
    )
    side_signs = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float64)
    generator = QuadrupedGaitGenerator(
        geometry=geometry,
        leg_order=("FL", "FR", "RL", "RR"),
        side_signs=side_signs,
        knee_direction_signs=(-1.0, -1.0, 1.0, 1.0),
        device="cpu",
        dtype=torch.float64,
    )
    foot_targets = torch.tensor(
        [
            [0.020, +0.075, -0.300],
            [0.020, -0.075, -0.300],
            [0.020, +0.075, -0.300],
            [0.020, -0.075, -0.300],
        ],
        dtype=torch.float64,
    )

    joint_targets, valid_ik = generator.solve_ik(foot_targets, side_signs)
    fk_points = generator.forward_kinematics(joint_targets, side_signs)
    knee_x = fk_points[:, 2, 0]

    assert bool(valid_ik.all())
    assert bool((knee_x[:2] < 0.0).all())
    assert bool((knee_x[2:] > 0.0).all())
    torch.testing.assert_close(fk_points[:, -1, :], foot_targets, atol=1e-6, rtol=0.0)
    assert bool((joint_targets.abs() <= math.pi + 1.0e-9).all())
