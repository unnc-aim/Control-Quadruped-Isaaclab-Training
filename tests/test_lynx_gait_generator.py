import importlib.util
import math
import sys
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "tasks" / "mdp" / "lynx_gait_generator.py"
SPEC = importlib.util.spec_from_file_location("lynx_gait_generator", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

LynxGaitGenerator = MODULE.LynxGaitGenerator


def test_standing_ik_matches_inner_knee_geometry() -> None:
    generator = LynxGaitGenerator(gait_type="walk", dtype=torch.float64)
    feet = generator.standing_foot_targets(center_x=0.020, ground_z=-0.300)

    planner_targets, valid = generator.solve_ik(feet)
    links = generator.forward_kinematics(planner_targets)

    assert bool(valid.all())
    torch.testing.assert_close(links[:, -1], feet, atol=1.0e-8, rtol=1.0e-8)
    assert bool((links[:2, 2, 0] > 0.0).all())  # front knees point forward
    assert bool((links[2:, 2, 0] < 0.0).all())  # rear knees point backward


def test_zero_pose_matches_lynxc_link_directions() -> None:
    generator = LynxGaitGenerator(gait_type="walk", dtype=torch.float64)
    zero_targets = torch.zeros(4, 3, dtype=torch.float64)

    links = generator.forward_kinematics(zero_targets)
    thigh_vectors = links[:, 2] - links[:, 1]
    calf_vectors = links[:, 3] - links[:, 2]

    assert bool((thigh_vectors[:, 2] > 0.0).all())
    assert bool((calf_vectors[:, 2] < 0.0).all())


def test_joint_directions_follow_lynxc_definitions() -> None:
    generator = LynxGaitGenerator(gait_type="trot", dtype=torch.float64)
    planner_targets = torch.tensor(
        [[0.1, -0.2, 0.3], [-0.4, 0.5, -0.6], [0.7, -0.8, 0.9], [-1.0, 1.1, -1.2]],
        dtype=torch.float64,
    )

    joint_targets = generator.planner_to_joint(planner_targets)

    torch.testing.assert_close(generator.joint_to_planner(joint_targets), planner_targets)
    torch.testing.assert_close(joint_targets[0], planner_targets[0])
    torch.testing.assert_close(joint_targets[1], -planner_targets[1])
    torch.testing.assert_close(joint_targets[2], planner_targets[2])
    torch.testing.assert_close(joint_targets[3], -planner_targets[3])
    torch.testing.assert_close(
        generator.joint_direction_signs,
        torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]], dtype=torch.float64),
    )


def test_walk_has_at_most_one_swing_leg() -> None:
    generator = LynxGaitGenerator(gait_type="walk", dtype=torch.float64)
    samples = torch.linspace(0.0, 2.0 * math.pi, 257, dtype=torch.float64)[:-1]
    phases = samples.unsqueeze(1) + generator.phase_offsets.unsqueeze(0)
    home = generator.standing_foot_targets(center_x=0.020, ground_z=-0.300).unsqueeze(0)
    step_vectors = torch.zeros(samples.shape[0], 4, 2, dtype=torch.float64)
    targets = generator.phase_to_targets(phases, step_vectors, 0.030, home)
    swing_count = (targets[..., 2] > home[..., 2] + 1.0e-10).sum(dim=1)

    assert int(swing_count.max()) <= 1


def test_walk_sequence_is_fl_rl_rr_fr() -> None:
    generator = LynxGaitGenerator(gait_type="walk", dtype=torch.float64)
    expected = torch.tensor([0.0, 0.5 * math.pi, 1.5 * math.pi, math.pi], dtype=torch.float64)

    torch.testing.assert_close(generator.phase_offsets, expected)
