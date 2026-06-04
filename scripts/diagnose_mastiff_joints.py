import argparse
from pathlib import Path
import sys

import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Diagnose Mastiff joint zero, sign, and limit behavior.")
parser.add_argument("--angles", type=float, nargs="+", default=[0.0, 0.3, -0.3, 0.8, -0.8], help="Joint angles in radians to test for each joint.")
parser.add_argument(
    "--joint_names",
    type=str,
    nargs="+",
    default=[
        "HAA_FRONT_LEFT", "HAA_FRONT_RIGHT", "HAA_REAR_LEFT", "HAA_REAR_RIGHT",
        "HFE_FRONT_LEFT",
        "HFE_FRONT_RIGHT",
        "HFE_REAR_LEFT",
        "HFE_REAR_RIGHT",
        "KFE_FRONT_LEFT",
        "KFE_FRONT_RIGHT",
        "KFE_REAR_LEFT",
        "KFE_REAR_RIGHT",
    ],
    help="Joint names to test in sequence.",
)
parser.add_argument("--settle_steps", type=int, default=120, help="Physics steps to wait after each command.")
parser.add_argument("--pause_steps", type=int, default=30, help="Physics steps to hold zero between commands.")
parser.add_argument("--hold_base_height", type=float, default=0.45, help="World-frame base height while holding the robot in the air.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext


_PROJECT_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_PATH))
from assets.Mastiff_CFG import Mastiff_CONFIG as ROBOT_CFG


def design_scene() -> tuple[Articulation, torch.Tensor]:
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    origin = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origin[0].tolist())

    robot_cfg = ROBOT_CFG.copy()
    robot_cfg.prim_path = "/World/Origin1/Robot"
    robot = Articulation(cfg=robot_cfg)
    return robot, origin


def get_joint_limits(robot: Articulation) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    joint_limits = getattr(robot.data, "soft_joint_pos_limits", None)
    if joint_limits is None:
        joint_limits = getattr(robot.data, "joint_pos_limits", None)
    if joint_limits is None:
        return None, None
    if joint_limits.ndim == 2:
        joint_limits = joint_limits.unsqueeze(0)
    return joint_limits[..., 0], joint_limits[..., 1]


def reset_robot(robot: Articulation, origins: torch.Tensor, hold_base_height: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += origins.to(device=root_state.device)
    root_state[:, 2] = hold_base_height
    locked_root_pose = root_state[:, :7].clone()
    locked_root_vel = torch.zeros_like(root_state[:, 7:])
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()

    robot.write_root_pose_to_sim(locked_root_pose)
    robot.write_root_velocity_to_sim(locked_root_vel)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()
    return locked_root_pose, locked_root_vel, joint_pos, joint_vel


def step_with_targets(
    sim: SimulationContext,
    robot: Articulation,
    target: torch.Tensor,
    locked_root_pose: torch.Tensor,
    locked_root_vel: torch.Tensor,
    num_steps: int,
) -> None:
    sim_dt = sim.get_physics_dt()
    for _ in range(num_steps):
        robot.write_root_pose_to_sim(locked_root_pose)
        robot.write_root_velocity_to_sim(locked_root_vel)
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)


def print_joint_snapshot(robot: Articulation, joint_name: str, joint_id: int, target: float) -> None:
    lower, upper = get_joint_limits(robot)
    measured = robot.data.joint_pos[0, joint_id].item()
    if lower is not None and upper is not None:
        limit_text = f"[{lower[0, joint_id].item():+.4f}, {upper[0, joint_id].item():+.4f}]"
    else:
        limit_text = "unavailable"
    print(
        f"[MastiffJointDiag] joint={joint_name} target={target:+.4f} measured={measured:+.4f} limits={limit_text}"
    )


def print_group_snapshot(robot: Articulation) -> None:
    groups = [
        "HAA_FRONT_LEFT", "HAA_FRONT_RIGHT", "HAA_REAR_LEFT", "HAA_REAR_RIGHT",
        "HFE_FRONT_LEFT", "HFE_FRONT_RIGHT", "HFE_REAR_LEFT", "HFE_REAR_RIGHT",
        "KFE_FRONT_LEFT", "KFE_FRONT_RIGHT", "KFE_REAR_LEFT", "KFE_REAR_RIGHT",
    ]
    values = []
    for name in groups:
        joint_ids, _ = robot.find_joints([name])
        if len(joint_ids) == 0:
            continue
        values.append(f"{name}={robot.data.joint_pos[0, joint_ids[0]].item():+.4f}")
    print("[MastiffJointDiag][Snapshot] " + ", ".join(values))


def run_diagnostic(sim: SimulationContext, robot: Articulation, origins: torch.Tensor) -> None:
    locked_root_pose, locked_root_vel, zero_joint_pos, _ = reset_robot(robot, origins, args_cli.hold_base_height)

    print("[MastiffJointDiag] joint order:", robot.joint_names)
    lower, upper = get_joint_limits(robot)
    if lower is not None and upper is not None:
        print("[MastiffJointDiag] joint limits:")
        for joint_id, joint_name in enumerate(robot.joint_names):
            print(
                f"  {joint_name}: [{lower[0, joint_id].item():+.4f}, {upper[0, joint_id].item():+.4f}]"
            )

    for joint_name in args_cli.joint_names:
        joint_ids, _ = robot.find_joints([joint_name])
        if len(joint_ids) == 0:
            print(f"[MastiffJointDiag] joint not found: {joint_name}")
            continue
        joint_id = joint_ids[0]
        print(f"[MastiffJointDiag] testing {joint_name} (joint_id={joint_id})")

        zero_target = zero_joint_pos.clone()
        step_with_targets(sim, robot, zero_target, locked_root_pose, locked_root_vel, args_cli.pause_steps)
        print_joint_snapshot(robot, joint_name, joint_id, 0.0)

        for angle in args_cli.angles:
            target = zero_joint_pos.clone()
            target[:, joint_id] = angle
            step_with_targets(sim, robot, target, locked_root_pose, locked_root_vel, args_cli.settle_steps)
            print_joint_snapshot(robot, joint_name, joint_id, angle)
            print_group_snapshot(robot)
            if not simulation_app.is_running():
                return

        step_with_targets(sim, robot, zero_target, locked_root_pose, locked_root_vel, args_cli.pause_steps)

    while simulation_app.is_running():
        step_with_targets(sim, robot, zero_joint_pos, locked_root_pose, locked_root_vel, 1)


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.8, 0.0, 1.8], [0.0, 0.0, 0.3])

    robot, origins = design_scene()
    origins = origins.to(device=sim.device)

    sim.reset()
    print("[MastiffJointDiag] setup complete")
    run_diagnostic(sim, robot, origins)


if __name__ == "__main__":
    main()
    simulation_app.close()
