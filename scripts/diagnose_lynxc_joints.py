import argparse
from pathlib import Path

import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Diagnose Lynxc joint zero, sign, and leg-link motion behavior.")
parser.add_argument("--angles", type=float, nargs="+", default=[0.0, 0.3, -0.3, 0.8, -0.8], help="Joint angles in radians to test for each joint.")
parser.add_argument(
    "--joint_names",
    type=str,
    nargs="+",
    default=[
        "FL0",
        "FL1",
        "FL2",
        "FR0",
        "FR1",
        "FR2",
        "RL0",
        "RL1",
        "RL2",
        "RR0",
        "RR1",
        "RR2",
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
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim import SimulationContext


_PROJECT_PATH = Path(__file__).resolve().parents[1]
_LYNXC_USD_PATH = _PROJECT_PATH / "assets" / "LynxC.usd"

LEG_ORDER = ("FL", "FR", "RL", "RR")
LEG_JOINTS = {
    "FL": ("FL0", "FL1", "FL2"),
    "FR": ("FR0", "FR1", "FR2"),
    "RL": ("RL0", "RL1", "RL2"),
    "RR": ("RR0", "RR1", "RR2"),
}
LEG_BODY_NAMES = {
    "FL": ("Hip", "Thigh_02", "Foot_02"),
    "FR": ("Hip_03", "Thigh", "Calf_02"),
    "RL": ("Hip_02", "Thigh_03", "Calf"),
    "RR": ("Hip_01", "Thigh_01", "Calf_03"),
}
JOINT_TO_LEG = {joint_name: leg_name for leg_name, joint_names in LEG_JOINTS.items() for joint_name in joint_names}


LYNXC_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(_LYNXC_USD_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.3),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
        rot=(0.7071068, 0.0, 0.7071068, 0.0)
    ),
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=2000.0,
            velocity_limit=2000.94,
            stiffness=1e5,
            damping=200.0,
        ),
    },
)


def design_scene() -> tuple[Articulation, torch.Tensor]:
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    origin = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origin[0].tolist())

    robot_cfg = LYNXC_CFG.copy()
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


# def step_with_direct_joint_state(
#     sim: SimulationContext,
#     robot: Articulation,
#     joint_pos: torch.Tensor,
#     joint_vel: torch.Tensor,
#     locked_root_pose: torch.Tensor,
#     locked_root_vel: torch.Tensor,
#     num_steps: int,
# ) -> None:
#     sim_dt = sim.get_physics_dt()
#     for _ in range(num_steps):
#         robot.write_root_pose_to_sim(locked_root_pose)
#         robot.write_root_velocity_to_sim(locked_root_vel)
#         robot.write_joint_state_to_sim(joint_pos, joint_vel)
#         robot.write_data_to_sim()
#         sim.step()
#         robot.update(sim_dt)


def step_with_effort_targets(
    sim: SimulationContext,
    robot: Articulation,
    efforts: torch.Tensor,
    locked_root_pose: torch.Tensor,
    locked_root_vel: torch.Tensor,
    num_steps: int,
) -> None:
    sim_dt = sim.get_physics_dt()
    zero_pos_target = torch.zeros_like(robot.data.joint_pos)
    zero_vel_target = torch.zeros_like(robot.data.joint_vel)
    for _ in range(num_steps):
        robot.write_root_pose_to_sim(locked_root_pose)
        robot.write_root_velocity_to_sim(locked_root_vel)
        robot.set_joint_position_target(zero_pos_target)
        robot.set_joint_velocity_target(zero_vel_target)
        robot.set_joint_effort_target(efforts)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)


def resolve_body_ids(robot: Articulation, body_names: tuple[str, ...]) -> dict[str, int]:
    available_names = list(robot.data.body_names)
    body_name_to_idx = {name: idx for idx, name in enumerate(available_names)}
    resolved: dict[str, int] = {}
    for body_name in body_names:
        if body_name not in body_name_to_idx:
            raise RuntimeError(f"Body '{body_name}' not found. Available bodies: {available_names}")
        resolved[body_name] = body_name_to_idx[body_name]
    return resolved


def print_joint_snapshot(robot: Articulation, joint_name: str, joint_id: int, target: float) -> None:
    lower, upper = get_joint_limits(robot)
    measured = robot.data.joint_pos[0, joint_id].item()
    if lower is not None and upper is not None:
        limit_text = f"[{lower[0, joint_id].item():+.4f}, {upper[0, joint_id].item():+.4f}]"
    else:
        limit_text = "unavailable"
    print(
        f"[LynxcJointDiag] joint={joint_name} target={target:+.4f} measured={measured:+.4f} limits={limit_text}"
    )


def print_joint_group_snapshot(robot: Articulation) -> None:
    values = []
    for leg_name in LEG_ORDER:
        for joint_name in LEG_JOINTS[leg_name]:
            joint_ids, _ = robot.find_joints([joint_name])
            if len(joint_ids) == 0:
                continue
            values.append(f"{joint_name}={robot.data.joint_pos[0, joint_ids[0]].item():+.4f}")
    print("[LynxcJointDiag][Snapshot] " + ", ".join(values))


def print_actuator_snapshot(robot: Articulation, joint_id: int, joint_name: str) -> None:
    pos_target = getattr(robot.data, "joint_pos_target", None)
    vel_target = getattr(robot.data, "joint_vel_target", None)
    effort_target = getattr(robot.data, "joint_effort_target", None)
    stiffness = getattr(robot.data, "joint_stiffness", None)
    damping = getattr(robot.data, "joint_damping", None)
    effort_limit = getattr(robot.data, "joint_effort_limits", None)
    vel_limit = getattr(robot.data, "joint_vel_limits", None)
    armature = getattr(robot.data, "joint_armature", None)
    friction = getattr(robot.data, "joint_friction_coeff", None)
    parts = [f"joint={joint_name}"]
    if pos_target is not None:
        parts.append(f"pos_target={pos_target[0, joint_id].item():+.4f}")
    if vel_target is not None:
        parts.append(f"vel_target={vel_target[0, joint_id].item():+.4f}")
    if effort_target is not None:
        parts.append(f"effort_target={effort_target[0, joint_id].item():+.4f}")
    if stiffness is not None:
        parts.append(f"stiffness={stiffness[0, joint_id].item():+.4f}")
    if damping is not None:
        parts.append(f"damping={damping[0, joint_id].item():+.4f}")
    if effort_limit is not None:
        parts.append(f"effort_limit={effort_limit[0, joint_id].item():+.4f}")
    if vel_limit is not None:
        parts.append(f"vel_limit={vel_limit[0, joint_id].item():+.4f}")
    if armature is not None:
        parts.append(f"armature={armature[0, joint_id].item():+.4f}")
    if friction is not None:
        parts.append(f"friction={friction[0, joint_id].item():+.4f}")
    print("[LynxcJointDiag][Actuator] " + ", ".join(parts))


def print_leg_link_snapshot(robot: Articulation, leg_name: str, body_ids: dict[str, int]) -> None:
    root_pos_w = robot.data.root_pos_w[0]
    pieces = []
    for body_name in LEG_BODY_NAMES[leg_name]:
        body_pos_w = robot.data.body_pos_w[0, body_ids[body_name]]
        rel = body_pos_w - root_pos_w
        pieces.append(f"{body_name}=({rel[0].item():+.4f},{rel[1].item():+.4f},{rel[2].item():+.4f})")
    print(f"[LynxcJointDiag][{leg_name}][Links] " + ", ".join(pieces))


def run_diagnostic(sim: SimulationContext, robot: Articulation, origins: torch.Tensor) -> None:
    locked_root_pose, locked_root_vel, zero_joint_pos, _ = reset_robot(robot, origins, args_cli.hold_base_height)
    tracked_body_names = tuple(body_name for leg_name in LEG_ORDER for body_name in LEG_BODY_NAMES[leg_name])
    body_ids = resolve_body_ids(robot, tracked_body_names)

    print("[LynxcJointDiag] usd:", _LYNXC_USD_PATH)
    print("[LynxcJointDiag] joint order:", robot.joint_names)
    print("[LynxcJointDiag] body order:", robot.data.body_names)
    for leg_name in LEG_ORDER:
        print(
            f"[LynxcJointDiag][BodyMap][{leg_name}] "
            f"hip={LEG_BODY_NAMES[leg_name][0]} thigh={LEG_BODY_NAMES[leg_name][1]} "
            f"calf={LEG_BODY_NAMES[leg_name][2]}"
        )

    lower, upper = get_joint_limits(robot)
    if lower is not None and upper is not None:
        print("[LynxcJointDiag] joint limits:")
        for joint_id, joint_name in enumerate(robot.joint_names):
            print(
                f"  {joint_name}: [{lower[0, joint_id].item():+.4f}, {upper[0, joint_id].item():+.4f}]"
            )

    for joint_name in args_cli.joint_names:
        joint_ids, _ = robot.find_joints([joint_name])
        if len(joint_ids) == 0:
            print(f"[LynxcJointDiag] joint not found: {joint_name}")
            continue

        joint_id = int(joint_ids[0])
        leg_name = JOINT_TO_LEG.get(joint_name, "UNKNOWN")
        print(f"[LynxcJointDiag] testing {joint_name} (joint_id={joint_id}, leg={leg_name})")

        zero_target = zero_joint_pos.clone()
        step_with_targets(sim, robot, zero_target, locked_root_pose, locked_root_vel, args_cli.pause_steps)
        print_joint_snapshot(robot, joint_name, joint_id, 0.0)
        print_actuator_snapshot(robot, joint_id, joint_name)
        if leg_name in LEG_BODY_NAMES:
            print_leg_link_snapshot(robot, leg_name, body_ids)

        for angle in args_cli.angles:
            target = zero_joint_pos.clone()
            target[:, joint_id] = angle
            step_with_targets(sim, robot, target, locked_root_pose, locked_root_vel, args_cli.settle_steps)
            print(f"[LynxcJointDiag] actuator-path target applied for {joint_name}")
            print_joint_snapshot(robot, joint_name, joint_id, angle)
            print_actuator_snapshot(robot, joint_id, joint_name)
            print_joint_group_snapshot(robot)
            if leg_name in LEG_BODY_NAMES:
                print_leg_link_snapshot(robot, leg_name, body_ids)

            effort_target = torch.zeros_like(zero_target)
            effort_target[:, joint_id] = 100.0
            step_with_effort_targets(sim, robot, effort_target, locked_root_pose, locked_root_vel, args_cli.settle_steps)
            print(f"[LynxcJointDiag] effort-path target applied for {joint_name}")
            print_joint_snapshot(robot, joint_name, joint_id, angle)
            print_actuator_snapshot(robot, joint_id, joint_name)
            print_joint_group_snapshot(robot)
            if leg_name in LEG_BODY_NAMES:
                print_leg_link_snapshot(robot, leg_name, body_ids)

            # direct_target = zero_joint_pos.clone()
            # direct_target[:, joint_id] = angle
            # direct_velocity = torch.zeros_like(direct_target)
            # step_with_direct_joint_state(sim, robot, direct_target, direct_velocity, locked_root_pose, locked_root_vel, 2)
            # print(f"[LynxcJointDiag] direct-state write applied for {joint_name}")
            # print_joint_snapshot(robot, joint_name, joint_id, angle)
            # print_joint_group_snapshot(robot)
            # if leg_name in LEG_BODY_NAMES:
            #     print_leg_link_snapshot(robot, leg_name, body_ids)

            step_with_targets(sim, robot, zero_target, locked_root_pose, locked_root_vel, args_cli.pause_steps)
            if not simulation_app.is_running():
                return

    while simulation_app.is_running():
        step_with_targets(sim, robot, zero_joint_pos, locked_root_pose, locked_root_vel, 1)


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.8, 0.0, 1.8], [0.0, 0.0, 0.3])

    robot, origins = design_scene()
    origins = origins.to(device=sim.device)

    sim.reset()
    print("[LynxcJointDiag] setup complete")
    run_diagnostic(sim, robot, origins)


if __name__ == "__main__":
    main()
    simulation_app.close()
