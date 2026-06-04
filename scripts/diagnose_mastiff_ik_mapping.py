import argparse
import importlib.util
import math
from pathlib import Path
import sys

ISAACLAB_SOURCE_PATH = Path.home() / "IsaacLab" / "source" / "isaaclab"
if ISAACLAB_SOURCE_PATH.exists():
    sys.path.insert(0, str(ISAACLAB_SOURCE_PATH))

import torch

from isaaclab.app import AppLauncher


LEG_ORDER = ("FL", "FR", "RL", "RR")
LEG_CONFIGS = {
    "FL": {"side_sign": 1.0, "joints": ("HAA_FRONT_LEFT", "HFE_FRONT_LEFT", "KFE_FRONT_LEFT")},
    "FR": {"side_sign": -1.0, "joints": ("HAA_FRONT_RIGHT", "HFE_FRONT_RIGHT", "KFE_FRONT_RIGHT")},
    "RL": {"side_sign": 1.0, "joints": ("HAA_REAR_LEFT", "HFE_REAR_LEFT", "KFE_REAR_LEFT")},
    "RR": {"side_sign": -1.0, "joints": ("HAA_REAR_RIGHT", "HFE_REAR_RIGHT", "KFE_REAR_RIGHT")},
}


LEG_JOINT_SIGNS = {
    "FL": torch.tensor([+1.0, +1.0, -1.0], dtype=torch.float64),
    "FR": torch.tensor([+1.0, -1.0, -1.0], dtype=torch.float64),
    "RL": torch.tensor([-1.0, +1.0, -1.0], dtype=torch.float64),
    "RR": torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float64),
}

# Mastiff USD body names do not encode leg identity consistently by suffix.
# These explicit maps follow the asset's base+joint child ordering from assets/Mastiff_CFG.py.
DEFAULT_HIP_BODY_NAMES = {
    "FL": "Hip_v1_02",
    "FR": "Hip_v1",
    "RL": "Hip_v1_01",
    "RR": "Hip_v1_03",
}
DEFAULT_FOOT_BODY_NAMES = {
    "FL": "Foot_v1",
    "FR": "Foot_v1_03",
    "RL": "Foot_v1_01",
    "RR": "Foot_v1_02",
}
DEFAULT_THIGH_BODY_NAMES = {
    "FL": "ThighL_v1_01",
    "FR": "ThighRstp_v1",
    "RL": "ThighL_v1",
    "RR": "ThighRstp_v1_01",
}


parser = argparse.ArgumentParser(
    description="Diagnose Mastiff zero-angle / standing-pose IK mapping by commanding simulator joint targets."
)
parser.add_argument("--femur-zero-angle-global-deg", type=float, default=-150.0)
parser.add_argument("--tibia-zero-angle-relative-deg", type=float, default=15.0)
parser.add_argument("--standing-haa-deg", type=float, default=0.0)
parser.add_argument("--standing-hfe-deg", type=float, default=-10.0)
parser.add_argument("--standing-kfe-deg", type=float, default=40.0)
parser.add_argument("--standing-input-space", choices=("sim", "planner"), default="planner")
parser.add_argument("--hold-base-height", type=float, default=0.45)
parser.add_argument("--settle-steps", type=int, default=180)
parser.add_argument("--pause-steps", type=int, default=60)
parser.add_argument(
    "--target-mode",
    choices=("sequence", "standing_input", "ik_roundtrip", "zero", "gait_cycle"),
    default="sequence",
    help="Which pose to command into the simulator.",
)
parser.add_argument("--gait-step-x", type=float, default=0.15, help="Body-frame gait step displacement in +X, meters.")
parser.add_argument("--gait-step-y", type=float, default=0.0, help="Body-frame gait step displacement in +Y, meters.")
parser.add_argument("--gait-step-height", type=float, default=0.10, help="Swing foot lift for gait-cycle playback, meters.")
parser.add_argument(
    "--gait-cycle-frames",
    type=int,
    default=120,
    help="Number of frames used to sample one full gait cycle.",
)
parser.add_argument(
    "--gait-frame-hold-steps",
    type=int,
    default=2,
    help="Simulator steps to hold each sampled gait frame before advancing to the next one.",
)
parser.add_argument(
    "--gait-log-every-frame",
    action="store_true",
    default=False,
    help="Print per-frame gait diagnostics including target, planner, sim, and measured joint values.",
)
parser.add_argument(
    "--gait-log-frame-stride",
    type=int,
    default=1,
    help="Only print every Nth frame when --gait-log-every-frame is enabled.",
)
for leg_name in LEG_ORDER:
    parser.add_argument(f"--{leg_name.lower()}-hip-body", type=str, default=DEFAULT_HIP_BODY_NAMES[leg_name])
    parser.add_argument(f"--{leg_name.lower()}-thigh-body", type=str, default=DEFAULT_THIGH_BODY_NAMES[leg_name])
    parser.add_argument(f"--{leg_name.lower()}-foot-body", type=str, default=DEFAULT_FOOT_BODY_NAMES[leg_name])
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext


PROJECT_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_PATH))
from assets.Mastiff_CFG import Mastiff_CONFIG as ROBOT_CFG


def load_generator_module():
    module_path = PROJECT_PATH / "tasks" / "mdp" / "quadruped_gait_generator.py"
    spec = importlib.util.spec_from_file_location("quadruped_gait_generator_diag", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load gait generator module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def sim_to_planner(joint_targets_sim: torch.Tensor, joint_signs: torch.Tensor) -> torch.Tensor:
    return joint_targets_sim * joint_signs


def planner_to_sim(joint_targets_planner: torch.Tensor, joint_signs: torch.Tensor) -> torch.Tensor:
    return joint_targets_planner * joint_signs


def leg_joint_sign_matrix() -> torch.Tensor:
    return torch.stack([LEG_JOINT_SIGNS[leg_name] for leg_name in LEG_ORDER], dim=0)


def measured_leg_joint_positions(robot: Articulation, joint_id_map: dict[str, tuple[int, int, int]]) -> torch.Tensor:
    measured = torch.zeros((len(LEG_ORDER), 3), dtype=torch.float64)
    for leg_index, leg_name in enumerate(LEG_ORDER):
        haa_id, hfe_id, kfe_id = joint_id_map[leg_name]
        measured[leg_index, 0] = float(robot.data.joint_pos[0, haa_id].item())
        measured[leg_index, 1] = float(robot.data.joint_pos[0, hfe_id].item())
        measured[leg_index, 2] = float(robot.data.joint_pos[0, kfe_id].item())
    return measured


def print_joint_group_snapshot(robot: Articulation, joint_id_map: dict[str, tuple[int, int, int]], label: str) -> None:
    measured = measured_leg_joint_positions(robot, joint_id_map)
    values = []
    for leg_index, leg_name in enumerate(LEG_ORDER):
        values.append(
            f"{leg_name}=(HAA={measured[leg_index, 0].item():+.4f},"
            f"HFE={measured[leg_index, 1].item():+.4f},"
            f"KFE={measured[leg_index, 2].item():+.4f})"
        )
    print(f"[MastiffIKDiag][Measured][{label}] " + "; ".join(values))


def print_leg_limit_snapshot(robot: Articulation, joint_id_map: dict[str, tuple[int, int, int]]) -> None:
    lower, upper = get_joint_limits(robot)
    if lower is None or upper is None:
        print("[MastiffIKDiag] joint limits unavailable")
        return
    for leg_name in LEG_ORDER:
        haa_id, hfe_id, kfe_id = joint_id_map[leg_name]
        print(
            f"[MastiffIKDiag][Limits][{leg_name}] "
            f"HAA=[{lower[0, haa_id].item():+.4f},{upper[0, haa_id].item():+.4f}] "
            f"HFE=[{lower[0, hfe_id].item():+.4f},{upper[0, hfe_id].item():+.4f}] "
            f"KFE=[{lower[0, kfe_id].item():+.4f},{upper[0, kfe_id].item():+.4f}]"
        )


def resolve_leg_body_ids(robot: Articulation, explicit_name_map: dict[str, str]) -> dict[str, int]:
    body_names = list(robot.data.body_names)
    body_name_to_idx = {name: idx for idx, name in enumerate(body_names)}
    resolved: dict[str, int] = {}
    for leg_name, body_name in explicit_name_map.items():
        if body_name not in body_name_to_idx:
            raise RuntimeError(
                f"Body '{body_name}' for leg {leg_name} not found. Available bodies: {body_names}"
            )
        resolved[leg_name] = body_name_to_idx[body_name]
    return resolved


def print_leg_body_mapping(
    robot: Articulation,
    foot_body_ids: dict[str, int],
    thigh_body_ids: dict[str, int],
    hip_body_ids: dict[str, int],
) -> None:
    for leg_name in LEG_ORDER:
        foot_name = robot.data.body_names[foot_body_ids[leg_name]]
        thigh_name = robot.data.body_names[thigh_body_ids[leg_name]]
        hip_name = robot.data.body_names[hip_body_ids[leg_name]]
        print(f"[MastiffIKDiag][BodyMap][{leg_name}] hip={hip_name} thigh={thigh_name} foot={foot_name}")


def actual_foot_positions_relative_to_hip(robot: Articulation, foot_body_ids: dict[str, int], hip_body_ids: dict[str, int]) -> torch.Tensor:
    body_pos_w = robot.data.body_pos_w[0].to(dtype=torch.float64)
    root_quat_w = robot.data.root_quat_w[0].to(dtype=torch.float64)
    rel_w = torch.zeros((len(LEG_ORDER), 3), dtype=torch.float64, device=body_pos_w.device)
    for leg_index, leg_name in enumerate(LEG_ORDER):
        rel_w[leg_index] = body_pos_w[foot_body_ids[leg_name]] - body_pos_w[hip_body_ids[leg_name]]
    root_quat_batch = root_quat_w.unsqueeze(0).repeat(len(LEG_ORDER), 1)
    rel_b = math_utils.quat_apply_inverse(root_quat_batch, rel_w)
    return rel_b.detach().cpu()


def build_pose_bundle() -> dict[str, torch.Tensor | object]:
    module = load_generator_module()
    geometry = module.QuadrupedGeometry(
        femur_zero_angle_global=math.radians(args_cli.femur_zero_angle_global_deg),
        tibia_zero_angle_relative=math.radians(args_cli.tibia_zero_angle_relative_deg),
    )
    generator = module.QuadrupedGaitGenerator(
        geometry=geometry,
        leg_order=LEG_ORDER,
        side_signs=[LEG_CONFIGS[leg_name]["side_sign"] for leg_name in LEG_ORDER],
        device="cpu",
        dtype=torch.float64,
    )

    joint_signs = leg_joint_sign_matrix()
    standing_input = torch.tensor(
        [
            math.radians(args_cli.standing_haa_deg),
            math.radians(args_cli.standing_hfe_deg),
            math.radians(args_cli.standing_kfe_deg),
        ],
        dtype=torch.float64,
    )
    if args_cli.standing_input_space == "sim":
        standing_sim = standing_input.unsqueeze(0).repeat(len(LEG_ORDER), 1)
        standing_planner_legs = sim_to_planner(standing_sim, joint_signs)
    else:
        standing_planner_legs = standing_input.unsqueeze(0).repeat(len(LEG_ORDER), 1)
        standing_sim = planner_to_sim(standing_planner_legs, joint_signs)

    side_signs = generator.side_signs.to(dtype=torch.float64)
    zero_home = generator.home_foot_positions(side_signs).to(dtype=torch.float64)
    standing_home = generator.nominal_standing_foot_positions(standing_planner_legs, side_signs)
    ik_planner, valid_ik = generator.solve_ik(standing_home, side_signs)
    ik_sim = planner_to_sim(ik_planner, joint_signs)

    return {
        "generator": generator,
        "geometry": geometry,
        "side_signs": side_signs,
        "joint_signs": joint_signs,
        "standing_input": standing_input,
        "standing_planner": standing_planner_legs,
        "standing_sim": standing_sim,
        "standing_home": standing_home,
        "zero_home": zero_home,
        "ik_planner": ik_planner,
        "ik_sim": ik_sim,
        "valid_ik": valid_ik,
    }


def build_sim_target(zero_joint_pos: torch.Tensor, joint_id_map: dict[str, tuple[int, int, int]], leg_joint_targets_sim: torch.Tensor) -> torch.Tensor:
    target = zero_joint_pos.clone()
    for leg_index, leg_name in enumerate(LEG_ORDER):
        haa_id, hfe_id, kfe_id = joint_id_map[leg_name]
        target[:, haa_id] = float(leg_joint_targets_sim[leg_index, 0].item())
        target[:, hfe_id] = float(leg_joint_targets_sim[leg_index, 1].item())
        target[:, kfe_id] = float(leg_joint_targets_sim[leg_index, 2].item())
    return target


def build_gait_cycle_bundle(bundle: dict[str, torch.Tensor | object]) -> dict[str, torch.Tensor]:
    frame_count = max(int(args_cli.gait_cycle_frames), 2)
    hold_steps = max(int(args_cli.gait_frame_hold_steps), 1)
    generator = bundle["generator"]
    side_signs = bundle["side_signs"]
    joint_signs = bundle["joint_signs"]
    standing_home = bundle["standing_home"]
    standing_planner = bundle["standing_planner"]

    phase_scalars = torch.linspace(0.0, 2.0 * math.pi, steps=frame_count + 1, dtype=torch.float64)[:-1]
    phase_offsets = generator.phase_offsets.to(dtype=torch.float64)
    phases = (phase_scalars.unsqueeze(-1) + phase_offsets.unsqueeze(0)) % (2.0 * math.pi)

    step_vector = torch.tensor([args_cli.gait_step_x, args_cli.gait_step_y], dtype=torch.float64)
    step_vectors = step_vector.view(1, 1, 2).repeat(frame_count, len(LEG_ORDER), 1)
    step_heights = torch.full((frame_count, len(LEG_ORDER)), float(args_cli.gait_step_height), dtype=torch.float64)
    ground_heights = torch.zeros((frame_count, len(LEG_ORDER)), dtype=torch.float64)
    home_positions = standing_home.unsqueeze(0).expand(frame_count, -1, -1)

    gait_planner, gait_targets, gait_valid_ik = generator.joint_targets_from_phase(
        phases,
        step_vectors,
        step_heights,
        ground_heights,
        side_signs=side_signs,
        home_positions=home_positions,
    )
    gait_planner = torch.where(gait_valid_ik.unsqueeze(-1), gait_planner, standing_planner.unsqueeze(0))
    gait_sim = planner_to_sim(gait_planner, joint_signs.unsqueeze(0))
    return {
        "phases": phases,
        "foot_targets": gait_targets,
        "planner_targets": gait_planner,
        "sim_targets": gait_sim,
        "valid_ik": gait_valid_ik,
        "frame_count": torch.tensor(frame_count, dtype=torch.int64),
        "hold_steps": torch.tensor(hold_steps, dtype=torch.int64),
    }


def print_pose_report(bundle: dict[str, torch.Tensor | object]) -> None:
    print("[MastiffIKDiag] configuration")
    print(
        f"  zero_angles_deg=(femur={args_cli.femur_zero_angle_global_deg:+.3f}, "
        f"tibia={args_cli.tibia_zero_angle_relative_deg:+.3f})"
    )
    print(
        f"  standing_input_deg=(HAA={args_cli.standing_haa_deg:+.3f}, "
        f"HFE={args_cli.standing_hfe_deg:+.3f}, KFE={args_cli.standing_kfe_deg:+.3f}) "
        f"space={args_cli.standing_input_space}"
    )
    for leg_index, leg_name in enumerate(LEG_ORDER):
        joint_signs = bundle["joint_signs"][leg_index].tolist()
        standing_planner_deg = torch.rad2deg(bundle["standing_planner"][leg_index]).tolist()
        standing_sim_deg = torch.rad2deg(bundle["standing_sim"][leg_index]).tolist()
        zero_home_mm = (bundle["zero_home"][leg_index] * 1000.0).tolist()
        standing_home_mm = (bundle["standing_home"][leg_index] * 1000.0).tolist()
        ik_planner_deg = torch.rad2deg(bundle["ik_planner"][leg_index]).tolist()
        ik_sim_deg = torch.rad2deg(bundle["ik_sim"][leg_index]).tolist()
        print(
            f"[MastiffIKDiag][{leg_name}] joint_signs=(HAA={joint_signs[0]:+.1f},HFE={joint_signs[1]:+.1f},KFE={joint_signs[2]:+.1f}) "
            f"standing_planner_deg=(HAA={standing_planner_deg[0]:+.3f},HFE={standing_planner_deg[1]:+.3f},KFE={standing_planner_deg[2]:+.3f}) "
            f"standing_sim_deg=(HAA={standing_sim_deg[0]:+.3f},HFE={standing_sim_deg[1]:+.3f},KFE={standing_sim_deg[2]:+.3f})"
        )
        print(
            f"[MastiffIKDiag][{leg_name}] zero_home_mm=({zero_home_mm[0]:+.1f},{zero_home_mm[1]:+.1f},{zero_home_mm[2]:+.1f}) "
            f"standing_home_mm=({standing_home_mm[0]:+.1f},{standing_home_mm[1]:+.1f},{standing_home_mm[2]:+.1f}) "
            f"ik_valid={'Y' if bool(bundle['valid_ik'][leg_index].item()) else 'N'}"
        )
        print(
            f"[MastiffIKDiag][{leg_name}] ik_planner_deg=(HAA={ik_planner_deg[0]:+.3f},HFE={ik_planner_deg[1]:+.3f},KFE={ik_planner_deg[2]:+.3f}) "
            f"ik_sim_deg=(HAA={ik_sim_deg[0]:+.3f},HFE={ik_sim_deg[1]:+.3f},KFE={ik_sim_deg[2]:+.3f})"
        )


def print_gait_cycle_report(gait_bundle: dict[str, torch.Tensor]) -> None:
    frame_count = int(gait_bundle["frame_count"].item())
    valid_counts = gait_bundle["valid_ik"].sum(dim=0).tolist()
    print(
        f"[MastiffIKDiag][GaitCycle] step_xy_m=({args_cli.gait_step_x:+.4f},{args_cli.gait_step_y:+.4f}) "
        f"step_height_m={args_cli.gait_step_height:+.4f} frames={frame_count} "
        f"hold_steps={int(gait_bundle['hold_steps'].item())}"
    )
    for leg_index, leg_name in enumerate(LEG_ORDER):
        first_target = gait_bundle["foot_targets"][0, leg_index] * 1000.0
        first_planner_deg = torch.rad2deg(gait_bundle["planner_targets"][0, leg_index]).tolist()
        first_sim_deg = torch.rad2deg(gait_bundle["sim_targets"][0, leg_index]).tolist()
        print(
            f"[MastiffIKDiag][GaitCycle][{leg_name}] "
            f"valid_frames={int(valid_counts[leg_index])}/{frame_count} "
            f"frame0_target_mm=({first_target[0].item():+.1f},{first_target[1].item():+.1f},{first_target[2].item():+.1f}) "
            f"frame0_planner_deg=(HAA={first_planner_deg[0]:+.3f},HFE={first_planner_deg[1]:+.3f},KFE={first_planner_deg[2]:+.3f}) "
            f"frame0_sim_deg=(HAA={first_sim_deg[0]:+.3f},HFE={first_sim_deg[1]:+.3f},KFE={first_sim_deg[2]:+.3f})"
        )


def measured_fk_foot_positions(
    measured_sim: torch.Tensor,
    joint_signs: torch.Tensor,
    generator,
    side_signs: torch.Tensor,
) -> torch.Tensor:
    measured_planner = sim_to_planner(measured_sim.to(dtype=torch.float64), joint_signs)
    fk_points = generator.forward_kinematics(measured_planner, side_signs.to(dtype=torch.float64))
    return fk_points[..., -1, :].detach().cpu()


def print_gait_cycle_frame_log(
    frame_index: int,
    gait_bundle: dict[str, torch.Tensor],
    measured_sim: torch.Tensor,
    actual_foot_rel: torch.Tensor,
    measured_fk_mm: torch.Tensor,
) -> None:
    phase_deg = torch.rad2deg(gait_bundle["phases"][frame_index])
    foot_targets_mm = gait_bundle["foot_targets"][frame_index] * 1000.0
    planner_deg = torch.rad2deg(gait_bundle["planner_targets"][frame_index])
    sim_deg = torch.rad2deg(gait_bundle["sim_targets"][frame_index])
    measured_deg = torch.rad2deg(measured_sim)
    actual_foot_mm = actual_foot_rel * 1000.0
    actual_error_mm = actual_foot_mm - foot_targets_mm
    measured_fk_error_mm = measured_fk_mm - foot_targets_mm
    valid_ik = gait_bundle["valid_ik"][frame_index]

    print(f"[MastiffIKDiag][Frame {frame_index:03d}]")
    for leg_index, leg_name in enumerate(LEG_ORDER):
        print(
            f"[MastiffIKDiag][Frame {frame_index:03d}][{leg_name}] "
            f"phase_deg={phase_deg[leg_index].item():6.1f} "
            f"ik_valid={'Y' if bool(valid_ik[leg_index].item()) else 'N'} "
            f"target_mm=(x={foot_targets_mm[leg_index, 0].item():+.1f},y={foot_targets_mm[leg_index, 1].item():+.1f},z={foot_targets_mm[leg_index, 2].item():+.1f}) "
            f"actual_mm=(x={actual_foot_mm[leg_index, 0].item():+.1f},y={actual_foot_mm[leg_index, 1].item():+.1f},z={actual_foot_mm[leg_index, 2].item():+.1f}) "
            f"actual_err_mm=(x={actual_error_mm[leg_index, 0].item():+.1f},y={actual_error_mm[leg_index, 1].item():+.1f},z={actual_error_mm[leg_index, 2].item():+.1f}) "
            f"fk_mm=(x={measured_fk_mm[leg_index, 0].item():+.1f},y={measured_fk_mm[leg_index, 1].item():+.1f},z={measured_fk_mm[leg_index, 2].item():+.1f}) "
            f"fk_err_mm=(x={measured_fk_error_mm[leg_index, 0].item():+.1f},y={measured_fk_error_mm[leg_index, 1].item():+.1f},z={measured_fk_error_mm[leg_index, 2].item():+.1f}) "
            f"planner_deg=(HAA={planner_deg[leg_index, 0].item():+.3f},HFE={planner_deg[leg_index, 1].item():+.3f},KFE={planner_deg[leg_index, 2].item():+.3f}) "
            f"sim_deg=(HAA={sim_deg[leg_index, 0].item():+.3f},HFE={sim_deg[leg_index, 1].item():+.3f},KFE={sim_deg[leg_index, 2].item():+.3f}) "
            f"measured_deg=(HAA={measured_deg[leg_index, 0].item():+.3f},HFE={measured_deg[leg_index, 1].item():+.3f},KFE={measured_deg[leg_index, 2].item():+.3f})"
        )


def run_pose(
    sim: SimulationContext,
    robot: Articulation,
    joint_id_map: dict[str, tuple[int, int, int]],
    zero_joint_pos: torch.Tensor,
    locked_root_pose: torch.Tensor,
    locked_root_vel: torch.Tensor,
    label: str,
    leg_joint_targets_sim: torch.Tensor,
) -> None:
    target = build_sim_target(zero_joint_pos, joint_id_map, leg_joint_targets_sim)
    print(f"[MastiffIKDiag] commanding pose: {label}")
    for leg_index, leg_name in enumerate(LEG_ORDER):
        values_deg = torch.rad2deg(leg_joint_targets_sim[leg_index]).tolist()
        print(
            f"[MastiffIKDiag][Command][{label}][{leg_name}] "
            f"sim_deg=(HAA={values_deg[0]:+.3f},HFE={values_deg[1]:+.3f},KFE={values_deg[2]:+.3f})"
        )
    step_with_targets(sim, robot, target, locked_root_pose, locked_root_vel, args_cli.settle_steps)
    print_joint_group_snapshot(robot, joint_id_map, label)


def run_gait_cycle(
    sim: SimulationContext,
    robot: Articulation,
    joint_id_map: dict[str, tuple[int, int, int]],
    foot_body_ids: dict[str, int],
    hip_body_ids: dict[str, int],
    joint_signs: torch.Tensor,
    generator,
    side_signs: torch.Tensor,
    zero_joint_pos: torch.Tensor,
    locked_root_pose: torch.Tensor,
    locked_root_vel: torch.Tensor,
    gait_bundle: dict[str, torch.Tensor],
) -> None:
    frame_count = int(gait_bundle["frame_count"].item())
    hold_steps = int(gait_bundle["hold_steps"].item())
    log_stride = max(int(args_cli.gait_log_frame_stride), 1)
    print("[MastiffIKDiag] commanding repeated gait cycle")
    first_cycle = True
    while simulation_app.is_running():
        for frame_index in range(frame_count):
            leg_joint_targets_sim = gait_bundle["sim_targets"][frame_index]
            target = build_sim_target(zero_joint_pos, joint_id_map, leg_joint_targets_sim)
            step_with_targets(sim, robot, target, locked_root_pose, locked_root_vel, hold_steps)
            if args_cli.gait_log_every_frame and (frame_index % log_stride == 0):
                measured_sim = measured_leg_joint_positions(robot, joint_id_map)
                actual_foot_rel = actual_foot_positions_relative_to_hip(robot, foot_body_ids, hip_body_ids)
                measured_fk_mm = measured_fk_foot_positions(measured_sim, joint_signs, generator, side_signs) * 1000.0
                print_gait_cycle_frame_log(frame_index, gait_bundle, measured_sim, actual_foot_rel, measured_fk_mm)
            elif first_cycle and frame_index in (0, frame_count // 4, frame_count // 2, (3 * frame_count) // 4):
                print_joint_group_snapshot(robot, joint_id_map, f"gait_cycle_frame{frame_index}")
            if not simulation_app.is_running():
                break
        first_cycle = False


def run_diagnostic(sim: SimulationContext, robot: Articulation, origins: torch.Tensor) -> None:
    locked_root_pose, locked_root_vel, zero_joint_pos, _ = reset_robot(robot, origins, args_cli.hold_base_height)
    joint_id_map: dict[str, tuple[int, int, int]] = {}
    for leg_name in LEG_ORDER:
        joint_ids = []
        for joint_name in LEG_CONFIGS[leg_name]["joints"]:
            found_ids, _ = robot.find_joints([joint_name])
            if len(found_ids) == 0:
                raise RuntimeError(f"Joint not found: {joint_name}")
            joint_ids.append(int(found_ids[0]))
        joint_id_map[leg_name] = tuple(joint_ids)

    print("[MastiffIKDiag] joint order:", robot.joint_names)
    print_leg_limit_snapshot(robot, joint_id_map)
    hip_body_name_map = {leg_name: getattr(args_cli, f"{leg_name.lower()}_hip_body") for leg_name in LEG_ORDER}
    thigh_body_name_map = {leg_name: getattr(args_cli, f"{leg_name.lower()}_thigh_body") for leg_name in LEG_ORDER}
    foot_body_name_map = {leg_name: getattr(args_cli, f"{leg_name.lower()}_foot_body") for leg_name in LEG_ORDER}
    foot_body_ids = resolve_leg_body_ids(robot, foot_body_name_map)
    thigh_body_ids = resolve_leg_body_ids(robot, thigh_body_name_map)
    hip_body_ids = resolve_leg_body_ids(robot, hip_body_name_map)
    print_leg_body_mapping(robot, foot_body_ids, thigh_body_ids, hip_body_ids)

    bundle = build_pose_bundle()
    print_pose_report(bundle)
    gait_bundle = build_gait_cycle_bundle(bundle)
    print_gait_cycle_report(gait_bundle)

    zero_pose_sim = torch.zeros(len(LEG_ORDER), 3, dtype=torch.float64)
    standing_sim = bundle["standing_sim"]
    ik_sim = bundle["ik_sim"]

    if args_cli.target_mode == "gait_cycle":
        run_gait_cycle(
            sim,
            robot,
            joint_id_map,
            foot_body_ids,
            hip_body_ids,
            bundle["joint_signs"],
            bundle["generator"],
            bundle["side_signs"],
            zero_joint_pos,
            locked_root_pose,
            locked_root_vel,
            gait_bundle,
        )
        return
    if args_cli.target_mode == "zero":
        run_pose(sim, robot, joint_id_map, zero_joint_pos, locked_root_pose, locked_root_vel, "zero", zero_pose_sim)
    elif args_cli.target_mode == "standing_input":
        run_pose(sim, robot, joint_id_map, zero_joint_pos, locked_root_pose, locked_root_vel, "standing_input", standing_sim)
    elif args_cli.target_mode == "ik_roundtrip":
        run_pose(sim, robot, joint_id_map, zero_joint_pos, locked_root_pose, locked_root_vel, "ik_roundtrip", ik_sim)
    else:
        run_pose(sim, robot, joint_id_map, zero_joint_pos, locked_root_pose, locked_root_vel, "zero", zero_pose_sim)
        step_with_targets(sim, robot, build_sim_target(zero_joint_pos, joint_id_map, zero_pose_sim), locked_root_pose, locked_root_vel, args_cli.pause_steps)
        run_pose(sim, robot, joint_id_map, zero_joint_pos, locked_root_pose, locked_root_vel, "standing_input", standing_sim)
        step_with_targets(sim, robot, build_sim_target(zero_joint_pos, joint_id_map, zero_pose_sim), locked_root_pose, locked_root_vel, args_cli.pause_steps)
        run_pose(sim, robot, joint_id_map, zero_joint_pos, locked_root_pose, locked_root_vel, "ik_roundtrip", ik_sim)

    hold_target = build_sim_target(zero_joint_pos, joint_id_map, ik_sim if args_cli.target_mode != "zero" else zero_pose_sim)
    while simulation_app.is_running():
        step_with_targets(sim, robot, hold_target, locked_root_pose, locked_root_vel, 1)


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.8, 0.0, 1.8], [0.0, 0.0, 0.3])

    robot, origins = design_scene()
    origins = origins.to(device=sim.device)

    sim.reset()
    print("[MastiffIKDiag] setup complete")
    run_diagnostic(sim, robot, origins)


if __name__ == "__main__":
    main()
    simulation_app.close()
