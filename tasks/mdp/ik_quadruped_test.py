from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ==================== 1. 机器人几何参数与几何模型 ====================

# 连杆长度 (mm，按 Mastiff 四足底盘参数)
L_COXA = 120.05
L_FEMUR = 260.0
L_TIBIA = 300.0

# USD/FK 几何零位：用于 planner joint <-> geometric angle 映射。
FEMUR_ZERO_ANGLE_GLOBAL = np.deg2rad(-150.0)
TIBIA_ZERO_ANGLE_RELATIVE = np.deg2rad(15.0)

# Mastiff sim/mechanical joint sign: q_sim = q_planner * SIGN。
LEG_JOINT_SIGNS = {
    "FL": np.array([+1.0, +1.0, -1.0]),
    "FR": np.array([+1.0, -1.0, -1.0]),
    "RL": np.array([-1.0, +1.0, -1.0]),
    "RR": np.array([-1.0, -1.0, -1.0]),
}

# Standing pose is expressed in planner space, matching QuadrupedGaitAction.
STANDING_HAA = np.deg2rad(0.0)
STANDING_HFE = np.deg2rad(0.0)
STANDING_KFE = np.deg2rad(40.0)
USE_STANDING_HOME = True

HEADLESS_OUTPUT_GIF = Path('/tmp/ik_quadruped_test.gif')
HEADLESS_OUTPUT_PNG = Path('/tmp/ik_quadruped_test_first_frame.png')

# 只用于四足可视化的固定身体几何。HAA 原点放在 body 四角附近。
BODY_LENGTH = 520.0
BODY_WIDTH = 180.0
HIP_Z = 0.0

LEG_CONFIGS = {
    # Trot: FL/RR 一组，FR/RL 一组，相位差 180 度
    "FL": {
        "hip": np.array([BODY_LENGTH / 2.0, BODY_WIDTH / 2.0, HIP_Z]),
        "side": 1.0,
        "phase_offset_deg": 0.0,
        "color": "tab:blue",
    },
    "FR": {
        "hip": np.array([BODY_LENGTH / 2.0, -BODY_WIDTH / 2.0, HIP_Z]),
        "side": -1.0,
        "phase_offset_deg": 180.0,
        "color": "tab:orange",
    },
    "RL": {
        "hip": np.array([-BODY_LENGTH / 2.0, BODY_WIDTH / 2.0, HIP_Z]),
        "side": 1.0,
        "phase_offset_deg": 180.0,
        "color": "tab:green",
    },
    "RR": {
        "hip": np.array([-BODY_LENGTH / 2.0, -BODY_WIDTH / 2.0, HIP_Z]),
        "side": -1.0,
        "phase_offset_deg": 0.0,
        "color": "tab:red",
    },
}


def rot_x(theta):
    """绕 X 轴旋转矩阵"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ]
    )


def forward_kinematics(theta1, theta2, theta3, side=1.0):
    """
    正运动学：输入 planner joint angle，输出单条腿局部坐标。

    side=+1 表示左腿，COXA 沿局部 +Y；
    side=-1 表示右腿，COXA 沿局部 -Y。

    注意：theta2/theta3 不是站姿 delta，而是经过 zero-angle 映射前的
    planner joint angle；真正的几何绝对角由 ZERO_ANGLE 常量决定。
    """
    side = 1.0 if side >= 0.0 else -1.0

    t1 = theta1
    t2 = theta2 + FEMUR_ZERO_ANGLE_GLOBAL
    t3 = theta3 + TIBIA_ZERO_ANGLE_RELATIVE

    R_haa = rot_x(t1)

    p0 = np.array([0.0, 0.0, 0.0])
    p1 = R_haa @ np.array([0.0, side * L_COXA, 0.0])

    femur_vec_local = np.array(
        [
            L_FEMUR * np.cos(t2),
            0.0,
            L_FEMUR * np.sin(t2),
        ]
    )
    tibia_vec_local = np.array(
        [
            L_TIBIA * np.cos(t2 + t3),
            0.0,
            L_TIBIA * np.sin(t2 + t3),
        ]
    )

    p2 = p1 + (R_haa @ femur_vec_local)
    p3 = p2 + (R_haa @ tibia_vec_local)
    return np.vstack([p0, p1, p2, p3])


def planner_to_sim_joint_targets(theta1_planner, theta2_planner, theta3_planner, leg_name):
    """把 planner 角按 Mastiff 的每腿 joint_sign 转为 sim/mechanical 角。"""
    planner = np.array([theta1_planner, theta2_planner, theta3_planner])
    return planner * LEG_JOINT_SIGNS[leg_name]


def standing_planner_joint_targets():
    """返回 standing pose 在 planner 空间下的 joint targets。"""
    return np.array([STANDING_HAA, STANDING_HFE, STANDING_KFE])


def nominal_home_foot_pos(side=1.0):
    """返回用于 gait home reference 的足端位置。"""
    if USE_STANDING_HOME:
        standing_q = standing_planner_joint_targets()
        return forward_kinematics(standing_q[0], standing_q[1], standing_q[2], side=side)[-1]
    return forward_kinematics(0.0, 0.0, 0.0, side=side)[-1]


# ==================== 2. 逆运动学 (Inverse Kinematics) ====================


def solve_left_leg_ik(target_x, target_y, target_z):
    """
    单条左腿 IK。

    输入: 左腿局部目标点坐标 (x, y, z)
    输出: planner joint angle (theta1, theta2, theta3) -> 弧度
    如果不可达，返回 None
    """
    r_yz = np.hypot(target_y, target_z)
    if r_yz < L_COXA:
        return None

    phi_yz = np.arctan2(target_z, target_y)
    delta = np.arccos(np.clip(L_COXA / r_yz, -1.0, 1.0))

    theta1_a = phi_yz - delta
    theta1_b = phi_yz + delta

    h_a = -target_y * np.sin(theta1_a) + target_z * np.cos(theta1_a)
    h_b = -target_y * np.sin(theta1_b) + target_z * np.cos(theta1_b)

    theta1 = theta1_a if h_a < h_b else theta1_b
    h = min(h_a, h_b)
    w = target_x
    l_virtual = np.hypot(w, h)

    if l_virtual < 1e-9:
        return None
    if l_virtual > (L_FEMUR + L_TIBIA) or l_virtual < abs(L_FEMUR - L_TIBIA):
        return None

    cos_beta = (L_FEMUR**2 + L_TIBIA**2 - l_virtual**2) / (2.0 * L_FEMUR * L_TIBIA)
    cos_beta = np.clip(cos_beta, -1.0, 1.0)
    beta = np.arccos(cos_beta)

    cos_alpha = (L_FEMUR**2 + l_virtual**2 - L_TIBIA**2) / (2.0 * L_FEMUR * l_virtual)
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    alpha = np.arccos(cos_alpha)

    gamma = np.arctan2(h, w)
    theta2_absolute = gamma - alpha
    theta3_relative = np.pi - beta

    q1 = theta1
    q2 = theta2_absolute - FEMUR_ZERO_ANGLE_GLOBAL
    q3 = theta3_relative - TIBIA_ZERO_ANGLE_RELATIVE

    return q1, q2, q3


def solve_ik(target_x, target_y, target_z, side=1.0):
    """
    四足版本 IK。

    左腿直接使用左腿 IK；
    右腿只做 Y 方向镜像，保持与 quadruped_gait_generator.py 一致。
    """
    side = 1.0 if side >= 0.0 else -1.0
    return solve_left_leg_ik(target_x, side * target_y, target_z)


# ==================== 3. 轨迹生成与离散化 ====================


def generate_base_phases(segment_length, delta_length, height):
    """根据 segment_length 生成一个步态周期的相位序列。"""
    phase_zero_shift = np.pi / 2.0

    stance_length = delta_length
    num_stance = int(max(stance_length / segment_length, 2))
    phi_stance = np.linspace(0.0, np.pi, num_stance, endpoint=False)

    swing_perimeter_approx = np.pi * np.sqrt((delta_length / 2.0) ** 2 + height**2) * 0.8
    num_swing = int(max(swing_perimeter_approx / segment_length, 5))
    phi_swing = np.linspace(np.pi, 2.0 * np.pi, num_swing, endpoint=False)

    base_phis = np.concatenate([phi_stance, phi_swing])
    return (base_phis + phase_zero_shift) % (2.0 * np.pi)


def target_from_phase(phi, side, delta_length, height, ground_z, alpha_deg, home_foot_pos=None):
    """从相位生成单条腿局部目标点。"""
    alpha_rad = np.radians(alpha_deg)
    if home_foot_pos is None:
        home_foot_pos = nominal_home_foot_pos(side=side)
    gait_dir = np.array([np.cos(alpha_rad), np.sin(alpha_rad)])

    x_loc = (delta_length / 2.0) * np.cos(phi)
    if phi < np.pi:
        z_loc = 0.0
    else:
        z_loc = height * np.sin(phi - np.pi)

    x_rot = x_loc * gait_dir[0]
    y_rot = x_loc * gait_dir[1]

    x = home_foot_pos[0] + x_rot
    y = home_foot_pos[1] + y_rot
    z = ground_z + z_loc if home_foot_pos is None else home_foot_pos[2] + z_loc
    return np.array([x, y, z])


def generate_discrete_path(segment_length, delta_length, height, ground_z, alpha_deg, side, phase_offset_deg):
    """生成单条腿一个步态周期的局部足端轨迹。"""
    phase_offset = np.radians(phase_offset_deg)
    phases = (generate_base_phases(segment_length, delta_length, height) + phase_offset) % (2.0 * np.pi)
    home_foot_pos = nominal_home_foot_pos(side=side)
    points = [target_from_phase(phi, side, delta_length, height, ground_z, alpha_deg, home_foot_pos=home_foot_pos) for phi in phases]
    return np.array(points)


def build_leg_paths(segment_length, delta_length, height, ground_z, alpha_deg):
    """生成四条腿的世界坐标轨迹。"""
    paths = {}
    for leg_name, leg_cfg in LEG_CONFIGS.items():
        local_path = generate_discrete_path(
            segment_length,
            delta_length,
            height,
            ground_z,
            alpha_deg,
            side=leg_cfg["side"],
            phase_offset_deg=leg_cfg["phase_offset_deg"],
        )
        paths[leg_name] = leg_cfg["hip"] + local_path
    return paths


# ==================== 4. 绘图工具 ====================


def set_equal_axes(ax, points, margin=60.0, tick_step=100.0):
    """设置 3D 坐标轴范围，并保持三轴单位长度一致。"""
    mins = points.min(axis=0) - margin
    maxs = points.max(axis=0) + margin
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    half_span = 0.5 * span

    x_lim = (center[0] - half_span, center[0] + half_span)
    y_lim = (center[1] - half_span, center[1] + half_span)
    z_lim = (center[2] - half_span, center[2] + half_span)

    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_zlim(*z_lim)
    ax.set_box_aspect((1.0, 1.0, 1.0))

    ax.set_xticks(np.arange(tick_step * np.floor(x_lim[0] / tick_step), x_lim[1] + tick_step, tick_step))
    ax.set_yticks(np.arange(tick_step * np.floor(y_lim[0] / tick_step), y_lim[1] + tick_step, tick_step))
    ax.set_zticks(np.arange(tick_step * np.floor(z_lim[0] / tick_step), z_lim[1] + tick_step, tick_step))
    return x_lim, y_lim, z_lim


def add_body(ax):
    """添加固定身体矩形和 HAA 原点。"""
    x_front = BODY_LENGTH / 2.0
    x_rear = -BODY_LENGTH / 2.0
    y_left = BODY_WIDTH / 2.0
    y_right = -BODY_WIDTH / 2.0
    z = HIP_Z

    body_vertices = [
        [
            [x_front, y_left, z],
            [x_front, y_right, z],
            [x_rear, y_right, z],
            [x_rear, y_left, z],
        ]
    ]
    body = Poly3DCollection(body_vertices, facecolors="lightgray", edgecolors="black", linewidths=1.5, alpha=0.45)
    ax.add_collection3d(body)

    hip_points = np.array([leg_cfg["hip"] for leg_cfg in LEG_CONFIGS.values()])
    ax.scatter(hip_points[:, 0], hip_points[:, 1], hip_points[:, 2], color="black", s=25, label="HAA origins")
    for leg_name, leg_cfg in LEG_CONFIGS.items():
        text_pos = leg_cfg["hip"] + np.array([0.0, 0.0, 25.0])
        ax.text(text_pos[0], text_pos[1], text_pos[2], leg_name, color=leg_cfg["color"], fontsize=10)


def add_ground_plane(ax, x_lim, y_lim, ground_z):
    """添加地面参考平面。"""
    ground_vertices = [
        [
            [x_lim[0], y_lim[0], ground_z],
            [x_lim[1], y_lim[0], ground_z],
            [x_lim[1], y_lim[1], ground_z],
            [x_lim[0], y_lim[1], ground_z],
        ]
    ]
    ground = Poly3DCollection(ground_vertices, facecolors="tab:green", edgecolors="none", alpha=0.08)
    ax.add_collection3d(ground)


# ==================== 5. 动画主程序 ====================


def run_animation():
    SEGMENT_LEN = 3.0
    TRAJ_DELTA = 200.0
    TRAJ_HEIGHT = 60.0
    zero_home = forward_kinematics(0.0, 0.0, 0.0, side=1.0)[-1]
    standing_home = nominal_home_foot_pos(side=1.0)
    traj_ground = standing_home[2] if USE_STANDING_HOME else zero_home[2]
    TRAJ_ALPHA = 0.0

    print('Zero home foot local:', np.round(zero_home, 3))
    print('Standing planner joint targets:', np.round(np.degrees(standing_planner_joint_targets()), 3))
    print('Standing home foot local:', np.round(standing_home, 3))
    for leg_name, leg_cfg in LEG_CONFIGS.items():
        standing_sim = np.degrees(planner_to_sim_joint_targets(*standing_planner_joint_targets(), leg_name))
        ik_roundtrip = solve_ik(standing_home[0], standing_home[1], standing_home[2], side=leg_cfg['side'])
        if ik_roundtrip is None:
            print(f'IK roundtrip planner targets {leg_name}: unavailable')
            continue
        ik_roundtrip_sim = np.degrees(planner_to_sim_joint_targets(*ik_roundtrip, leg_name))
        print(f'Standing sim joint targets {leg_name}:', np.round(standing_sim, 3))
        print(f'IK roundtrip planner targets {leg_name}:', np.round(np.degrees(np.array(ik_roundtrip)), 3))
        print(f'IK roundtrip sim targets {leg_name}:', np.round(ik_roundtrip_sim, 3))
    print('Leg joint signs:', {name: tuple(signs.tolist()) for name, signs in LEG_JOINT_SIGNS.items()})
    print('Using standing home:', USE_STANDING_HOME)

    leg_paths = build_leg_paths(SEGMENT_LEN, TRAJ_DELTA, TRAJ_HEIGHT, traj_ground, TRAJ_ALPHA)
    frame_count = len(next(iter(leg_paths.values())))
    print(f"Generated {frame_count} frames for four-leg IK animation")

    unreachable_counts = {}
    for leg_name, leg_cfg in LEG_CONFIGS.items():
        local_path = leg_paths[leg_name] - leg_cfg["hip"]
        unreachable_counts[leg_name] = sum(solve_ik(p[0], p[1], p[2], side=leg_cfg["side"]) is None for p in local_path)
    if any(unreachable_counts.values()):
        print(f"IK warning, unreachable frames: {unreachable_counts}")

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    leg_lines = {}
    target_dots = {}
    artists = []

    for leg_name, leg_cfg in LEG_CONFIGS.items():
        color = leg_cfg["color"]
        path_points = leg_paths[leg_name]

        standing_q = standing_planner_joint_targets()
        initial_positions = leg_cfg["hip"] + forward_kinematics(standing_q[0], standing_q[1], standing_q[2], side=leg_cfg["side"])
        leg_line, = ax.plot(
            initial_positions[:, 0],
            initial_positions[:, 1],
            initial_positions[:, 2],
            "-o",
            linewidth=3,
            color=color,
            markersize=5,
            label=f"{leg_name} leg",
        )
        target_dot = ax.scatter(
            [path_points[0, 0]],
            [path_points[0, 1]],
            [path_points[0, 2]],
            color=color,
            s=45,
            marker="x",
        )
        ax.plot(
            path_points[:, 0],
            path_points[:, 1],
            path_points[:, 2],
            color=color,
            alpha=0.25,
            linewidth=1,
            label=f"{leg_name} target path",
        )

        leg_lines[leg_name] = leg_line
        target_dots[leg_name] = target_dot
        artists.extend([leg_line, target_dot])

    add_body(ax)

    vis_chunks = [np.array([leg_cfg["hip"] for leg_cfg in LEG_CONFIGS.values()])]
    for leg_name, leg_cfg in LEG_CONFIGS.items():
        vis_chunks.append(leg_paths[leg_name])
        standing_q = standing_planner_joint_targets()
        vis_chunks.append(leg_cfg["hip"] + forward_kinematics(standing_q[0], standing_q[1], standing_q[2], side=leg_cfg["side"]))
    vis_points = np.vstack(vis_chunks)
    x_lim, y_lim, z_lim = set_equal_axes(ax, vis_points)
    add_ground_plane(ax, x_lim, y_lim, traj_ground)

    ax.set_xlabel("X (mm, forward)")
    ax.set_ylabel("Y (mm, left)")
    ax.set_zlabel("Z (mm, up)")
    ax.set_title(f"Quadruped IK Trot Animation (Step={TRAJ_DELTA}mm, Height={TRAJ_HEIGHT}mm)")
    ax.legend(loc="upper left")
    ax.view_init(elev=24, azim=-48)

    def update(frame):
        for leg_name, leg_cfg in LEG_CONFIGS.items():
            target_world = leg_paths[leg_name][frame % frame_count]
            target_local = target_world - leg_cfg["hip"]

            ik_result = solve_ik(target_local[0], target_local[1], target_local[2], side=leg_cfg["side"])
            if ik_result is None:
                print(f"Frame {frame}: IK failed for {leg_name}, target={target_local}")
                continue

            th1, th2, th3 = ik_result
            positions_world = leg_cfg["hip"] + forward_kinematics(th1, th2, th3, side=leg_cfg["side"])

            leg_line = leg_lines[leg_name]
            leg_line.set_data(positions_world[:, 0], positions_world[:, 1])
            leg_line.set_3d_properties(positions_world[:, 2])

            target_dots[leg_name]._offsets3d = (
                [target_world[0]],
                [target_world[1]],
                [target_world[2]],
            )

        return artists

    ani = animation.FuncAnimation(fig, update, frames=frame_count, interval=50, blit=False)

    backend = plt.get_backend().lower()
    is_headless = 'agg' in backend
    if is_headless:
        print(f'Non-interactive backend detected: {plt.get_backend()}')
        try:
            writer = animation.PillowWriter(fps=20)
            ani.save(HEADLESS_OUTPUT_GIF, writer=writer)
            print(f'Saved animation to {HEADLESS_OUTPUT_GIF}')
        except Exception as exc:
            print(f'GIF export failed: {exc}')
            update(0)
            fig.savefig(HEADLESS_OUTPUT_PNG, dpi=160, bbox_inches='tight')
            print(f'Saved first frame to {HEADLESS_OUTPUT_PNG}')
        finally:
            plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    run_animation()
