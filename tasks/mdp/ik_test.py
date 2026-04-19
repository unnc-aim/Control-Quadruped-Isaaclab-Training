import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ==================== 1. 机器人几何参数与几何模型 ====================

# 连杆长度 (mm，按 Mastiff 四足底盘参数)
L_COXA = 120.05
L_FEMUR = 260
L_TIBIA = 300

# 初始姿态绝对角 (按项目 CPGConfig 的四足狗几何定义)
# femur_xy = (femur_y, femur_x), tibia_xy = (tibia_y, tibia_x)
FEMUR_REST_ANGLE_GLOBAL = np.deg2rad(-40)
TIBIA_REST_ANGLE_RELATIVE = np.deg2rad(-100)# 84.61

def rot_x(theta):
    """绕 X 轴旋转矩阵"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ])

def forward_kinematics(theta1, theta2, theta3):
    """
    正运动学：输入控制角 (相对于初始姿态的偏差)，输出各关节坐标
    """
    # 这里的输入 theta 是相对于 "Home Pose" 的 delta
    # t2/t3 转成绝对几何角，t1 为 HAA 外展/内收角
    t1 = theta1
    t2 = theta2 + FEMUR_REST_ANGLE_GLOBAL
    t3 = theta3 + TIBIA_REST_ANGLE_RELATIVE

    # HAA 旋转后，COXA 沿局部 +Y；FEMUR/TIBIA 在局部 XZ 平面内运动
    R_haa = rot_x(t1)

    p0 = np.array([0.0, 0.0, 0.0])
    p1 = R_haa @ np.array([0.0, L_COXA, 0.0])

    femur_vec_local = np.array([
        L_FEMUR * np.cos(t2),
        0.0,
        L_FEMUR * np.sin(t2),
    ])
    tibia_vec_local = np.array([
        L_TIBIA * np.cos(t2 + t3),
        0.0,
        L_TIBIA * np.sin(t2 + t3),
    ])

    p2 = p1 + (R_haa @ femur_vec_local)
    p3 = p2 + (R_haa @ tibia_vec_local)
    return np.vstack([p0, p1, p2, p3])

# ==================== 2. 逆运动学 (Inverse Kinematics) ====================

def solve_ik(target_x, target_y, target_z):
    """
    输入: 目标点坐标 (x, y, z)
    输出: 关节控制角 (d_theta1, d_theta2, d_theta3) -> 弧度
    如果不可达，返回 None
    """
    # --- 1. 求解 HAA (Theta 1) ---
    # 约束：将足端旋回到 HAA 局部坐标后，其 y 分量应为 L_COXA
    r_yz = np.hypot(target_y, target_z)
    if r_yz < L_COXA:
        return None

    phi_yz = np.arctan2(target_z, target_y)
    # 选取使局部 z 为负的分支（狗腿常见向下折叠构型）
    theta1 = phi_yz + np.arccos(np.clip(L_COXA / r_yz, -1.0, 1.0))

    # --- 2. 转换到 Femur-Tibia 平面 ---
    c1, s1 = np.cos(theta1), np.sin(theta1)
    # R_x(-theta1) * p_target
    w = target_x
    h = -target_y * s1 + target_z * c1
    L_virtual = np.hypot(w, h)

    if L_virtual < 1e-9:
        return None
    
    # --- 3. 检查可达性 ---
    if L_virtual > (L_FEMUR + L_TIBIA) or L_virtual < abs(L_FEMUR - L_TIBIA):
        return None # 目标点超出工作空间
    
    # --- 4. 余弦定理求解 J2, J3 ---
    # alpha: Femur 连杆与 L_virtual 的夹角
    # beta:  Femur 与 Tibia 的夹角 (内部角)
    
    cos_beta = (L_FEMUR**2 + L_TIBIA**2 - L_virtual**2) / (2 * L_FEMUR * L_TIBIA)
    # 限制数值误差在 [-1, 1]
    cos_beta = np.clip(cos_beta, -1.0, 1.0) 
    beta = np.arccos(cos_beta)
    
    cos_alpha = (L_FEMUR**2 + L_virtual**2 - L_TIBIA**2) / (2 * L_FEMUR * L_virtual)
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    alpha = np.arccos(cos_alpha)
    
    # L_virtual 相对于水平线的倾角
    gamma = np.arctan2(h, w)
    
    # 计算物理绝对角度（四足狗常见的 "Knee Down" 解）
    theta2_absolute = gamma - alpha
    theta3_relative = np.pi - beta
    
    # --- 5. 转换为控制量 (Delta) ---
    d_theta1 = theta1  # HAA 初始为 0
    d_theta2 = theta2_absolute - FEMUR_REST_ANGLE_GLOBAL
    d_theta3 = theta3_relative - TIBIA_REST_ANGLE_RELATIVE
    
    return d_theta1, d_theta2, d_theta3

# ==================== 3. 轨迹生成与离散化 ====================

def generate_discrete_path(segment_length, delta_length, height, ground_z, alpha_deg):
    """
    根据 segment_length 将轨迹离散化为点序列
    """
    alpha_rad = np.radians(alpha_deg)
    # 将“0位”对齐到支撑相正中间：使输入初始姿态(0,0,0)对应中点触地点
    home_foot_pos = forward_kinematics(0.0, 0.0, 0.0)[-1]
    gait_dir = np.array([np.cos(alpha_rad), np.sin(alpha_rad)])
    phase_zero_shift = np.pi / 2.0
    x_loc_at_zero = (delta_length / 2.0) * np.cos(phase_zero_shift)
    center_offset_xy = home_foot_pos[:2] - x_loc_at_zero * gait_dir
    
    points = []
    
    # --- A. 支撑相 (直线) ---
    # 将相位 0 定义为轨迹右端点，且从支撑相开始（足端先向左运动）
    stance_length = delta_length
    num_stance = int(max(stance_length / segment_length, 2))
    phi_stance = np.linspace(0.0, np.pi, num_stance, endpoint=False)
    
    # --- B. 摆动相 (半椭圆) ---
    swing_perimeter_approx = np.pi * np.sqrt((delta_length/2)**2 + height**2) * 0.8
    num_swing = int(max(swing_perimeter_approx / segment_length, 5))
    phi_swing = np.linspace(np.pi, 2*np.pi, num_swing, endpoint=False)
    
    # 合并相位：Stance -> Swing
    base_phis = np.concatenate([phi_stance, phi_swing])
    # 相位平移后，frame=0 落在支撑相中点（phi=pi/2）
    all_phis = (base_phis + phase_zero_shift) % (2 * np.pi)
    
    for phi in all_phis:
        # phi=0 在支撑相中点；初始会先向左完成后半段支撑，再进入摆动回到右端
        x_loc = (delta_length / 2.0) * np.cos(phi)
        if phi < np.pi:  # Stance
            z_loc = 0.0
        else:  # Swing
            z_loc = height * np.sin(phi - np.pi)
            
        # 全局变换
        x_rot = x_loc * gait_dir[0]
        y_rot = x_loc * gait_dir[1]
        
        X = center_offset_xy[0] + x_rot
        Y = center_offset_xy[1] + y_rot
        Z = ground_z + z_loc
        points.append([X, Y, Z])
    
    return np.array(points)

# ==================== 4. 动画主程序 ====================

def run_animation():
    # --- 参数设置 ---
    SEGMENT_LEN = 3.0    # 离散段长度 (mm) -> 越小动画越细腻/慢
    TRAJ_DELTA = 200.0    # 步幅
    TRAJ_HEIGHT = 60.0   # 抬腿高度
    TRAJ_GROUND = forward_kinematics(0.0, 0.0, 0.0)[-1, 2]  # 让0位严格落在支撑相中点触地点
    TRAJ_ALPHA = 0.0     # 摆动平面夹角
    
    # 1. 生成轨迹点
    path_points = generate_discrete_path(SEGMENT_LEN, TRAJ_DELTA, TRAJ_HEIGHT, TRAJ_GROUND, TRAJ_ALPHA)
    print(f"Generated {len(path_points)} frames based on segment length {SEGMENT_LEN}mm")
    
    # 2. 准备绘图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 初始化绘图对象
    # 机械臂 (Line)
    leg_line, = ax.plot([], [], [], '-o', linewidth=4, color='black', markersize=6, label='Leg')
    # 轨迹 (Line)
    traj_line, = ax.plot(path_points[:,0], path_points[:,1], path_points[:,2], 
                         color='red', alpha=0.3, linewidth=1, label='Target Path')
    # 当前目标点 (Scatter)
    target_dot = ax.scatter([], [], [], color='blue', s=50, label='Current Target')
    
    # 设置场景范围（覆盖整条轨迹和初始姿态），并保持三轴单位长度一致
    tick_step = 40
    home_positions = forward_kinematics(0.0, 0.0, 0.0)
    vis_points = np.vstack([path_points, home_positions, np.zeros((1, 3))])
    mins = vis_points.min(axis=0) - 40.0
    maxs = vis_points.max(axis=0) + 40.0
    x_lim = (
        tick_step * np.floor(mins[0] / tick_step),
        tick_step * np.ceil(maxs[0] / tick_step),
    )
    y_lim = (
        tick_step * np.floor(mins[1] / tick_step),
        tick_step * np.ceil(maxs[1] / tick_step),
    )
    z_lim = (
        tick_step * np.floor(mins[2] / tick_step),
        tick_step * np.ceil(maxs[2] / tick_step),
    )
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_zlim(*z_lim)
    ax.set_box_aspect((
        x_lim[1] - x_lim[0],
        y_lim[1] - y_lim[0],
        z_lim[1] - z_lim[0],
    ))
    ax.set_xticks(np.arange(x_lim[0], x_lim[1] + tick_step, tick_step))
    ax.set_yticks(np.arange(y_lim[0], y_lim[1] + tick_step, tick_step))
    ax.set_zticks(np.arange(z_lim[0], z_lim[1] + tick_step, tick_step))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'IK Walking Animation (Step={TRAJ_DELTA}mm, Seg={SEGMENT_LEN}mm)')
    ax.legend()
    ax.view_init(elev=20, azim=-45)

    # 3. 动画更新函数
    def update(frame):
        # 获取当前帧的目标点
        target = path_points[frame % len(path_points)]
        
        # IK 求解
        ik_result = solve_ik(target[0], target[1], target[2])
        
        if ik_result:
            th1, th2, th3 = ik_result
            # 计算正运动学用于绘图
            positions = forward_kinematics(th1, th2, th3)
            
            # 更新机械臂
            leg_line.set_data(positions[:, 0], positions[:, 1])
            leg_line.set_3d_properties(positions[:, 2])
            
            # 更新目标点指示
            target_dot._offsets3d = ([target[0]], [target[1]], [target[2]])
            
            return leg_line, target_dot
        else:
            print(f"Frame {frame}: IK Failed (Unreachable)")
            return leg_line, target_dot

    # 4. 创建动画
    # interval: 帧间隔(ms)
    ani = animation.FuncAnimation(fig, update, frames=len(path_points), 
                                  interval=50, blit=False)
    
    plt.show()

if __name__ == "__main__":
    run_animation()
