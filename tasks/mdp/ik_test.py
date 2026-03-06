import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ==================== 1. 机器人几何参数与 DH 模型 ====================

# 连杆长度 (基于之前的计算结果)
L_COXA = 52.0
L_FEMUR = 66.03
L_TIBIA = 128.48

# 初始姿态的绝对角度 (用于将 IK 结果转换为 DH 输入)
# Femur: 向上 77.23度, Tibia: 相对 Femur 向下 154.46度
FEMUR_REST_ANGLE_GLOBAL = np.arctan2(64.4, 14.6) 
# Tibia Rest Angle (Relative to Femur in geometric sense)
# 之前算出 Tibia Global 是 -77.23, 所以相对角度是 -154.46
TIBIA_REST_ANGLE_RELATIVE = np.arctan2(-125.3, 28.4) - FEMUR_REST_ANGLE_GLOBAL

def dh_matrix(theta, d, a, alpha):
    """标准 DH 变换矩阵"""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d],
        [0,   0,      0,     1]
    ])

def forward_kinematics(theta1, theta2, theta3):
    """
    正运动学：输入控制角 (相对于初始姿态的偏差)，输出各关节坐标
    """
    # 这里的输入 theta 是相对于 "Home Pose" 的 delta
    # DH 表中的 theta = 输入 + 初始偏移
    
    t1 = theta1
    t2 = theta2 + FEMUR_REST_ANGLE_GLOBAL
    t3 = theta3 + TIBIA_REST_ANGLE_RELATIVE
    
    dh_table = [
        [t1, 0, L_COXA, np.pi/2],
        [t2, 0, L_FEMUR, 0],
        [t3, 0, L_TIBIA, 0]
    ]
    
    transforms = []
    T = np.eye(4)
    positions = [T[:3, 3]]
    
    for params in dh_table:
        T = T @ dh_matrix(*params)
        transforms.append(T)
        positions.append(T[:3, 3])
        
    return np.array(positions)

# ==================== 2. 逆运动学 (Inverse Kinematics) ====================

def solve_ik(target_x, target_y, target_z):
    """
    输入: 目标点坐标 (x, y, z)
    输出: 关节控制角 (d_theta1, d_theta2, d_theta3) -> 弧度
    如果不可达，返回 None
    """
    # --- 1. 求解 Coxa (Theta 1) ---
    # 在 XY 平面投影求解
    theta1 = np.arctan2(target_y, target_x)
    
    # --- 2. 转换到 Femur-Tibia 平面 ---
    # 计算目标点在水平面上的径向距离
    r_projection = np.sqrt(target_x**2 + target_y**2)
    
    # 减去 Coxa 长度，得到 planar 链的水平跨度
    w = r_projection - L_COXA
    h = target_z
    
    # 目标点到 J2 (Femur关节) 的直线距离
    L_virtual = np.sqrt(w**2 + h**2)
    
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
    
    # 计算物理绝对角度
    # 这里假设 "Knee Up" (膝盖向上) 构型
    theta2_absolute = gamma + alpha 
    
    # Tibia 相对 Femur 的角度。
    # 在几何三角形中，内角是 beta。
    # 伸直时为 0度? 不，DH定义通常 X轴沿连杆。
    # 如果 Tibia 往回折，相对角度通常是负的 (如 -150度)
    # 对应的几何关系是: theta3_rel = beta - pi
    theta3_relative = beta - np.pi
    
    # --- 5. 转换为控制量 (Delta) ---
    d_theta1 = theta1 - 0 # Coxa 初始为 0
    d_theta2 = theta2_absolute - FEMUR_REST_ANGLE_GLOBAL
    d_theta3 = theta3_relative - TIBIA_REST_ANGLE_RELATIVE
    
    return d_theta1, d_theta2, d_theta3

# ==================== 3. 轨迹生成与离散化 ====================

def generate_discrete_path(segment_length, delta_length, height, ground_z, alpha_deg):
    """
    根据 segment_length 将轨迹离散化为点序列
    """
    alpha_rad = np.radians(alpha_deg)
    center_offset_x = 120.0
    
    points = []
    
    # --- A. 摆动相 (半椭圆) ---
    # 弧长估算 (Ramanujan近似 或 简单数值积分)
    # 为简单起见，用点数采样近似弧长控制
    swing_perimeter_approx = np.pi * np.sqrt((delta_length/2)**2 + height**2) * 0.8 # 粗略估算
    num_swing = int(max(swing_perimeter_approx / segment_length, 5))
    
    phi_swing = np.linspace(0, np.pi, num_swing)
    
    # --- B. 支撑相 (直线) ---
    stance_length = delta_length
    num_stance = int(max(stance_length / segment_length, 2))
    
    phi_stance = np.linspace(np.pi, 2*np.pi, num_stance)
    
    # 合并相位
    all_phis = np.concatenate([phi_swing, phi_stance])
    
    for phi in all_phis:
        # CPG 局部坐标
        if phi <= np.pi: # Swing
            x_loc = -(delta_length/2.0) * np.cos(phi)
            z_loc = height * np.sin(phi)
        else: # Stance
            x_loc = -(delta_length/2.0) * np.cos(phi)
            z_loc = 0.0
            
        # 全局变换
        x_rot = x_loc * np.cos(alpha_rad)
        y_rot = x_loc * np.sin(alpha_rad)
        
        X = center_offset_x + x_rot
        Y = 0.0 + y_rot
        Z = ground_z + z_loc
        points.append([X, Y, Z])
        
    return np.array(points)

# ==================== 4. 动画主程序 ====================

def run_animation():
    # --- 参数设置 ---
    SEGMENT_LEN = 5.0    # 离散段长度 (mm) -> 越小动画越细腻/慢
    TRAJ_DELTA = 70.0   # 步幅
    TRAJ_HEIGHT = 50.0   # 抬腿高度
    TRAJ_GROUND = -70.0  # 地面高度
    TRAJ_ALPHA = 80.0    # 平面夹角
    
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
    
    # 设置场景范围
    ax.set_xlim(-20, 180)
    ax.set_ylim(-80, 80)
    ax.set_zlim(-150, 80)
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