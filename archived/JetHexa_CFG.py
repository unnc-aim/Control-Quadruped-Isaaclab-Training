import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import RayCasterCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


JetHexa_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/mocap/Projects/Hexa/assets/JetHexa.usd",
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
        pos=(0.0, 0.0, 0.2),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0}
    ),
    actuators={
        # Leg actuators - coxa joints (hip yaw)
        "coxa_actuator": ImplicitActuatorCfg(
            joint_names_expr=["coxa_joint_.*"],
            effort_limit=87.0,  # Typical servo torque for hexapod
            velocity_limit=10.0,
            stiffness=800.0,
            damping=40.0,
        ),
        # Leg actuators - femur joints (hip pitch)
        "femur_actuator": ImplicitActuatorCfg(
            joint_names_expr=["femur_joint_.*"],
            effort_limit=87.0,
            velocity_limit=10.0,
            stiffness=800.0,
            damping=40.0,
        ),
        # Leg actuators - tibia joints (knee pitch)
        "tibia_actuator": ImplicitActuatorCfg(
            joint_names_expr=["tibia_joint_.*"],
            effort_limit=87.0,
            velocity_limit=10.0,
            stiffness=800.0,
            damping=40.0,
        ),
    },
)
