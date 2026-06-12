import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from pathlib import Path

_PROJECT_PATH = Path(__file__).resolve().parents[1]


Lynxc_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(_PROJECT_PATH / "assets" / "LynxC.usd"),
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
        pos=(0.0, 0.0, 0.8),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
        rot=(0.7071068, 0.0, 0.7071068, 0.0)
    ),
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=3000.0,
            velocity_limit_sim=200.0,
            stiffness=1e5,
            damping=200.0,
        ),
    },
)
