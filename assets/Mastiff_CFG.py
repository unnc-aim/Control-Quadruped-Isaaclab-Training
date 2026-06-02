import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import RayCasterCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from pathlib import Path
_PROJECT_PATH = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Motor: 10010L (all 12 joints use the same motor)
#   Rated torque      : 40 Nm
#   Peak torque       : 120 Nm
#   No-load max speed : 200 rpm @ 48 V  →  200 * 2π/60 ≈ 20.94 rad/s
#   Gear ratio        : 10 : 1  (output-shaft values already above)
#   Phase resistance  : 0.11 Ω
#   Phase inductance  : 85 μH
#   Encoder           : 14-bit magnetic (absolute, single-turn on output shaft)
# ---------------------------------------------------------------------------

Mastiff_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(_PROJECT_PATH / "assets" / "Mastiff.usd"),
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
        joint_vel={".*": 0.0}
    ),
    actuators={
        # Hip Abduction/Adduction joints (HAA)
        "haa_actuator": ImplicitActuatorCfg(
            joint_names_expr=["HAA_.*"],
            effort_limit=4000.0,           # rated (continuous) torque [Nm]
            velocity_limit=2000.94,        # no-load max speed @ 48 V [rad/s]
            stiffness=1e6,               # Kp  [Nm/rad]
            damping=2.0,                 # Kd  [Nm·s/rad]
        ),
        # Hip Flexion/Extension joints (HFE)
        "hfe_actuator": ImplicitActuatorCfg(
            joint_names_expr=["HFE_.*"],
            effort_limit=4000.0,
            velocity_limit=2000.94,
            stiffness=1e6,
            damping=200.0,
        ),
        # Knee Flexion/Extension joints (KFE)
        "kfe_actuator": ImplicitActuatorCfg(
            joint_names_expr=["KFE_.*"],
            effort_limit=4000.0,
            velocity_limit=2000.94,
            stiffness=1e6,
            damping=200.0,
        ),
    },
)

# joint_names: ['HAA_FRONT_LEFT', 'HAA_FRONT_RIGHT', 'HAA_REAR_LEFT', 'HAA_REAR_RIGHT', 'HFE_FRONT_LEFT', 'HFE_FRONT_RIGHT', 'HFE_REAR_LEFT', 'HFE_REAR_RIGHT', 'KFE_FRONT_LEFT', 'KFE_FRONT_RIGHT', 'KFE_REAR_LEFT', 'KFE_REAR_RIGHT']
# -------------------------------------------------------
# body_names: ['Body_v1', 'Hip_v1_02', 'Hip_v1', 'Hip_v1_01', 'Hip_v1_03', 'ThighL_v1_01', 'ThighRstp_v1', 'ThighL_v1', 'ThighRstp_v1_01', 'Foot_v1', 'Foot_v1_03', 'Foot_v1_01', 'Foot_v1_02']
