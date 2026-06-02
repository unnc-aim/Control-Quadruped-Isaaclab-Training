from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .mastiff_flat_task import MastiffFlatEnvCfg


@configclass
class DirectActionsCfg:
    """Direct 12-DoF joint position actions for Mastiff."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        # Keep explicit order for stable policy-action semantics.
        joint_names=[
            "HAA_FRONT_LEFT",
            "HFE_FRONT_LEFT",
            "KFE_FRONT_LEFT",
            "HAA_FRONT_RIGHT",
            "HFE_FRONT_RIGHT",
            "KFE_FRONT_RIGHT",
            "HAA_REAR_LEFT",
            "HFE_REAR_LEFT",
            "KFE_REAR_LEFT",
            "HAA_REAR_RIGHT",
            "HFE_REAR_RIGHT",
            "KFE_REAR_RIGHT",
        ],
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class MastiffFlatDirectEnvCfg(MastiffFlatEnvCfg):
    """Flat-task variant with direct joint-position control."""

    actions: DirectActionsCfg = DirectActionsCfg()
