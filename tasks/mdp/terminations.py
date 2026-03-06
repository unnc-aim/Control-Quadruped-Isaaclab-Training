
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.command_manager import CommandTerm


def joint_pos_out_of_manual_limit(
    env: ManagerBasedRLEnv, bounds: tuple[float, float], 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_names = None
) -> torch.Tensor:
    """Terminate when asset's joint positions are outside of configured bounds.

    Note:
        This function is similar to :func:`joint_pos_out_of_limit` but allows
        the user to specify the bounds manually. If joint_names is provided,
        it will automatically find the joint IDs for those joints.
    
    Args:
        env: The environment instance.
        bounds: Tuple of (lower_bound, upper_bound) for joint positions.
        asset_cfg: Asset configuration with robot name and joint_ids.
        joint_names: List of joint names to check. If provided, joint_ids
                    will be automatically resolved.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # If joint_names is provided, resolve joint IDs automatically
    if joint_names is not None:
        if isinstance(joint_names, str):
            joint_names = [joint_names]
        
        # Get all joint names from the asset
        all_joint_names = asset.joint_names
        
        # Find indices of specified joints
        joint_indices = []
        for joint_name in joint_names:
            try:
                joint_idx = all_joint_names.index(joint_name)
                joint_indices.append(joint_idx)
            except ValueError:
                # Joint name not found, print warning but continue
                print(f"Warning: Joint '{joint_name}' not found in asset "
                      f"'{asset_cfg.name}'.")
                print(f"Available joints: {all_joint_names}")
        
        if len(joint_indices) == 0:
            print(f"Error: No valid joints found from {joint_names}. "
                  f"Using all joints.")
            joint_ids = slice(None)
        else:
            joint_ids = joint_indices
    else:
        # Use joint_ids from asset_cfg if provided, otherwise use all joints
        if asset_cfg.joint_ids is None:
            joint_ids = slice(None)
        else:
            joint_ids = asset_cfg.joint_ids
    # print(asset.data.joint_pos[:, joint_ids])
    # compute any violations
    out_of_upper_limits = torch.any(
        asset.data.joint_pos[:, joint_ids] > bounds[1], dim=1
    )
    out_of_lower_limits = torch.any(
        asset.data.joint_pos[:, joint_ids] < bounds[0], dim=1
    )
    return torch.logical_or(out_of_upper_limits, out_of_lower_limits)