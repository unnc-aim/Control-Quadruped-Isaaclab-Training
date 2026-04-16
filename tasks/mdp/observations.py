from __future__ import annotations

import functools
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedEnv


# ---------------------------------------------------------------------------
# Generic NaN/Inf sanitiser for observation functions
# ---------------------------------------------------------------------------

def nan_safe(func):
    """Decorator: replaces NaN with 0 and clips ±Inf in the returned tensor.

    When the physics solver becomes unstable (robot flips, penetrates ground,
    etc.), *any* observation derived from physics state can contain NaN or Inf.
    A single NaN entering the rsl_rl empirical normaliser corrupts its running
    mean/var **permanently**, which propagates to the actor's log_std and
    crashes with ``normal expects all elements of std >= 0.0``.

    Wrapping every observation function with this decorator breaks the chain at
    the source: the rollout buffer will never contain NaN, so the normaliser
    stays healthy and training can continue even if some envs briefly explode.
    """
    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return torch.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    return _wrapper


# ---------------------------------------------------------------------------
# Custom observation functions
# ---------------------------------------------------------------------------

def height_scan_safe(
    env: "ManagerBasedEnv",
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.5,
) -> torch.Tensor:
    """Height scan that sanitises NaN / Inf before returning.

    ``torch.clamp`` does **not** remove NaN – ``clamp(NaN, -1, 1) == NaN``.
    When the physics solver becomes unstable (e.g. the robot flips) the ray-
    caster can produce NaN hit positions, which then poison the observation
    buffer, the empirical normaliser and ultimately the policy weights
    (manifesting as ``normal expects all elements of std >= 0.0``).

    This wrapper replaces every NaN with 0 and every ±Inf with ±1 *before*
    the value leaves the observation function, so the downstream ``clip``
    in ``ObsTerm`` only has to deal with finite numbers.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    heights = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
    # nan_to_num: NaN → 0.0, +Inf → posinf (default ~1e38), -Inf → neginf
    heights = torch.nan_to_num(heights, nan=0.0, posinf=1.0, neginf=-1.0)
    return heights


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1)


def diagonal_gait_symmetry(
    env: "ManagerBasedRLEnv",
    fl_rr_cfg: SceneEntityCfg,
    fr_rl_cfg: SceneEntityCfg,
    scale: float = 10.0,
) -> torch.Tensor:
    """Reward diagonal gait symmetry for quadrupeds (FL/RR and FR/RL).

    The term compares corresponding joint trajectories of diagonal leg pairs and
    returns an exponential reward in [0, 1]. Higher values indicate better
    diagonal synchronization (typical trot-like symmetry).
    """
    asset: Articulation = env.scene[fl_rr_cfg.name]

    fl_rr_joint_pos = asset.data.joint_pos[:, fl_rr_cfg.joint_ids]
    fr_rl_joint_pos = asset.data.joint_pos[:, fr_rl_cfg.joint_ids]

    fl_hfe, fl_kfe, rr_hfe, rr_kfe = torch.chunk(fl_rr_joint_pos, chunks=4, dim=1)
    fr_hfe, fr_kfe, rl_hfe, rl_kfe = torch.chunk(fr_rl_joint_pos, chunks=4, dim=1)

    err_fl_rr = torch.square(fl_hfe - rr_hfe) + torch.square(fl_kfe - rr_kfe)
    err_fr_rl = torch.square(fr_hfe - rl_hfe) + torch.square(fr_kfe - rl_kfe)

    mean_err = 0.5 * (err_fl_rr + err_fr_rl).squeeze(1)
    return torch.exp(-scale * mean_err)
