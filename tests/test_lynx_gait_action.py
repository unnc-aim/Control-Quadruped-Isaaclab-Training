import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
MDP_DIR = ROOT / "tasks" / "mdp"


def _load_action_module():
    isaaclab_pkg = sys.modules.setdefault("isaaclab", types.ModuleType("isaaclab"))
    assets_pkg = sys.modules.setdefault("isaaclab.assets", types.ModuleType("isaaclab.assets"))
    articulation_mod = sys.modules.setdefault(
        "isaaclab.assets.articulation", types.ModuleType("isaaclab.assets.articulation")
    )
    managers_pkg = sys.modules.setdefault("isaaclab.managers", types.ModuleType("isaaclab.managers"))
    action_mod = sys.modules.setdefault(
        "isaaclab.managers.action_manager", types.ModuleType("isaaclab.managers.action_manager")
    )
    utils_mod = sys.modules.setdefault("isaaclab.utils", types.ModuleType("isaaclab.utils"))

    class Articulation:
        pass

    class ActionTerm:
        pass

    class ActionTermCfg:
        pass

    articulation_mod.Articulation = Articulation
    action_mod.ActionTerm = ActionTerm
    action_mod.ActionTermCfg = ActionTermCfg
    utils_mod.configclass = lambda cls: cls
    assets_pkg.articulation = articulation_mod
    managers_pkg.action_manager = action_mod
    isaaclab_pkg.assets = assets_pkg
    isaaclab_pkg.managers = managers_pkg
    isaaclab_pkg.utils = utils_mod

    tasks_pkg = types.ModuleType("tasks")
    tasks_pkg.__path__ = [str(ROOT / "tasks")]
    mdp_pkg = types.ModuleType("tasks.mdp")
    mdp_pkg.__path__ = [str(MDP_DIR)]
    sys.modules["tasks"] = tasks_pkg
    sys.modules["tasks.mdp"] = mdp_pkg

    generator_spec = importlib.util.spec_from_file_location(
        "tasks.mdp.lynx_gait_generator", MDP_DIR / "lynx_gait_generator.py"
    )
    generator_module = importlib.util.module_from_spec(generator_spec)
    sys.modules[generator_spec.name] = generator_module
    assert generator_spec.loader is not None
    generator_spec.loader.exec_module(generator_module)

    action_spec = importlib.util.spec_from_file_location(
        "tasks.mdp.lynx_gait_action", MDP_DIR / "lynx_gait_action.py"
    )
    action_module = importlib.util.module_from_spec(action_spec)
    sys.modules[action_spec.name] = action_module
    assert action_spec.loader is not None
    action_spec.loader.exec_module(action_module)
    return action_module


MODULE = _load_action_module()
LynxGaitAction = MODULE.LynxGaitAction


def _make_action(num_envs: int = 1):
    action = object.__new__(LynxGaitAction)
    action.num_envs = num_envs
    action._leg_count = 4
    action._rl_action_dim = 12
    action._raw_actions = torch.zeros(num_envs, 12)
    action._length_residual = torch.zeros(num_envs, 4)
    action._height_residual = torch.zeros(num_envs, 4)
    action._trajectory_z_residual = torch.zeros(num_envs, 4)
    action.cfg = types.SimpleNamespace(
        step_length_residual_scale=0.02,
        step_height_residual_scale=0.03,
        trajectory_z_residual_scale=0.10,
        trajectory_z_min=-0.05,
        trajectory_z_max=0.10,
    )
    return action


def test_action_layout_is_length_height_trajectory_per_leg() -> None:
    action = _make_action()
    values = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, -1.0, -0.5]])

    action.process_actions(values)

    torch.testing.assert_close(action._length_residual, torch.tensor([[0.002, 0.008, 0.014, 0.020]]))
    torch.testing.assert_close(action._height_residual, torch.tensor([[0.006, 0.015, 0.024, -0.030]]))
    torch.testing.assert_close(action._trajectory_z_residual, torch.tensor([[0.030, 0.060, 0.090, -0.050]]))
    assert action.action_dim == 12
    assert action.ACTION_FIELDS == ("length", "step_height", "trajectory_z")


def test_action_rejects_non_12_dimensional_input() -> None:
    action = _make_action()
    with pytest.raises(RuntimeError, match="shape"):
        action.process_actions(torch.zeros(1, 8))


def test_trajectory_z_clamps_to_configured_range() -> None:
    action = _make_action()
    raw = torch.tensor([[-0.20, -0.04, 0.03, 0.20]])
    torch.testing.assert_close(
        action._clamp_trajectory_z(raw), torch.tensor([[-0.05, -0.04, 0.03, 0.10]])
    )


def test_invalid_ik_falls_back_per_leg_only() -> None:
    planner = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    standing = torch.full_like(planner, -1.0)
    valid = torch.tensor([[True, False, True, False]])

    selected = LynxGaitAction._fallback_invalid_targets(planner, valid, standing)

    torch.testing.assert_close(selected[:, 0], planner[:, 0])
    torch.testing.assert_close(selected[:, 1], standing[:, 1])
    torch.testing.assert_close(selected[:, 2], planner[:, 2])
    torch.testing.assert_close(selected[:, 3], standing[:, 3])
