import gymnasium as gym
from . import agents
from .agents import MastiffFlatPPORunnerCfg, MastiffTerrainPPORunnerCfg
##
# Register Gym environments.
##

gym.register(
    id="mastiff-flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mastiff_flat_task:MastiffFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.mastiff_rsl_rl_ppo:MastiffFlatPPORunnerCfg",
    },
)

gym.register(
    id="mastiff-terrain-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mastiff_terrain_task:MastiffTerrainEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.mastiff_rsl_rl_ppo:MastiffTerrainPPORunnerCfg",
    },
)
