"""VLA pipeline V1 envs.

Registers the HC10DT + Robotiq 2F-85 cube pick-place task with Isaac Lab's
gymnasium registry when imported.
"""
import gymnasium as gym

from .yaskawa_pick_cube_cfg import YaskawaPickCubeIkRelEnvCfg

gym.register(
    id="Isaac-PickCube-HC10DT-Robotiq-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": YaskawaPickCubeIkRelEnvCfg},
)
