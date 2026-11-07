"""SO-101 state and wrist-camera tasks. Add ``presets=sim2real`` to either for robot-transfer randomization.

The pickup ids are grasp-and-lift only. The placement ids are unchanged.
"""

import gymnasium as gym

_PACKAGE = "isaaclab_tutorial.tasks.place_vial.config.so101"

gym.register(
    id="IsaacTutorial-Place-Vial-SO101",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PACKAGE}.env_cfg:SO101VialEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PACKAGE}.agents.rsl_rl_ppo_cfg:SO101StatePPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="IsaacTutorial-Place-Vial-SO101-Camera",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PACKAGE}.camera_env_cfg:SO101VialCameraEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PACKAGE}.agents.rsl_rl_distillation_cfg:SO101CameraDistillationRunnerCfg",
        "rsl_rl_ppo_cfg_entry_point": f"{_PACKAGE}.agents.rsl_rl_ppo_cfg:SO101CameraPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="IsaacTutorial-Pickup-Vial-SO101",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PACKAGE}.pickup_env_cfg:SO101PickupVialEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PACKAGE}.agents.rsl_rl_ppo_cfg:SO101PickupStatePPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="IsaacTutorial-Pickup-Vial-SO101-Camera",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PACKAGE}.pickup_camera_env_cfg:SO101PickupVialCameraEnvCfg",
        "rsl_rl_cfg_entry_point": (f"{_PACKAGE}.agents.rsl_rl_distillation_cfg:SO101PickupCameraDistillationRunnerCfg"),
        "rsl_rl_ppo_cfg_entry_point": f"{_PACKAGE}.agents.rsl_rl_ppo_cfg:SO101PickupCameraPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
