"""Wrist-camera grasp-and-lift task, distilled from the pickup state teacher.

The student sees the same wrist RGB history and proprioception as the placement camera task. There is no overhead
camera. ``presets=sim2real`` stays opt-in; appearance randomization is part of that preset only.
"""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import preset

from .camera_env_cfg import CameraObservationsCfg, CameraSim2RealEventsCfg, SO101CameraSceneCfg
from .env_cfg import EventsCfg
from .pickup_env_cfg import (
    PickupCriticStateGroupCfg,
    PickupEventsCfg,
    PickupPolicyStateGroupCfg,
    SO101PickupVialEnvCfg,
    _apply_pickup_reset,
)


@configclass
class PickupCameraObservationsCfg(CameraObservationsCfg):
    """Wrist RGB, proprioception, the pickup state teacher, and the privileged critic."""

    teacher_state: PickupPolicyStateGroupCfg = PickupPolicyStateGroupCfg()
    critic: PickupCriticStateGroupCfg = PickupCriticStateGroupCfg()


@configclass
class PickupCameraSim2RealEventsCfg(CameraSim2RealEventsCfg):
    """Sim2real physics and appearance, with the pickup dataset reset."""

    def __post_init__(self):
        super().__post_init__()
        _apply_pickup_reset(self)


@configclass
class SO101PickupVialCameraEnvCfg(SO101PickupVialEnvCfg):
    """Pickup from wrist RGB and proprioception, distilled from the pickup state teacher."""

    scene: SO101CameraSceneCfg = SO101CameraSceneCfg(num_envs=1024, env_spacing=0.9, replicate_physics=True)
    observations: PickupCameraObservationsCfg = PickupCameraObservationsCfg()
    events: EventsCfg = preset(default=PickupEventsCfg(), sim2real=PickupCameraSim2RealEventsCfg())

    def play_mode(self):
        from isaaclab_tutorial.utils import evaluation

        super().play_mode()
        if not evaluation.EXACT_EVALUATION_ACTIVE:
            self.scene.num_envs = min(self.scene.num_envs, 8)
