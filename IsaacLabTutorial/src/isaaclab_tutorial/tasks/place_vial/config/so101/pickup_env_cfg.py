"""SO-101 grasp-and-lift task. The placement task in :mod:`.env_cfg` is unchanged.

Success is both jaws on the vial and the vial root more than 5 cm above its rest height, held for ten control
steps. Resets come from the home and pregrasp rows of the shared dataset. Home rows add ±4 cm of xy noise and a
yaw curriculum that starts at ±30 degrees. There is no second root-state reset on top of that dataset reset.
``presets=sim2real`` stays opt-in.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import preset

from isaaclab_tutorial.tasks.place_vial import mdp
from isaaclab_tutorial.tasks.place_vial.reset.curriculum import (
    PICKUP_HEADING_STAGES,
    PICKUP_HOME_HEADING_NOISE,
    PICKUP_HOME_POSITION_NOISE,
    PICKUP_PHASES,
    widen_home_heading,
)

from .env_cfg import (
    TABLETOP_VIAL_POSITION,
    CriticStateGroupCfg,
    EventsCfg,
    ObservationsCfg,
    PolicyStateGroupCfg,
    RewardsCfg,
    Sim2RealEventsCfg,
    SO101VialEnvCfg,
    TerminationsCfg,
)

# Dataset home poses already span the generator half-range. Stage-1 noise may add ±4 cm, and samples outside this
# window or overlapping the rack are redrawn.
_DATASET_XY_HALF = (0.030, 0.040)
PICKUP_HOME_XY_BOUNDS = (
    (
        TABLETOP_VIAL_POSITION[0] - _DATASET_XY_HALF[0] - PICKUP_HOME_POSITION_NOISE,
        TABLETOP_VIAL_POSITION[0] + _DATASET_XY_HALF[0] + PICKUP_HOME_POSITION_NOISE,
    ),
    (
        TABLETOP_VIAL_POSITION[1] - _DATASET_XY_HALF[1] - PICKUP_HOME_POSITION_NOISE,
        TABLETOP_VIAL_POSITION[1] + _DATASET_XY_HALF[1] + PICKUP_HOME_POSITION_NOISE,
    ),
)


def _apply_pickup_reset(events) -> None:
    """Point the shared dataset reset at home/pregrasp rows and the stage-1 vial range."""
    events.reset_from_dataset.params.update(
        phase_weights=PICKUP_PHASES,
        home_position_noise=PICKUP_HOME_POSITION_NOISE,
        home_heading_noise=PICKUP_HOME_HEADING_NOISE,
        home_rack_clearance=0.001,
        home_xy_bounds=PICKUP_HOME_XY_BOUNDS,
    )


@configclass
class PickupEventsCfg(EventsCfg):
    """Dataset reset only. Home rows are jittered inside the reachable tabletop window."""

    def __post_init__(self):
        _apply_pickup_reset(self)


@configclass
class PickupSim2RealEventsCfg(Sim2RealEventsCfg):
    """Sim2real physics with the pickup phase weights and stage-1 vial range."""

    def __post_init__(self):
        super().__post_init__()
        _apply_pickup_reset(self)


@configclass
class PickupPolicyStateGroupCfg(PolicyStateGroupCfg):
    """State actor inputs, with grasp/lift/success flags instead of the placement insertion flag."""

    progress = ObsTerm(func=mdp.pickup_progress_flags)


@configclass
class PickupCriticStateGroupCfg(CriticStateGroupCfg):
    """Privileged critic. The progress flags match the pickup actor."""

    progress = ObsTerm(func=mdp.pickup_progress_flags)


@configclass
class PickupObservationsCfg(ObservationsCfg):
    policy: PickupPolicyStateGroupCfg = PickupPolicyStateGroupCfg()
    critic: PickupCriticStateGroupCfg = PickupCriticStateGroupCfg()


@configclass
class PickupRewardsCfg(RewardsCfg):
    """Approach, grasp and lift shaping. No transport, insertion or placement-success terms."""

    held_goal = None
    approach_progress = RewTerm(
        func=mdp.approach_progress,
        weight=1.0,
        params={"height_offset": preset(default=0.0, sim2real=0.008), "task": "pickup"},
    )
    milestones = RewTerm(func=mdp.pickup_milestones, weight=10.0)
    success = RewTerm(func=mdp.success, weight=200.0, params={"task": "pickup"})


@configclass
class PickupTerminationsCfg(TerminationsCfg):
    success = DoneTerm(func=mdp.pickup_success)
    vial_dropped = DoneTerm(func=mdp.vial_dropped, params={"task": "pickup"})


@configclass
class PickupCurriculumCfg:
    """Stage 1 is ±30 degrees of yaw. Wider stages run only after the success rate stays high."""

    home_heading = CurrTerm(
        func=widen_home_heading,
        params={
            "stages": PICKUP_HEADING_STAGES,
            "success_threshold": 0.6,
            "episodes_per_stage": 8192,
        },
    )


@configclass
class SO101PickupVialEnvCfg(SO101VialEnvCfg):
    """Grasp and lift a vial from home and pregrasp dataset rows."""

    observations: PickupObservationsCfg = PickupObservationsCfg()
    events: EventsCfg = preset(default=PickupEventsCfg(), sim2real=PickupSim2RealEventsCfg())
    rewards: PickupRewardsCfg = PickupRewardsCfg()
    terminations: PickupTerminationsCfg = PickupTerminationsCfg()
    curriculum: PickupCurriculumCfg = PickupCurriculumCfg()

    def play_mode(self):
        """Evaluate from the home pose at the stage-1 yaw, without advancing the curriculum."""
        super().play_mode()
        self.curriculum = None
