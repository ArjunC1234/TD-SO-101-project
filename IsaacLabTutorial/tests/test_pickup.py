"""Grasp-and-lift task contracts. These tests do not launch the simulator."""

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import gymnasium as gym
import pytest
import torch

from isaaclab.managers import EventTermCfg, TerminationTermCfg

from isaaclab_tasks.utils.hydra import resolve_presets

import isaaclab_tutorial.tasks  # noqa: F401  (registers the tasks)
from isaaclab_tutorial.assets import RESET_DATASET
from isaaclab_tutorial.tasks.place_vial.config.so101.agents.rsl_rl_distillation_cfg import (
    SO101CameraDistillationRunnerCfg,
    SO101PickupCameraDistillationRunnerCfg,
)
from isaaclab_tutorial.tasks.place_vial.config.so101.agents.rsl_rl_ppo_cfg import (
    SO101CameraPPORunnerCfg,
    SO101PickupCameraPPORunnerCfg,
    SO101PickupStatePPORunnerCfg,
    SO101StatePPORunnerCfg,
)
from isaaclab_tutorial.tasks.place_vial.config.so101.camera_env_cfg import SO101CameraSceneCfg
from isaaclab_tutorial.tasks.place_vial.config.so101.env_cfg import SO101VialEnvCfg
from isaaclab_tutorial.tasks.place_vial.config.so101.pickup_camera_env_cfg import (
    PickupCameraSim2RealEventsCfg,
    SO101PickupVialCameraEnvCfg,
)
from isaaclab_tutorial.tasks.place_vial.config.so101.pickup_env_cfg import (
    PICKUP_HOME_XY_BOUNDS,
    PickupEventsCfg,
    PickupSim2RealEventsCfg,
    SO101PickupVialEnvCfg,
)
from isaaclab_tutorial.tasks.place_vial.mdp import rewards, terminations
from isaaclab_tutorial.tasks.place_vial.mdp.events import ResetFromDataset
from isaaclab_tutorial.tasks.place_vial.mdp.geometry import tabletop_vial_overlaps_rack
from isaaclab_tutorial.tasks.place_vial.mdp.observations import pickup_progress_flags
from isaaclab_tutorial.tasks.place_vial.mdp.progress import PickupProgress
from isaaclab_tutorial.tasks.place_vial.mdp.task import PICKUP_LIFT_HEIGHT, VIAL_REST_HEIGHT
from isaaclab_tutorial.tasks.place_vial.reset.curriculum import (
    CANONICAL_START,
    PICKUP_HEADING_STAGES,
    PICKUP_HOME_HEADING_NOISE,
    PICKUP_HOME_POSITION_NOISE,
    PICKUP_PHASES,
    widen_home_heading,
)

pytestmark = pytest.mark.unit


def _pickup(*presets):
    return resolve_presets(SO101PickupVialEnvCfg(), ["newton_mjwarp", *presets])


def _pickup_camera(*presets):
    return resolve_presets(SO101PickupVialCameraEnvCfg(), ["newton_mjwarp", "newton_renderer", *presets])


def _event_names(events) -> list[str]:
    names = []
    for name in vars(events):
        if name.startswith("_"):
            continue
        func = getattr(getattr(events, name), "func", None)
        if func is not None:
            names.append(getattr(func, "__name__", type(func).__name__))
    return names


def _heading(quat: torch.Tensor) -> torch.Tensor:
    x, y, z, w = quat.unbind(dim=-1)
    return torch.atan2(2.0 * (y * z - w * x), 2.0 * (x * z + w * y))


def test_pickup_tasks_are_registered_beside_placement():
    state = gym.spec("IsaacTutorial-Pickup-Vial-SO101").kwargs
    camera = gym.spec("IsaacTutorial-Pickup-Vial-SO101-Camera").kwargs
    assert state["env_cfg_entry_point"].endswith("pickup_env_cfg:SO101PickupVialEnvCfg")
    assert state["rsl_rl_cfg_entry_point"].endswith("rsl_rl_ppo_cfg:SO101PickupStatePPORunnerCfg")
    assert camera["env_cfg_entry_point"].endswith("pickup_camera_env_cfg:SO101PickupVialCameraEnvCfg")
    assert camera["rsl_rl_cfg_entry_point"].endswith("rsl_rl_distillation_cfg:SO101PickupCameraDistillationRunnerCfg")
    assert camera["rsl_rl_ppo_cfg_entry_point"].endswith("rsl_rl_ppo_cfg:SO101PickupCameraPPORunnerCfg")
    assert SO101PickupStatePPORunnerCfg().experiment_name != SO101StatePPORunnerCfg().experiment_name
    assert SO101PickupCameraPPORunnerCfg().experiment_name != SO101CameraPPORunnerCfg().experiment_name
    assert (
        SO101PickupCameraDistillationRunnerCfg().experiment_name != SO101CameraDistillationRunnerCfg().experiment_name
    )


def test_pickup_state_task_removes_placement_terms_and_keeps_dataset_reset():
    cfg = _pickup()
    place = resolve_presets(SO101VialEnvCfg(), ["newton_mjwarp"])
    assert type(cfg.events) is PickupEventsCfg
    assert cfg.events.reset_from_dataset.params["phase_weights"] == PICKUP_PHASES
    assert cfg.events.reset_from_dataset.params["home_position_noise"] == pytest.approx(PICKUP_HOME_POSITION_NOISE)
    assert cfg.events.reset_from_dataset.params["home_heading_noise"] == pytest.approx(PICKUP_HOME_HEADING_NOISE)
    assert cfg.events.reset_from_dataset.params["home_xy_bounds"] == PICKUP_HOME_XY_BOUNDS
    assert "reset_root_state_uniform" not in _event_names(cfg.events)
    assert _event_names(cfg.events).count("ResetFromDataset") == 1
    assert cfg.rewards.held_goal is None
    assert cfg.rewards.milestones.func is rewards.pickup_milestones
    assert cfg.rewards.success.params["task"] == "pickup"
    assert cfg.rewards.approach_progress.params["task"] == "pickup"
    assert cfg.rewards.approach_progress.params["height_offset"] == 0.0
    assert cfg.terminations.success.func is terminations.pickup_success
    assert cfg.terminations.vial_dropped.params["task"] == "pickup"
    assert cfg.curriculum.home_heading.params["stages"] == PICKUP_HEADING_STAGES
    assert cfg.curriculum.home_heading.params["stages"][0] == pytest.approx(math.radians(30))
    assert cfg.observations.policy.progress.func is pickup_progress_flags
    assert place.rewards.held_goal is not None
    assert place.terminations.success.func is terminations.placement_success
    assert "home_xy_bounds" not in place.events.reset_from_dataset.params
    assert place.curriculum is None


def test_pickup_sim2real_keeps_stage_one_reset_range():
    cfg = _pickup("sim2real")
    assert type(cfg.events) is PickupSim2RealEventsCfg
    params = cfg.events.reset_from_dataset.params
    assert params["phase_weights"] == PICKUP_PHASES
    assert params["home_position_noise"] == pytest.approx(0.04)
    assert params["home_heading_noise"] == pytest.approx(math.radians(30))
    assert params["home_rack_position_noise"] == pytest.approx(0.005)
    assert cfg.rewards.jaw_support_force is not None
    assert cfg.rewards.held_goal is None
    assert cfg.rewards.approach_progress.params["height_offset"] == pytest.approx(0.008)


def test_pickup_camera_is_wrist_only_and_matches_the_state_teacher():
    nominal = _pickup_camera()
    state = _pickup()
    assert nominal.scene.num_envs == 1024
    assert type(nominal.scene) is SO101CameraSceneCfg
    assert type(nominal.events) is PickupEventsCfg
    assert not any("overhead" in name or name.startswith("top_") for name in vars(nominal.scene))
    image = nominal.observations.wrist_rgb.image.params
    assert image["projection_size"] == (48, 64)
    assert image["history_length"] == 2
    for groups in SO101PickupCameraDistillationRunnerCfg().obs_groups.values():
        assert all(hasattr(nominal.observations, group) for group in groups)
    assert nominal.observations.teacher_state.to_dict() == state.observations.policy.to_dict()
    randomized = _pickup_camera("sim2real")
    assert type(randomized.events) is PickupCameraSim2RealEventsCfg
    assert randomized.events.reset_from_dataset.params["phase_weights"] == PICKUP_PHASES
    assert randomized.events.camera_mount is not None


def test_play_starts_at_home_without_advancing_yaw():
    cfg = _pickup()
    cfg.play_mode()
    assert cfg.curriculum is None
    assert cfg.events.reset_from_dataset.params["sequential"] is True
    assert cfg.events.reset_from_dataset.params["phase_weights"] == CANONICAL_START
    assert cfg.events.reset_from_dataset.params["home_heading_noise"] == pytest.approx(math.radians(30))
    assert cfg.events.reset_from_dataset.params["home_position_noise"] == pytest.approx(0.04)


def _term(monkeypatch, height, fixed, moving, support=0.0):
    count = len(height)
    env = SimpleNamespace(
        num_envs=count,
        device="cpu",
        extras={},
        episode_length_buf=torch.zeros(count, dtype=torch.long),
        step_dt=1.0 / 30.0,
        scene=SimpleNamespace(sensors={}),
        _so101_support_height=torch.full((count,), support),
    )
    monkeypatch.setattr(terminations, "jaw_contacts", lambda env: (fixed, moving))
    monkeypatch.setattr(terminations, "vial_height", lambda env: height)
    monkeypatch.setattr(terminations, "contact_force", lambda env, name: torch.zeros(count))
    return env, terminations.pickup_success(TerminationTermCfg(func=terminations.pickup_success), env)


def test_pickup_success_needs_both_jaws_and_more_than_five_centimetres_for_ten_steps(monkeypatch):
    rest = VIAL_REST_HEIGHT
    height = torch.tensor(
        [
            rest + PICKUP_LIFT_HEIGHT,
            rest + PICKUP_LIFT_HEIGHT + 1.0e-4,
            rest + PICKUP_LIFT_HEIGHT + 0.01,
            rest + 0.007,
        ]
    )
    fixed = torch.ones(4, dtype=torch.bool)
    moving = torch.tensor([True, True, False, True])
    env, term = _term(monkeypatch, height, fixed, moving)
    for _ in range(9):
        env.episode_length_buf += 1
        assert not term(env).any()
    env.episode_length_buf += 1
    assert term(env).tolist() == [False, True, False, False]
    assert term.progress.lifted.tolist() == [False, True, False, False]
    assert term.progress.grasped.tolist() == [True, True, False, True]
    assert term.progress.time_to_success.tolist() == [-1, 10, -1, -1]
    assert env._so101_terminal_hole.tolist() == [-1, -1, -1, -1]
    assert env._so101_terminal_progress.shape == (4, 4)
    assert not env._so101_terminal_progress[:, 2].any()


def test_losing_contact_restarts_the_ten_step_hold(monkeypatch):
    height = torch.tensor([VIAL_REST_HEIGHT + PICKUP_LIFT_HEIGHT + 0.01])
    fixed = torch.ones(1, dtype=torch.bool)
    moving = torch.ones(1, dtype=torch.bool)
    env, term = _term(monkeypatch, height, fixed, moving)
    for _ in range(4):
        env.episode_length_buf += 1
        assert not term(env).item()
    moving[0] = False
    env.episode_length_buf += 1
    assert not term(env).item()
    assert term.progress.stable_count.item() == 0
    moving[0] = True
    for _ in range(9):
        env.episode_length_buf += 1
        assert not term(env).item()
    env.episode_length_buf += 1
    assert term(env).item()
    assert term.progress.time_to_success.item() == 15


def test_mat_height_raises_the_five_centimetre_line(monkeypatch):
    support = 0.01
    height = torch.tensor([VIAL_REST_HEIGHT + PICKUP_LIFT_HEIGHT + support - 1.0e-4])
    fixed = torch.ones(1, dtype=torch.bool)
    moving = torch.ones(1, dtype=torch.bool)
    env, term = _term(monkeypatch, height, fixed, moving, support=support)
    for _ in range(12):
        env.episode_length_buf += 1
        assert not term(env).item()
    height += 2.0e-4
    env.episode_length_buf[:] = 0
    term.progress.reset()
    for _ in range(9):
        env.episode_length_buf += 1
        assert not term(env).item()
    env.episode_length_buf += 1
    assert term(env).item()


def test_pickup_milestones_pay_grasp_and_lift_once(monkeypatch):
    reward = rewards.pickup_milestones.__new__(rewards.pickup_milestones)
    reward.paid = torch.zeros((1, 2), dtype=torch.bool)
    reward.started = torch.zeros(1, dtype=torch.bool)
    progress = SimpleNamespace(grasped=torch.tensor([False]), lifted=torch.tensor([False]))
    monkeypatch.setattr(rewards, "pickup_progress", lambda env: progress)
    assert reward(None).tolist() == [0.0]
    progress.grasped[0] = True
    assert reward(None).tolist() == [1.0]
    progress.lifted[0] = True
    assert reward(None).tolist() == [2.0]
    assert reward(None).tolist() == [0.0]


def test_approach_stops_when_the_pickup_grasp_latches(monkeypatch):
    reward = rewards.approach_progress.__new__(rewards.approach_progress)
    reward.previous_distance = torch.tensor([float("nan")])
    grasped = torch.tensor([False])
    monkeypatch.setattr(rewards, "pickup_progress", lambda env: SimpleNamespace(grasped=grasped))
    monkeypatch.setattr(rewards, "grasp_center_w", lambda env: torch.zeros(1, 3))
    target = torch.tensor([[0.02, 0.0, 0.0]])
    monkeypatch.setattr(rewards, "vial_grasp_point_w", lambda env: target)
    assert reward(None, task="pickup").item() == 0.0
    target[:, 0] = 0.01
    assert reward(None, task="pickup").item() == 1.0
    grasped[0] = True
    target[:, 0] = 0.0
    assert reward(None, task="pickup").item() == 0.0


def test_pickup_progress_hold_resets_without_touching_other_environments():
    state = PickupProgress(2, "cpu", stable_steps=2, grasp_steps=1)
    hold = torch.tensor([True, True])
    state.update(hold, hold, torch.tensor([1, 1]))
    assert state.update(hold, hold, torch.tensor([2, 2])).tolist() == [True, True]
    state.reset(torch.tensor([0]))
    assert state.success.tolist() == [False, True]
    assert state.lifted.tolist() == [False, True]


def _reset_term(count, phase_weights):
    class Scene(dict):
        pass

    rack_pose = torch.tensor([0.18, 0.08, 0.04, 0.0, 0.0, 0.0, 1.0]).repeat(count, 1)
    rack = MagicMock()
    rack.data.default_root_pose.torch = rack_pose
    scene = Scene(robot=MagicMock(), vial=MagicMock(), rack=rack)
    scene.env_origins = torch.zeros(count, 3)
    env = SimpleNamespace(num_envs=count, device="cpu", scene=scene)
    params = {"dataset_path": str(RESET_DATASET), "phase_weights": phase_weights}
    return env, ResetFromDataset(EventTermCfg(func=ResetFromDataset, mode="reset", params=params), env), rack_pose


def test_home_vial_noise_stays_inside_reach_and_pregrasp_rows_stay_put():
    count = 256
    env, term, rack_pose = _reset_term(count, PICKUP_PHASES)
    params = {
        "dataset_path": str(RESET_DATASET),
        "phase_weights": PICKUP_PHASES,
        "home_position_noise": PICKUP_HOME_POSITION_NOISE,
        "home_heading_noise": PICKUP_HOME_HEADING_NOISE,
        "home_rack_clearance": 0.001,
        "home_xy_bounds": PICKUP_HOME_XY_BOUNDS,
    }
    term(env, None, **params, sequential=True)
    rows = env._so101_reset_row
    phase = term.states["phase"][rows]
    original = term.states["vial_pose"][rows]
    actual = env._so101_reset_vial_pose
    home = phase == 0
    assert set(phase.tolist()) == {0, 1}
    assert not tabletop_vial_overlaps_rack(actual[home], rack_pose[home], 0.001).any()
    (x_min, x_max), (y_min, y_max) = PICKUP_HOME_XY_BOUNDS
    assert ((actual[home, 0] >= x_min) & (actual[home, 0] <= x_max)).all()
    assert ((actual[home, 1] >= y_min) & (actual[home, 1] <= y_max)).all()
    assert (actual[home, :2] - original[home, :2]).abs().amax() <= PICKUP_HOME_POSITION_NOISE + 1.0e-5
    torch.testing.assert_close(actual[home, 2], original[home, 2])
    heading_delta = _heading(actual[home, 3:]) - _heading(original[home, 3:])
    heading_delta = torch.atan2(heading_delta.sin(), heading_delta.cos())
    assert heading_delta.abs().amax() <= PICKUP_HOME_HEADING_NOISE + 1.0e-4
    torch.testing.assert_close(actual[~home], original[~home])


def test_out_of_reach_home_noise_falls_back_to_the_dataset_pose():
    env, term, _rack = _reset_term(32, PICKUP_PHASES)
    params = {
        "dataset_path": str(RESET_DATASET),
        "phase_weights": PICKUP_PHASES,
        "home_position_noise": 0.04,
        "home_heading_noise": 0.0,
        "home_xy_bounds": ((0.0, 0.0), (0.0, 0.0)),
    }
    term(env, None, **params, sequential=True)
    rows = env._so101_reset_row
    home = term.states["phase"][rows] == 0
    torch.testing.assert_close(env._so101_reset_vial_pose[home, :2], term.states["vial_pose"][rows][home, :2])


def test_heading_curriculum_starts_at_stage_one_and_widens_only_after_success():
    def _env(success, stepped=True):
        events = SimpleNamespace(params={"home_heading_noise": 0.0, "home_position_noise": 0.04})

        class Manager:
            def get_term_cfg(self, name):
                assert name == "reset_from_dataset"
                return events

            def set_term_cfg(self, name, cfg):
                assert name == "reset_from_dataset"

        count = len(success)
        env = SimpleNamespace(
            num_envs=count,
            device="cpu",
            episode_length_buf=torch.full((count,), int(stepped), dtype=torch.long),
            event_manager=Manager(),
            _so101_pickup_progress=SimpleNamespace(success=torch.tensor(success, dtype=torch.bool)),
            events=events,
        )
        return env

    cold = _env([False, False])
    cold.episode_length_buf[:] = 0
    state = widen_home_heading(cold, None, stages=(0.1, 0.2, 0.3), episodes_per_stage=2, success_threshold=0.5)
    assert state["stage"] == 1.0
    assert cold.events.params["home_heading_noise"] == pytest.approx(0.1)
    assert cold.events.params["home_position_noise"] == pytest.approx(0.04)

    failing = _env([False, False])
    widen_home_heading(failing, None, stages=(0.1, 0.2, 0.3), episodes_per_stage=2, success_threshold=0.5)
    assert failing.events.params["home_heading_noise"] == pytest.approx(0.1)

    learning = _env([True, True])
    first = widen_home_heading(learning, None, stages=(0.1, 0.2, 0.3), episodes_per_stage=2, success_threshold=0.5)
    assert first["stage"] == 2.0
    assert learning.events.params["home_heading_noise"] == pytest.approx(0.2)
    second = widen_home_heading(learning, None, stages=(0.1, 0.2, 0.3), episodes_per_stage=2, success_threshold=0.5)
    assert second["stage"] == 3.0
    third = widen_home_heading(learning, None, stages=(0.1, 0.2, 0.3), episodes_per_stage=2, success_threshold=0.5)
    assert third["stage"] == 3.0
    assert learning.events.params["home_heading_noise"] == pytest.approx(0.3)
    assert learning.events.params["home_position_noise"] == pytest.approx(0.04)
