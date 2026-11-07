"""Reset-phase sampling weights over the physically generated task horizon."""

from __future__ import annotations

import math

import torch

# Phase order in the reset dataset:
#   0 home, 1 pregrasp, 2 grasp, 3 lift, 4 reorient, 5 transport, 6 insert, 7 release.
# Every phase is a physics-validated state reached by executing the task from the home pose.

CANONICAL_START = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
"""Evaluation and interactive play start every episode from the real robot's home pose."""

ALL_PHASES = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
"""Training samples every phase uniformly, so each part of the task is practised from the first iteration."""

PICKUP_PHASES = (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
"""Pickup training samples home and pregrasp only. Later rows already hold or place the vial."""

PICKUP_HOME_POSITION_NOISE = 0.04
"""Stage-1 uniform half-width [m] added to home-row vial xy inside :class:`ResetFromDataset`."""

PICKUP_HEADING_STAGES = (math.radians(30), math.radians(45), math.radians(60))
"""Home-row yaw half-widths [rad]. Stage 1 is the default; later stages are wider."""

PICKUP_HOME_HEADING_NOISE = PICKUP_HEADING_STAGES[0]
"""Stage-1 yaw half-width [rad]. The pickup curriculum starts here."""


def _index_envs(env, env_ids):
    if env_ids is None or env_ids == slice(None):
        return torch.arange(env.num_envs, device=env.device)
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device)[env_ids]
    return env_ids


def widen_home_heading(
    env,
    env_ids,
    stages: tuple[float, ...] = PICKUP_HEADING_STAGES,
    success_threshold: float = 0.6,
    episodes_per_stage: int = 8192,
    event_term: str = "reset_from_dataset",
) -> dict[str, float]:
    """Advance home-row yaw after the pickup success rate stays at or above ``success_threshold``.

    Stage 1 is ``stages[0]`` (30 degrees). The position half-width is left alone. The stage is shared by every
    environment because the dataset reset draws one yaw bound for the whole batch. Episodes that have not stepped
    yet are ignored, so the initial reset does not count as failure.
    """
    if len(stages) == 0:
        raise ValueError("stages must contain the stage-1 yaw half-width.")
    if episodes_per_stage <= 0:
        raise ValueError("episodes_per_stage must be positive.")
    state = env.__dict__.setdefault("_so101_pickup_heading_curriculum", {"stage": 0, "successes": 0, "episodes": 0})
    progress = getattr(env, "_so101_pickup_progress", None)
    if progress is not None and state["stage"] < len(stages) - 1:
        ids = _index_envs(env, env_ids)
        counted = ids[env.episode_length_buf[ids] > 0]
        state["episodes"] += int(counted.numel())
        state["successes"] += int(progress.success[counted].sum().item())
        if state["episodes"] >= episodes_per_stage:
            if state["successes"] / state["episodes"] >= success_threshold:
                state["stage"] += 1
            state["successes"] = 0
            state["episodes"] = 0
    heading_noise = float(stages[state["stage"]])
    term = env.event_manager.get_term_cfg(event_term)
    term.params["home_heading_noise"] = heading_noise
    env.event_manager.set_term_cfg(event_term, term)
    return {"stage": float(state["stage"] + 1), "heading_noise_rad": heading_noise}
