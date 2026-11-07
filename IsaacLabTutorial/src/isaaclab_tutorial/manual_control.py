"""Interactive terminal control for the SO-101 vial-placement simulation."""

from __future__ import annotations

import argparse
import json
import math
import queue
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from isaaclab_tutorial.so101_joints import JOINT_INDEX, JOINTS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

MAX_PULSE_TICKS = 3_000
DEFAULT_GRIPPER_TICKS = 30

HELP_TEXT = """Commands:
  help                              Show this help.
  state                             Print joint, end-effector, vial, and reward state.
  action A0 A1 A2 A3 A4 A5         Hold a normalized six-joint action until stopped.
  pulse JOINT VALUE TICKS           Apply one joint action for a fixed number of 30 Hz ticks.
  open [TICKS]                      Open the gripper for 30 ticks by default.
  close [TICKS]                     Close the gripper for 30 ticks by default.
  stop                              Send zero action and hold the measured pose.
  reset                             Reset to the next canonical home pose.
  quit                              Close the simulation.

Joint order for `action`:
  shoulder_pan shoulder_lift elbow_flex wrist_flex wrist_roll gripper

Examples:
  pulse shoulder_pan 0.5 30
  action 0 -0.25 0 0 0 0
  close 20
"""


class CommandError(ValueError):
    """Raised when a manual-control command is malformed."""


CommandKind = Literal["help", "state", "action", "pulse", "stop", "reset", "quit"]


@dataclass(frozen=True)
class ManualCommand:
    """A validated command that can be consumed by the simulation thread."""

    kind: CommandKind
    values: tuple[float, ...] = ()
    joint: str | None = None
    value: float = 0.0
    ticks: int | None = None


def _finite_float(text: str, label: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise CommandError(f"{label} must be a number, got {text!r}") from exc
    if not math.isfinite(value):
        raise CommandError(f"{label} must be finite")
    return value


def _ticks(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise CommandError(f"ticks must be an integer, got {text!r}") from exc
    if not 1 <= value <= MAX_PULSE_TICKS:
        raise CommandError(f"ticks must be between 1 and {MAX_PULSE_TICKS}")
    return value


def parse_command(line: str) -> ManualCommand:
    """Parse one terminal command without evaluating arbitrary Python."""

    parts = line.strip().split()
    if not parts:
        raise CommandError("enter a command, or type 'help'")
    name, *args = parts
    name = name.lower()
    aliases = {"status": "state", "home": "reset", "exit": "quit"}
    name = aliases.get(name, name)

    if name in {"help", "state", "stop", "reset", "quit"}:
        if args:
            raise CommandError(f"{name} does not accept arguments")
        return ManualCommand(name)  # type: ignore[arg-type]

    if name == "action":
        if len(args) != len(JOINTS):
            raise CommandError(f"action requires exactly {len(JOINTS)} values")
        values = tuple(_finite_float(value, f"action[{index}]") for index, value in enumerate(args))
        return ManualCommand("action", values=values)

    if name == "pulse":
        if len(args) != 3:
            raise CommandError("pulse requires: pulse JOINT VALUE TICKS")
        joint = args[0].lower()
        if joint not in JOINT_INDEX:
            raise CommandError(f"unknown joint {joint!r}; choose one of: {', '.join(JOINTS)}")
        return ManualCommand(
            "pulse",
            joint=joint,
            value=_finite_float(args[1], "pulse value"),
            ticks=_ticks(args[2]),
        )

    if name in {"open", "close"}:
        if len(args) > 1:
            raise CommandError(f"{name} accepts at most one TICKS argument")
        ticks = _ticks(args[0]) if args else DEFAULT_GRIPPER_TICKS
        return ManualCommand(
            "pulse",
            joint="gripper",
            value=1.0 if name == "open" else -1.0,
            ticks=ticks,
        )

    raise CommandError(f"unknown command {name!r}; type 'help' for available commands")


class ActionState:
    """The normalized action currently being applied to the simulation."""

    def __init__(self) -> None:
        self._values = [0.0] * len(JOINTS)
        self._remaining_ticks: int | None = None

    @staticmethod
    def _clamp(value: float) -> float:
        return max(-1.0, min(1.0, value))

    @property
    def values(self) -> tuple[float, ...]:
        return tuple(self._values)

    @property
    def remaining_ticks(self) -> int | None:
        return self._remaining_ticks

    def hold(self, values: tuple[float, ...]) -> None:
        if len(values) != len(JOINTS):
            raise ValueError(f"expected {len(JOINTS)} action values")
        self._values = [self._clamp(value) for value in values]
        self._remaining_ticks = None

    def pulse(self, joint: str, value: float, ticks: int) -> None:
        if joint not in JOINT_INDEX:
            raise ValueError(f"unknown joint: {joint}")
        if not 1 <= ticks <= MAX_PULSE_TICKS:
            raise ValueError(f"ticks must be between 1 and {MAX_PULSE_TICKS}")
        self.stop()
        self._values[JOINT_INDEX[joint]] = self._clamp(value)
        self._remaining_ticks = ticks

    def stop(self) -> None:
        self._values = [0.0] * len(JOINTS)
        self._remaining_ticks = None

    def after_step(self) -> None:
        if self._remaining_ticks is None:
            return
        self._remaining_ticks -= 1
        if self._remaining_ticks == 0:
            self.stop()


def _input_worker(commands: queue.Queue[ManualCommand], stopping: threading.Event) -> None:
    """Read stdin without ever touching the Isaac Lab environment."""

    while not stopping.is_set():
        try:
            line = input("so101> ")
        except (EOFError, KeyboardInterrupt):
            commands.put(ManualCommand("quit"))
            return
        try:
            commands.put(parse_command(line))
        except CommandError as exc:
            print(f"Error: {exc}", flush=True)


def snapshot_state(env: ManagerBasedRLEnv, reward: torch.Tensor | None = None) -> dict:
    """Return one JSON-serializable snapshot for display and future MCP reuse."""

    from isaaclab_tutorial.tasks.place_vial.mdp.task import (
        fingertip_positions_w,
        grasp_center_w,
        jaw_contacts,
        vial_grasp_point_w,
    )

    robot = env.scene["robot"]
    joint_ids = robot.find_joints(list(JOINTS), preserve_order=True)[0]
    positions = robot.data.joint_pos.torch[0, joint_ids]
    velocities = robot.data.joint_vel.torch[0, joint_ids]
    targets = robot.data.joint_pos_target.torch[0, joint_ids]
    end_effector = grasp_center_w(env)[0]
    vial = env.scene["vial"].data.root_pos_w.torch[0]
    fixed, moving = fingertip_positions_w(env)
    fixed_contact, moving_contact = jaw_contacts(env)
    return {
        "vial_grasp_point_m": vial_grasp_point_w(env)[0].tolist(),
        "vial_quaternion_xyzw": env.scene["vial"].data.root_quat_w.torch[0].tolist(),
        "fixed_fingertip_m": fixed[0].tolist(),
        "moving_fingertip_m": moving[0].tolist(),
        "jaw_contacts": {"fixed": bool(fixed_contact[0]), "moving": bool(moving_contact[0])},
        "joint_limits_rad": robot.data.soft_joint_pos_limits.torch[0, joint_ids].tolist(),
        "joints": {
            name: {
                "position_rad": float(positions[index]),
                "target_rad": float(targets[index]),
                "velocity_rad_s": float(velocities[index]),
            }
            for index, name in enumerate(JOINTS)
        },
        "end_effector_position_m": [float(value) for value in end_effector],
        "vial_position_m": [float(value) for value in vial],
        "reward": float(reward[0]) if reward is not None and reward.numel() else 0.0,
    }


def format_state(snapshot: dict) -> str:
    """Format a state snapshot for a human-readable terminal display."""

    lines = ["Joint                 Position      Target      Velocity"]
    for name in JOINTS:
        joint = snapshot["joints"][name]
        lines.append(
            f"{name:<20} {math.degrees(joint['position_rad']):>8.2f} deg"
            f" {math.degrees(joint['target_rad']):>8.2f} deg"
            f" {math.degrees(joint['velocity_rad_s']):>9.2f} deg/s"
        )
    end_effector = snapshot["end_effector_position_m"]
    vial = snapshot["vial_position_m"]
    lines.extend(
        (
            "",
            f"End effector [m]: x={end_effector[0]:.3f}, y={end_effector[1]:.3f}, z={end_effector[2]:.3f}",
            f"Vial [m]:         x={vial[0]:.3f}, y={vial[1]:.3f}, z={vial[2]:.3f}",
            f"Reward: {snapshot['reward']:.4f}",
        )
    )
    return "\n".join(lines)


def configure_environment(args):
    from isaaclab_tasks.utils.hydra import resolve_presets

    from isaaclab_tutorial.tasks.place_vial.config.so101.env_cfg import SO101VialEnvCfg
    from isaaclab_tutorial.tasks.place_vial.reset.curriculum import CANONICAL_START

    presets = [preset.strip() for preset in args.presets.split(",") if preset.strip()]
    env_cfg = resolve_presets(SO101VialEnvCfg(), presets)
    env_cfg.play_mode()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args.device
    env_cfg.seed = args.seed
    env_cfg.rewards = None
    env_cfg.terminations = None
    env_cfg.curriculum = None
    env_cfg.events.reset_from_dataset.params.update(sequential=True, phase_weights=CANONICAL_START)
    return env_cfg


def normalized_action_tensor(env: ManagerBasedRLEnv, values: tuple[float, ...] | list[float]) -> torch.Tensor:
    """Build one validated, clamped action tensor in the canonical SO-101 joint order."""

    if len(values) != len(JOINTS):
        raise ValueError(f"expected {len(JOINTS)} action values, got {len(values)}")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("all action values must be finite")
    clamped = [max(-1.0, min(1.0, value)) for value in values]
    return torch.tensor([clamped], dtype=torch.float32, device=env.device)


def _handle_command(
    command: ManualCommand,
    env: ManagerBasedRLEnv,
    action_state: ActionState,
    reward: torch.Tensor | None,
) -> bool:
    """Apply a command on the simulation thread. Return false to request shutdown."""

    if command.kind == "help":
        print(HELP_TEXT, flush=True)
    elif command.kind == "state":
        print(format_state(snapshot_state(env, reward)), flush=True)
    elif command.kind == "action":
        action_state.hold(command.values)
        print(f"Holding action: {action_state.values}", flush=True)
    elif command.kind == "pulse":
        assert command.joint is not None and command.ticks is not None
        action_state.pulse(command.joint, command.value, command.ticks)
        applied_value = action_state.values[JOINT_INDEX[command.joint]]
        print(f"Pulsing {command.joint} at {applied_value:.3f} for {command.ticks} ticks.", flush=True)
    elif command.kind == "stop":
        action_state.stop()
        print("Action stopped; holding the measured pose.", flush=True)
    elif command.kind == "reset":
        action_state.stop()
        env.reset()
        print("Environment reset to a canonical home pose.", flush=True)
    elif command.kind == "quit":
        return False
    return True


def _run_interactive(env: ManagerBasedRLEnv, *, unthrottled: bool) -> int:
    action_state = ActionState()
    commands: queue.Queue[ManualCommand] = queue.Queue()
    stopping = threading.Event()
    input_thread = threading.Thread(target=_input_worker, args=(commands, stopping), daemon=True)
    input_thread.start()
    reward: torch.Tensor | None = None
    running = True
    deadline = time.perf_counter()
    print(HELP_TEXT, flush=True)

    try:
        while running and env.sim.is_running():
            while True:
                try:
                    command = commands.get_nowait()
                except queue.Empty:
                    break
                running = _handle_command(command, env, action_state, reward)
                if not running:
                    break
            if not running:
                break

            action = normalized_action_tensor(env, action_state.values)
            _, reward, _, _, _ = env.step(action)
            action_state.after_step()

            if not unthrottled:
                deadline += env.step_dt
                delay = deadline - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    deadline = time.perf_counter()
    finally:
        stopping.set()
        action_state.stop()
    return 0


def _run_smoke_test(env: ManagerBasedRLEnv) -> int:
    """Exercise the same action path headlessly and emit a machine-readable result."""

    robot = env.scene["robot"]
    joint_ids = robot.find_joints(list(JOINTS), preserve_order=True)[0]
    before = robot.data.joint_pos.torch[0, joint_ids].clone()
    action = torch.zeros((1, len(JOINTS)), dtype=torch.float32, device=env.device)
    action[0, JOINT_INDEX["shoulder_pan"]] = 0.5
    reward = None
    for _ in range(30):
        _, reward, _, _, _ = env.step(action)
    for _ in range(10):
        _, reward, _, _, _ = env.step(torch.zeros_like(action))
    after = robot.data.joint_pos.torch[0, joint_ids].clone()
    delta = after - before
    limits = robot.data.soft_joint_pos_limits.torch[0, joint_ids]
    finite = bool(torch.isfinite(after).all())
    within_limits = bool(((after >= limits[:, 0] - 1e-4) & (after <= limits[:, 1] + 1e-4)).all())
    moved = float(delta[JOINT_INDEX["shoulder_pan"]]) > 0.01
    result = {
        "ok": finite and within_limits and moved,
        "finite": finite,
        "within_limits": within_limits,
        "shoulder_pan_delta_rad": float(delta[JOINT_INDEX["shoulder_pan"]]),
        "state": snapshot_state(env, reward),
    }
    print(f"MANUAL_CONTROL_SMOKE_RESULT={json.dumps(result, sort_keys=True)}", flush=True)
    return 0 if result["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    """Launch one SO-101 environment and control it from a terminal."""

    from isaaclab.app import add_launcher_args, launch_simulation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presets", default="newton_mjwarp", help="Comma-separated Isaac Lab presets.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unthrottled", action="store_true", help="Run faster than wall-clock time.")
    parser.add_argument("--smoke-test", action="store_true", help="Run a bounded headless joint-motion check and exit.")
    add_launcher_args(parser)
    parser.set_defaults(visualizer="newton")
    args = parser.parse_args(argv)

    from isaaclab.envs import ManagerBasedRLEnv

    env_cfg = configure_environment(args)
    with launch_simulation(env_cfg, args):
        env = ManagerBasedRLEnv(env_cfg)
        try:
            env.reset()
            return _run_smoke_test(env) if args.smoke_test else _run_interactive(env, unthrottled=args.unthrottled)
        finally:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
