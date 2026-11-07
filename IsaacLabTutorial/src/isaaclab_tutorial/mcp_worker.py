"""Private Isaac Lab worker process for the SO-101 MCP server.

The worker speaks authenticated ``multiprocessing.connection`` messages with the
MCP front process. Keeping Isaac in a child process prevents simulator log output
from corrupting the MCP server's STDIO protocol.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
import traceback
from multiprocessing.connection import Client
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _response_error(exc: BaseException) -> dict[str, Any]:
    return {
        "ok": False,
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
    }


class SimulatorWorker:
    """Own the Isaac Lab application and environment on this process's main thread."""

    def __init__(self, *, visualizer: str, device: str, presets: str, seed: int, scene_setup=None) -> None:
        from isaaclab.app import add_launcher_args, launch_simulation
        from isaaclab.envs import ManagerBasedRLEnv

        from isaaclab_tutorial.manual_control import configure_environment

        launcher_parser = argparse.ArgumentParser(add_help=False)
        add_launcher_args(launcher_parser)
        launcher_args = launcher_parser.parse_args(["--device", device, "--visualizer", visualizer])
        config_args = SimpleNamespace(presets=presets, device=device, seed=seed)
        env_cfg = configure_environment(config_args)
        if scene_setup is not None:
            scene_setup(env_cfg)

        self._launch_context = launch_simulation(env_cfg, launcher_args)
        self._launch_context.__enter__()
        try:
            self.env = ManagerBasedRLEnv(env_cfg)
            self.env.reset()
            self.capture_hold()
        except BaseException:
            self._launch_context.__exit__(*__import__("sys").exc_info())
            raise

    def state(self, reward=None) -> dict[str, Any]:
        from isaaclab_tutorial.manual_control import snapshot_state

        return snapshot_state(self.env, reward)

    def apply_action(self, values: list[float], ticks: int) -> dict[str, Any]:
        from isaaclab_tutorial.manual_control import normalized_action_tensor

        if not self.env.sim.is_running():
            raise RuntimeError("the simulator window has been closed")
        if not 1 <= ticks <= 3_000:
            raise ValueError("ticks must be between 1 and 3000")
        if len(values) != 6:
            raise ValueError("actions must contain exactly 6 values")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("all action values must be finite")

        action = normalized_action_tensor(self.env, values)
        reward = None
        for _ in range(ticks):
            _, reward, _, _, _ = self.env.step(action)

        self.capture_hold()
        self.idle_step()
        return self.state(reward)

    def reset(self) -> dict[str, Any]:
        self.env.reset()
        self.capture_hold()
        return self.state()

    def capture_hold(self):
        self.hold_target = self.env.scene["robot"].data.joint_pos_target.torch.clone()

    def position_tick(self, target):
        """Step physics with a fixed absolute servo target, bypassing relative action rewriting."""
        import torch

        robot = self.env.scene["robot"]
        for _ in range(self.env.cfg.decimation):
            robot.set_joint_position_target_index(target=target)
            robot.set_joint_velocity_target_index(target=torch.zeros_like(target))
            self.env.scene.write_data_to_sim()
            self.env.sim.step()
            self.env.scene.update(self.env.physics_dt)

    def move_positions(self, positions, ticks=60):
        import torch

        from isaaclab_tutorial.so101_joints import JOINTS

        if len(positions) != 6 or not all(math.isfinite(v) for v in positions):
            raise ValueError("positions must contain six finite radians")
        if not isinstance(ticks, int) or not 1 <= ticks <= 3000:
            raise ValueError("ticks must be an integer in [1, 3000]")
        robot = self.env.scene["robot"]
        ids = robot.find_joints(list(JOINTS), preserve_order=True)[0]
        target = self.hold_target.clone()
        desired = torch.tensor(positions, device=self.env.device)
        limits = robot.data.soft_joint_pos_limits.torch[0, ids]
        if bool(((desired < limits[:, 0]) | (desired > limits[:, 1])).any()):
            raise ValueError("requested joint position exceeds model soft limits")
        target[0, ids] = desired
        start = robot.data.joint_pos.torch.clone()
        for tick in range(1, ticks + 1):
            if not self.env.sim.is_running():
                raise RuntimeError("viewer closed during motion")
            self.hold_target = start + (target - start) * (tick / ticks)
            self.position_tick(self.hold_target)
        for _ in range(15):
            self.idle_step()
        # Bounded endpoint correction compensates static servo/load error in free space.
        # Contact can prevent arrival: report the residual, never declare success from a command alone.
        for _ in range(10):
            error = desired - robot.data.joint_pos.torch[0, ids]
            if float(error.abs().max()) <= 0.005:
                break
            self.hold_target[0, ids] = torch.clamp(
                self.hold_target[0, ids] + 0.7 * error.clamp(-0.03, 0.03),
                min=limits[:, 0],
                max=limits[:, 1],
            )
            for _ in range(15):
                self.idle_step()
        result = self.state()
        result["command_error_rad"] = (desired - robot.data.joint_pos.torch[0, ids]).tolist()
        result["reached_joint_tolerance"] = max(abs(v) for v in result["command_error_rad"]) <= 0.005
        return result

    def idle_step(self):
        """Refresh the viewer while retaining the last absolute servo targets."""
        return self.position_tick(self.hold_target)

    def close(self) -> None:
        try:
            self.env.close()
        finally:
            self._launch_context.__exit__(None, None, None)


def _serve(connection, options: argparse.Namespace) -> int:
    worker: SimulatorWorker | None = None
    try:
        connection.send({"event": "connected", "pid": os.getpid()})
        worker = SimulatorWorker(
            visualizer=options.visualizer,
            device=options.device,
            presets=options.presets,
            seed=options.seed,
        )
        connection.send({"ok": True, "result": {"state": worker.state(), "pid": os.getpid()}})

        # Local fallback for clients whose cached MCP tool list predates absolute motion.
        mailbox = Path(__file__).resolve().parents[2] / "outputs" / "so101_control"
        inbox = mailbox / "request.json"
        seen = json.loads(inbox.read_text())["id"] if inbox.exists() else None
        deadline = time.perf_counter()
        while worker.env.sim.is_running():
            while connection.poll():
                request = connection.recv()
                operation = request.get("operation")
                try:
                    if operation == "state":
                        result = worker.state()
                    elif operation == "action":
                        result = worker.apply_action(request["values"], request["ticks"])
                    elif operation == "reset":
                        result = worker.reset()
                    elif operation == "positions":
                        result = worker.move_positions(request["positions"], request["ticks"])
                    elif operation == "shutdown":
                        connection.send({"ok": True, "result": {"stopped": True}})
                        return 0
                    else:
                        raise ValueError(f"unknown worker operation: {operation!r}")
                    connection.send({"ok": True, "result": result})
                except BaseException as exc:
                    connection.send(_response_error(exc))

            if inbox.exists():
                try:
                    local_request = json.loads(inbox.read_text())
                    if local_request["id"] != seen:
                        seen = local_request["id"]
                        try:
                            result = worker.move_positions(
                                local_request["positions_rad"], local_request.get("ticks", 90)
                            )
                            response = {"id": seen, "ok": True, "state": result}
                        except Exception as exc:
                            response = {"id": seen, **_response_error(exc)}
                        temporary = mailbox / "response.tmp"
                        temporary.write_text(json.dumps(response))
                        temporary.replace(mailbox / "response.json")
                except (json.JSONDecodeError, KeyError):
                    pass  # An incomplete write is retried on the next tick.

            worker.idle_step()
            deadline += worker.env.step_dt
            delay = deadline - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                deadline = time.perf_counter()
        return 0
    except EOFError:
        return 0
    except BaseException as exc:
        with contextlib.suppress(BaseException):
            connection.send(_response_error(exc))
        return 1
    finally:
        if worker is not None:
            worker.close()
        connection.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--auth-key", required=True)
    parser.add_argument("--visualizer", choices=("newton", "none"), default="newton")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--presets", default="newton_mjwarp")
    parser.add_argument("--seed", type=int, default=42)
    options = parser.parse_args(argv)
    connection = Client((options.host, options.port), authkey=bytes.fromhex(options.auth_key))
    return _serve(connection, options)


if __name__ == "__main__":
    raise SystemExit(main())
