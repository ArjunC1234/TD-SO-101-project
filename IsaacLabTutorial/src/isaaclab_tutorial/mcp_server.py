"""MCP tools for controlling the SO-101 Isaac Lab simulation from Codex."""

from __future__ import annotations

import atexit
import math
import secrets
import subprocess
import sys
import tempfile
import threading
import warnings
from multiprocessing.connection import Connection, Listener, wait
from pathlib import Path
from typing import Any, Literal

warnings.filterwarnings("ignore", message="Field 'lifespan' has an incomplete definition.*")

# The filter must be installed before importing FastMCP, which emits this upstream warning.
from mcp.server.fastmcp import FastMCP  # noqa: E402
from mcp.types import ToolAnnotations  # noqa: E402

from isaaclab_tutorial.so101_joints import JOINT_INDEX, JOINTS  # noqa: E402

JointName = Literal["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
VisualizerName = Literal["newton", "none"]

MAX_TOOL_TICKS = 3_000
DEFAULT_GRIPPER_TICKS = 30
WORKER_START_TIMEOUT_SECONDS = 180
WORKER_TOOL_TIMEOUT_SECONDS = 180


class WorkerError(RuntimeError):
    """Raised when the Isaac worker cannot complete a request."""


class WorkerClient:
    """Manage one authenticated local worker and serialize all simulator calls."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._process: subprocess.Popen | None = None
        self._connection: Connection | None = None
        self._log_handle = None
        self.log_path = Path(tempfile.gettempdir()) / "so101_mcp_worker.log"
        self.visualizer: str | None = None

    @property
    def running(self) -> bool:
        return (
            self._process is not None
            and self._process.poll() is None
            and self._connection is not None
            and not self._connection.closed
        )

    @staticmethod
    def _receive(connection: Connection, timeout: float) -> dict[str, Any]:
        if not wait([connection], timeout=timeout):
            raise WorkerError(f"simulator worker timed out after {timeout:.0f} seconds")
        response = connection.recv()
        if not isinstance(response, dict):
            raise WorkerError("simulator worker returned an invalid response")
        return response

    def start(
        self,
        *,
        visualizer: VisualizerName = "newton",
        device: str = "cuda:0",
        presets: str = "newton_mjwarp",
        seed: int = 42,
    ) -> dict[str, Any]:
        with self._lock:
            if self.running:
                if visualizer != self.visualizer:
                    raise WorkerError(
                        f"simulator is already running with visualizer={self.visualizer!r}; stop it before changing"
                    )
                return {"already_running": True, "visualizer": self.visualizer, "state": self.request("state")}

            self.stop()
            auth_key = secrets.token_bytes(32)
            listener = Listener(("127.0.0.1", 0), authkey=auth_key)
            host, port = listener.address
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = self.log_path.open("a", encoding="utf-8")
            command = [
                sys.executable,
                "-m",
                "isaaclab_tutorial.mcp_worker",
                "--host",
                host,
                "--port",
                str(port),
                "--auth-key",
                auth_key.hex(),
                "--visualizer",
                visualizer,
                "--device",
                device,
                "--presets",
                presets,
                "--seed",
                str(seed),
            ]
            creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            try:
                self._process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=self._log_handle,
                    stderr=subprocess.STDOUT,
                    creationflags=creation_flags,
                )
                self._connection = listener.accept()
                hello = self._receive(self._connection, 10)
                if hello.get("event") != "connected":
                    raise WorkerError("simulator worker did not complete its connection handshake")
                response = self._receive(self._connection, WORKER_START_TIMEOUT_SECONDS)
                result = self._unwrap(response)
                self.visualizer = visualizer
                return {
                    "already_running": False,
                    "visualizer": visualizer,
                    "worker_pid": result["pid"],
                    "worker_log": str(self.log_path),
                    "state": result["state"],
                }
            except BaseException:
                self.stop()
                raise
            finally:
                listener.close()

    @staticmethod
    def _unwrap(response: dict[str, Any]) -> Any:
        if response.get("ok"):
            return response.get("result")
        detail = response.get("error", "unknown simulator worker error")
        raise WorkerError(str(detail))

    def request(self, operation: str, **payload: Any) -> Any:
        with self._lock:
            if not self.running or self._connection is None:
                raise WorkerError("simulator is not running; call start_simulator first")
            try:
                self._connection.send({"operation": operation, **payload})
                return self._unwrap(self._receive(self._connection, WORKER_TOOL_TIMEOUT_SECONDS))
            except (EOFError, BrokenPipeError, OSError) as exc:
                self.stop()
                raise WorkerError(f"lost connection to simulator worker: {exc}") from exc

    def stop(self) -> dict[str, Any]:
        with self._lock:
            was_running = self.running
            if was_running and self._connection is not None:
                try:
                    self._connection.send({"operation": "shutdown"})
                    self._receive(self._connection, 15)
                except (EOFError, BrokenPipeError, OSError, WorkerError):
                    pass
            if self._connection is not None:
                self._connection.close()
                self._connection = None
            if self._process is not None:
                try:
                    self._process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self._process.terminate()
                    self._process.wait(timeout=10)
                self._process = None
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None
            self.visualizer = None
            return {"stopped": was_running}


def _validate_ticks(ticks: int) -> int:
    if not 1 <= ticks <= MAX_TOOL_TICKS:
        raise ValueError(f"ticks must be between 1 and {MAX_TOOL_TICKS}")
    return ticks


def _validated_actions(actions: list[float]) -> list[float]:
    if len(actions) != len(JOINTS):
        raise ValueError(f"actions must contain exactly {len(JOINTS)} values in this order: {', '.join(JOINTS)}")
    if not all(math.isfinite(value) for value in actions):
        raise ValueError("all action values must be finite")
    return [max(-1.0, min(1.0, value)) for value in actions]


runtime = WorkerClient()
atexit.register(runtime.stop)

server = FastMCP(
    "so101-sim",
    instructions=(
        "Control one local SO-101 Isaac Lab simulation. Call start_simulator before other tools. "
        "Read docs/SO101_CONTROL_GUIDE.md in the IsaacLabTutorial project before manipulation. "
        "Prefer move_joint_positions for precise poses and inspect get_robot_state between movements. "
        "Actions are normalized "
        "relative joint-position commands in [-1, 1], not angles. Never claim a physical robot moved: these tools "
        "control only the simulation. Call stop_simulator when the user is finished."
    ),
    log_level="WARNING",
)


@server.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False)
)
def start_simulator(
    visualizer: VisualizerName = "newton",
    device: str = "cuda:0",
    seed: int = 42,
) -> dict[str, Any]:
    """Start the local SO-101 simulation and return its initial state.

    Use ``newton`` to show the live simulator window or ``none`` for headless operation.
    """

    return runtime.start(visualizer=visualizer, device=device, seed=seed)


@server.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
)
def get_robot_state() -> dict[str, Any]:
    """Read joint positions, targets, velocities, end-effector position, and vial position."""

    return runtime.request("state")


@server.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)
)
def move_joint_positions(positions_rad: list[float], ticks: int = 60) -> dict[str, Any]:
    """Interpolate to six absolute joint targets in radians, then retain them while idle.

    Order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.
    Returns measured state and command errors; servo targets do not guarantee arrival.
    This is joint interpolation, not collision-aware Cartesian planning.
    """
    if len(positions_rad) != 6 or not all(math.isfinite(v) for v in positions_rad):
        raise ValueError("positions_rad requires six finite values")
    return runtime.request("positions", positions=positions_rad, ticks=_validate_ticks(ticks))


@server.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)
)
def pulse_joint(joint: JointName, value: float, ticks: int = 30) -> dict[str, Any]:
    """Move one simulated joint with a bounded normalized action, then hold the resulting pose."""

    _validate_ticks(ticks)
    if not math.isfinite(value):
        raise ValueError("value must be finite")
    actions = [0.0] * len(JOINTS)
    actions[JOINT_INDEX[joint]] = max(-1.0, min(1.0, value))
    return runtime.request("action", values=actions, ticks=ticks)


@server.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)
)
def apply_joint_actions(actions: list[float], ticks: int = 1) -> dict[str, Any]:
    """Apply six normalized actions for a bounded number of 30 Hz control ticks, then hold.

    The required order is shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.
    """

    return runtime.request("action", values=_validated_actions(actions), ticks=_validate_ticks(ticks))


@server.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)
)
def open_gripper(ticks: int = DEFAULT_GRIPPER_TICKS) -> dict[str, Any]:
    """Open the simulated gripper for a bounded number of 30 Hz control ticks, then hold."""

    return pulse_joint("gripper", 1.0, ticks)


@server.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)
)
def close_gripper(ticks: int = DEFAULT_GRIPPER_TICKS) -> dict[str, Any]:
    """Close the simulated gripper for a bounded number of 30 Hz control ticks, then hold."""

    return pulse_joint("gripper", -1.0, ticks)


@server.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)
)
def reset_simulator() -> dict[str, Any]:
    """Reset the simulation to the next canonical home pose and return the new state."""

    return runtime.request("reset")


@server.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False)
)
def stop_simulator() -> dict[str, Any]:
    """Close the local simulator worker and its viewer window."""

    return runtime.stop()


def main() -> None:
    """Run the SO-101 MCP server over STDIO."""

    server.run(transport="stdio")


if __name__ == "__main__":
    main()
