"""Drive the Newton SO-101 scene with the downloaded ACT policy.

Two fixed cameras, ``up`` and ``side``, match the policy's 640 by 480 views.
The Newton window starts on the overhead camera. Both feeds are also written
to outputs/act_cameras so the side view is visible.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import subprocess
import sys
from pathlib import Path

from isaaclab_tutorial.mcp_worker import SimulatorWorker
from isaaclab_tutorial.so101_joints import JOINTS

POLICY_SERVER = Path(__file__).resolve().parents[2] / "scripts" / "act_policy_server.py"
PREVIEW_DIR = Path(__file__).resolve().parents[1] / "outputs" / "act_cameras"
STEPS = 80
TICKS_PER_ACTION = 4
LOOK_AT = (0.20, 0.02, 0.08)
UP_EYE = (0.20, 0.02, 0.55)
SIDE_EYE = (0.20, -0.45, 0.28)


def clamp(values: list[float], limits: list[list[float]]) -> list[float]:
    clamped = []
    for value, (low, high) in zip(values, limits, strict=True):
        clamped.append(min(max(value, low), high))
    return clamped


def look_quat(eye: tuple[float, float, float], target: tuple[float, float, float], up: tuple[float, float, float]):
    """Quaternion (x, y, z, w) for a world-convention camera: +X forward, +Z up."""

    import torch

    from isaaclab.utils.math import quat_from_matrix

    forward = torch.tensor(target, dtype=torch.float32) - torch.tensor(eye, dtype=torch.float32)
    forward = forward / forward.norm()
    up_axis = torch.tensor(up, dtype=torch.float32)
    up_axis = up_axis - forward * torch.dot(up_axis, forward)
    up_axis = up_axis / up_axis.norm()
    side_axis = torch.cross(up_axis, forward, dim=0)
    matrix = torch.stack((forward, side_axis, up_axis), dim=1).unsqueeze(0)
    quat = quat_from_matrix(matrix)[0]
    if not torch.isfinite(quat).all():
        raise RuntimeError(f"camera orientation at {eye} is invalid")
    return tuple(float(value) for value in quat)


def policy_camera(name: str, eye: tuple[float, float, float], up: tuple[float, float, float]):
    from isaaclab.sensors import CameraCfg
    from isaaclab.sim import PinholeCameraCfg
    from isaaclab_newton.renderers import NewtonWarpRendererCfg

    renderer = NewtonWarpRendererCfg(enable_shadows=True)
    return CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955),
        offset=CameraCfg.OffsetCfg(
            pos=eye,
            rot=look_quat(eye, LOOK_AT, up),
            convention="world",
        ),
        data_types=["rgb"],
        width=640,
        height=480,
        update_period=0.0,
        update_latest_camera_pose=True,
        renderer_cfg=renderer,
        background_color=(0.3, 0.3, 0.3),
    )


def add_policy_cameras(env_cfg) -> None:
    env_cfg.scene.up_camera = policy_camera("UpCamera", UP_EYE, (1.0, 0.0, 0.0))
    env_cfg.scene.side_camera = policy_camera("SideCamera", SIDE_EYE, (0.0, 0.0, 1.0))
    env_cfg.sim.default_visualizer_cfg.eye = UP_EYE
    env_cfg.sim.default_visualizer_cfg.lookat = LOOK_AT
    env_cfg.sim.default_visualizer_cfg.streaming_view = True


def camera_rgb(env, name: str):
    import torch

    image = env.scene.sensors[name].data.output["rgb"].torch[0, ..., :3].detach().cpu()
    image = (image.clamp(0, 1) * 255).to(torch.uint8) if image.dtype.is_floating_point else image.to(torch.uint8)
    return image.contiguous()


def encode_rgb(image) -> str:
    return base64.b64encode(image.numpy().tobytes()).decode("ascii")


def save_preview(up, side) -> None:
    from PIL import Image

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    pair = Image.new("RGB", (640 * 2, 480))
    pair.paste(Image.fromarray(up.numpy()), (0, 0))
    pair.paste(Image.fromarray(side.numpy()), (640, 0))
    pair.save(PREVIEW_DIR / "up_and_side.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-python", required=True, type=Path, help="Python executable in a LeRobot 0.4.2 environment"
    )
    args = parser.parse_args()
    if not args.policy_python.is_file():
        parser.error(f"Policy Python executable not found: {args.policy_python}")
    policy = subprocess.Popen(
        [str(args.policy_python.resolve()), str(POLICY_SERVER)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert policy.stdin is not None and policy.stdout is not None
    ready = json.loads(policy.stdout.readline())
    if not ready.get("ready"):
        policy.kill()
        raise RuntimeError(f"policy server failed to start: {ready}")

    worker = None
    try:
        worker = SimulatorWorker(
            visualizer="newton",
            device="cuda:0",
            presets="newton_mjwarp,newton_renderer",
            seed=42,
            scene_setup=add_policy_cameras,
        )
        print("[ACT] Newton viewer is on the overhead camera. Side camera is rendering too.", flush=True)
        for step in range(STEPS):
            if not worker.env.sim.is_running():
                print("[ACT] Viewer closed.", flush=True)
                return 0
            snapshot = worker.state()
            up = camera_rgb(worker.env, "up_camera")
            side = camera_rgb(worker.env, "side_camera")
            if step == 0:
                save_preview(up, side)
                print(f"[ACT] Wrote camera preview to {PREVIEW_DIR / 'up_and_side.png'}", flush=True)
            state_deg = [math.degrees(snapshot["joints"][name]["position_rad"]) for name in JOINTS]
            policy.stdin.write(
                json.dumps({"state_deg": state_deg, "up": encode_rgb(up), "side": encode_rgb(side)}) + "\n"
            )
            policy.stdin.flush()
            reply = json.loads(policy.stdout.readline())
            if "error" in reply:
                raise RuntimeError(reply["error"])
            target_rad = clamp(
                [math.radians(value) for value in reply["action_deg"]],
                snapshot["joint_limits_rad"],
            )
            result = worker.move_positions(target_rad, ticks=TICKS_PER_ACTION)
            print(
                f"[ACT] step={step} vial_z={result['vial_position_m'][2]:.3f} "
                f"jaws={result['jaw_contacts']['fixed']}/{result['jaw_contacts']['moving']}",
                flush=True,
            )
    finally:
        policy.kill()
        if worker is not None:
            worker.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
