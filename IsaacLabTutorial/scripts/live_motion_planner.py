"""Headless IK mirror: state writes affect only this planning copy, never the live viewer."""

import json
import time
import traceback
from pathlib import Path

import torch

from isaaclab_tutorial.mcp_worker import SimulatorWorker
from isaaclab_tutorial.so101_joints import JOINTS
from isaaclab_tutorial.tasks.place_vial.reset.generator import (
    WORKSHOP_PREGRASP_JOINT_POSITION,
    GeneratorCfg,
    _Generator,
)


def main():
    folder = Path(__file__).resolve().parents[1] / "outputs/live_planner"
    folder.mkdir(parents=True, exist_ok=True)
    worker = SimulatorWorker(visualizer="none", device="cuda:0", presets="newton_mjwarp", seed=42)
    generator = _Generator(worker.env, GeneratorCfg(batch_size=1))
    print("PLANNER_READY", flush=True)
    seen = None
    try:
        while True:
            path = folder / "request.json"
            if not path.exists():
                time.sleep(0.1)
                continue
            request = json.loads(path.read_text())
            if request["id"] == seen:
                time.sleep(0.1)
                continue
            seen = request["id"]
            print(f"PLANNING {seen}", flush=True)
            if request["mode"] == "quit":
                break
            try:
                state = request["state"]
                q = torch.tensor([[state["joints"][n]["position_rad"] for n in JOINTS]], device=worker.env.device)
                # This pinned Newton fork uses XYZW, despite the legacy snapshot field name.
                quat = state.get("vial_quaternion_xyzw", state.get("vial_quaternion_wxyz"))
                pose = torch.tensor([state["vial_position_m"] + quat], device=worker.env.device)
                generator._write_robot(q)
                generator._write_vial(pose)
                generator._commit_reset_state()
                generator.candidate_ik_valid.fill_(True)
                if request["mode"] == "approach":
                    seed = q.new_tensor([WORKSHOP_PREGRASP_JOINT_POSITION])
                    solved = generator._solve_pregrasp(
                        pose,
                        q,
                        ik_seed_joint_position=seed,
                        vertical_offset=request.get("height", 0.04),
                    )
                else:
                    desired = q.new_tensor([request["desired_vial_pose_xyzw"]])
                    solved = generator._solve_vial_pose(
                        desired, q, position_tolerance=request.get("position_tolerance", 0.008)
                    )
                result = {
                    "id": seen,
                    "valid": bool(generator.candidate_ik_valid[0]),
                    "positions_rad": solved[0].tolist(),
                    "diagnostics": generator.candidate_diagnostics,
                    "rack_position_m": generator.rack.data.root_pos_w.torch[0].tolist(),
                    "rack_quaternion_xyzw": generator.rack.data.root_quat_w.torch[0].tolist(),
                }
            except Exception:
                result = {"id": seen, "error": traceback.format_exc()}
            temp = folder / "response.tmp"
            temp.write_text(json.dumps(result, indent=2))
            temp.replace(folder / "response.json")
            print(json.dumps(result), flush=True)
    finally:
        worker.close()


if __name__ == "__main__":
    main()
