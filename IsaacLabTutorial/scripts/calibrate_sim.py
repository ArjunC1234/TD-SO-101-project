"""Reproducible model calibration; teleported FK probes are diagnostics, never task success."""

import argparse
import json
from pathlib import Path

import torch

from isaaclab_tutorial.mcp_worker import SimulatorWorker
from isaaclab_tutorial.so101_joints import JOINTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="docs/so101_calibration.json")
    args = parser.parse_args()
    worker = SimulatorWorker(visualizer="none", device=args.device, presets="newton_mjwarp", seed=42)
    try:
        env = worker.env
        robot = env.scene["robot"]
        ids = robot.find_joints(list(JOINTS), preserve_order=True)[0]
        home = robot.data.joint_pos.torch.clone()
        limits = robot.data.soft_joint_pos_limits.torch[0, ids].clone()

        def place(q):
            robot.write_joint_position_to_sim_index(position=q)
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
            robot.set_joint_position_target_index(target=q)
            env.scene.reset()
            env.scene.write_data_to_sim()
            env.sim.forward()
            env.scene.update(env.physics_dt)
            worker.capture_hold()
            return worker.state()

        initial = place(home)
        report = {
            "preset": "newton_mjwarp",
            "seed": 42,
            "device": args.device,
            "joint_order": JOINTS,
            "home": initial,
            "physics_dt": env.physics_dt,
            "control_dt": env.step_dt,
            "jacobian_m_per_rad": [],
            "pulse_probes": [],
        }
        for j, name in enumerate(JOINTS):
            points = []
            for sign in (-1, 1):
                q = home.clone()
                q[0, ids[j]] += sign * 0.005
                points.append(torch.tensor(place(q)["end_effector_position_m"]))
            report["jacobian_m_per_rad"].append(((points[1] - points[0]) / 0.01).tolist())
            for sign in (-1, 1):
                place(home)
                action = [0.0] * 6
                action[j] = sign * 0.5
                result = worker.apply_action(action, 15)
                report["pulse_probes"].append(
                    {
                        "joint": name,
                        "action": action[j],
                        "ticks": 15,
                        "joint_delta_rad": [
                            result["joints"][n]["position_rad"] - initial["joints"][n]["position_rad"] for n in JOINTS
                        ],
                        "ee_m": result["end_effector_position_m"],
                    }
                )
        place(home)
        for _ in range(300):
            worker.idle_step()
        report["hold_after_10_sim_seconds"] = worker.state()
        for _ in range(300):
            worker.idle_step()
        report["hold_after_20_sim_seconds"] = worker.state()
        target = home[0, ids].clone()
        target[0] += 0.10
        report["absolute_move_probe"] = worker.move_positions(target.tolist(), 60)

        torch.manual_seed(42)
        samples = limits[:, 0] + torch.rand(256, 6, device=env.device) * (limits[:, 1] - limits[:, 0])
        samples[:, 5] = home[0, ids[5]]
        points = []
        for sample in samples:
            q = home.clone()
            q[0, ids] = sample
            points.append(place(q)["end_effector_position_m"])
        points = torch.tensor(points)
        report["geometric_samples"] = {
            "count": len(points),
            "collision_checked": False,
            "min_xyz_m": points.amin(0).tolist(),
            "max_xyz_m": points.amax(0).tolist(),
            "joint_positions_rad": samples.cpu().tolist(),
            "tcp_positions_m": points.tolist(),
        }
        place(home)
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"CALIBRATION_SAVED={path.resolve()}", flush=True)
    finally:
        worker.close()


if __name__ == "__main__":
    main()
