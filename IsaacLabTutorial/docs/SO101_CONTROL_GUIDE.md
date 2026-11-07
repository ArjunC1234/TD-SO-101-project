# SO-101 simulation: calibrated control and manipulation guide

Read this before executing vague requests such as “pick up the vial.” Calibration performed on 2026-10-31
using this checkout, Newton MJWarp, `cuda:0`, one environment, seed 42. Raw measurements are in
[so101_calibration.json](so101_calibration.json). Reproduce with
`uv run python scripts/calibrate_sim.py` from the project directory. This is simulation calibration;
it does not calibrate a connected physical arm or establish hardware encoder offsets.

## First actions in a new session

1. Start the simulator, then read `get_robot_state`. If it is already running, reuse it.
2. Verify the tools include `move_joint_positions` and the state includes `vial_grasp_point_m`,
   `joint_limits_rad`, fingertip positions, and `jaw_contacts`. Restart the MCP server/client after a code
   update if the new tool is missing. Never assume an old live worker uses the new code.
3. Preserve the current scene unless a reset is needed. Reset advances through canonical rows and can move
   the vial; always re-read the object pose afterward.
4. For a pickup, identify the object, its grasp geometry, desired lift height, and a collision-free approach.
   A default small lift of 30 mm is a verification maneuver, not completion of an unspecified placement task.

## Degrees of freedom and limits

The arm has **five pose joints plus one gripper joint**. It cannot independently specify every component of
a general six-dimensional Cartesian pose. Prioritize position and grasp-axis alignment; relax redundant
orientation demands for an axially symmetric vial. Gripper opening is not an extra wrist orientation axis.

All API joint positions are radians; positions are world-frame meters. `ticks` use a 30 Hz control interval;
physics is 120 Hz. Simulation time can advance faster than wall time during a motion request.

| Index / joint | Role | Loaded soft limits, degrees | Measured initial home, degrees |
|---|---|---:|---:|
| 0 `shoulder_pan` | Sweep arm about base; changes azimuth | -107.80 to 107.80 | -6.996 |
| 1 `shoulder_lift` | Shoulder pitch; changes reach and height | -98.00 to 98.00 | -51.879 |
| 2 `elbow_flex` | Elbow pitch; folds/extends the arm | -88.49 to 101.29 | 17.878 |
| 3 `wrist_flex` | Wrist pitch; changes approach angle and TCP position | -93.10 to 93.10 | 84.871 |
| 4 `wrist_roll` | Roll gripper about its longitudinal direction | -154.01 to 159.59 | -46.090 |
| 5 `gripper` | Moving jaw opening; positive opens | -11.23 to 113.61 | 14.639 |

Use runtime limits as authoritative. Soft limits are command bounds, not assurances of table, rack,
self-collision, or object clearance. Do not drive to a limit to infer reach.

## Coordinate and grasp conventions

World +Z is upward; X/Y are scene axes, not screen directions. Initial home TCP was approximately
`[0.1501, 0.0349, 0.2248]` m. Positive pan locally moves the TCP toward negative Y.

`end_effector_position_m` is the **midpoint of the fingertip pads**, not the wrist origin. Opening the jaw
moves this midpoint even if the five arm joints remain fixed. Recompute alignment after changing aperture.
Use `fixed_fingertip_m` and `moving_fingertip_m` to check which side of the object each pad occupies.

`vial_position_m` is the vial asset root. The task's cap/shoulder grasp location is:

```text
grasp_world = vial_root_world + rotate_xyzw(vial_quaternion_xyzw, [0, 0, 0.093145])
```

Read `vial_grasp_point_m`, which already computes this transform. In the calibration scene the root was
`[0.23780, -0.05619, 0.04954]` m but the grasp point was `[0.32551, -0.02503, 0.05291]` m.
The earlier pickup attempt targeted the wrong part of the vial and did not lift it.

For the generator's gripper frame, +X is the jaw-closing axis, +Y follows pad length, and +Z is the approach
axis. For a lying vial, align pad length with the vial's world axis and closing direction perpendicular to
that axis. Read `_solve_pregrasp` in `tasks/place_vial/reset/generator.py` for the actual orientation construction.
This pinned Isaac/Newton fork uses **XYZW**, including `isaaclab.utils.math`. The earlier
`vial_quaternion_wxyz` snapshot key was mislabeled: its values are XYZW. Old running workers and the
calibration JSON retain that legacy key; interpret those arrays as XYZW too. New workers use
`vial_quaternion_xyzw`. Verify the convention by reconstructing the reported grasp point.

## Precisely commanding joints

Prefer `move_joint_positions(positions_rad=[six values], ticks=60)`. It interpolates absolute servo targets,
settles, applies bounded endpoint correction, and keeps fixed servo targets while idle. It rejects nonfinite
and out-of-limit targets. Start with the latest measured joint vector and change only intended coordinates.

The returned `command_error_rad` is desired minus measured position. `reached_joint_tolerance` means all six
errors are at most 0.005 rad (0.286 degrees) when sampled. It is not a Cartesian accuracy, grasp, velocity,
or collision guarantee. Check subsequent state and the relevant task geometry as well.

The controller allows at most ten correction rounds, each with fifteen settling ticks, and limits each
correction increment. If tolerance is not reached, inspect contact, limits, and pose before retrying. Do not
keep increasing a command against the table. Expected gripper contact may intentionally prevent reaching
the requested closed-jaw angle; use bilateral contact and lift evidence instead of forcing that angle.

Fixed targets stop the previous continuous zero-action drift. Static load error still exists, so holding a
servo target and reaching an exact angle are different. At home without endpoint compensation, the elbow
settled about 1.31 degrees away from its target; its additional drift from 10 to 20 simulated seconds was
only 0.041 degrees. In the measured free-space test (home with pan increased by 0.10 rad), endpoint
correction reached a maximum joint error of **0.003651 rad / 0.209 degrees**, below the 0.005 rad criterion.
This is one tested move, not a guarantee across poses or loads; the elbow was still moving at about
0.00867 rad/s at that sample. Re-read state after settling before declaring a precise task complete.

Do not mix a learned policy or the terminal controller with the MCP worker at the same time. Direct
absolute control bypasses the RL action manager and does not maintain RL observation/reward history.

For an older client whose tool list lacks `move_joint_positions`, the updated worker also accepts a local
request at `outputs/so101_control/request.json`: `{"id":"unique-id","positions_rad":[six radians],"ticks":120}`.
Write through a temporary file and atomically rename it; read the matching ID in `response.json` before
continuing. The worker ignores the request present at startup. This uses the same absolute controller and
must not run concurrently with other motion commands. Prefer the MCP tool once the client is refreshed.

### Relative pulses are response probes, not angle commands

`pulse_joint` and `apply_joint_actions` retain the original relative target interface:

```text
q_target = q_measured + action * scale
arm scale = 0.033 rad; gripper scale = 0.020 rad; action clamped to [-1, 1]
```

This target is recomputed every physics substep. Multiplying ticks by scale does not predict actual travel.
At home, starting from the same state for each probe, 15 ticks at ±0.5 produced these measured joint changes:

| Joint | -0.5 change, degrees | +0.5 change, degrees |
|---|---:|---:|
| shoulder_pan | -2.655 | +2.655 |
| shoulder_lift | -3.556 | +4.877 |
| elbow_flex | **+0.064** | +4.076 |
| wrist_flex | -2.163 | +3.114 |
| wrist_roll | -4.302 | +4.282 |
| gripper | -4.468 | +4.435 |

These are pose-specific responses, not a universal lookup table. Gravity, friction, load, and contact change
them. Noncommanded joints can also move during relative-action pulses. Read all six joints after a pulse.

## Cartesian movement and reachable space

The measured local TCP Jacobian at home is below, in **millimeters of XYZ motion per +1 degree** of joint
change. It was obtained from central differences of actual model forward kinematics, ±0.005 rad. It is a
local geometric derivative, not a measured dynamic response or a fixed map valid throughout the workspace.

| Joint | dX | dY | dZ |
|---|---:|---:|---:|
| shoulder_pan | +0.247 | -3.090 | 0.000 |
| shoulder_lift | +1.312 | +0.161 | -2.567 |
| elbow_flex | -0.274 | -0.034 | -3.811 |
| wrist_flex | -1.857 | -0.228 | -2.077 |
| wrist_roll | +0.075 | +0.130 | +0.091 |
| gripper | +0.298 | -0.444 | +0.406 |

Wrist roll can slightly translate this off-axis TCP. Shoulder/elbow/wrist pitch motions are coupled: a
single joint rarely moves “straight up” or “straight forward.” Near singularities, some Cartesian directions
require large joint changes. Recompute the Jacobian or use IK at the current pose.

A deterministic 256-pose sample across joint limits, with jaw opening fixed at home, produced TCP extrema:

| World coordinate | Sample minimum, m | Sample maximum, m |
|---|---:|---:|
| X | -0.3370 | +0.3988 |
| Y | -0.3956 | +0.4412 |
| Z | -0.1678 | +0.5585 |

This is a **sampled geometric envelope**, not an exhaustive workspace bound, reachable box, or safe operating
volume. Each extreme belongs to a different posture. Some sampled postures go through the table. A point
inside this box can still be unreachable with the required orientation or a collision-free path.
The calibration JSON preserves all sampled joint/TCP pairs for reproducibility.

For an arbitrary target, use the project's Newton IK machinery in `reset/generator.py` (`_build_ik`,
`_solve_pregrasp`, `_solve_vial_pose`) as the implementation reference. Those are internal generator methods,
not exposed MCP tools. The current MCP supports absolute joints and state; it does not yet provide a general
Cartesian planning tool. Do not pretend a Cartesian request was solved by selecting a nearby sampled pose.

For small free-space corrections, compute a current 3x5 position Jacobian with gripper fixed, then use
`dq = J.T @ solve(J @ J.T + lambda**2 * I, target_xyz - tcp_xyz)`. Limit each update to a small joint change
(initially <=0.03 rad), enforce limits, execute, and remeasure. This solves position only; add orientation
constraints or use the existing IK for a grasp. Reduce steps near contact and reject a correction if error
grows or clearance decreases. A practical initial near-object position acceptance threshold is 3 mm;
that is a task criterion to verify, not calibration evidence of guaranteed 3 mm accuracy.

## Turning “pick up the vial” into a verified operation

1. Observe current object position **and orientation**, cap grasp point, both pads, contacts, and joint limits.
   For other objects, obtain their dimensions and a valid grasp frame first; the existing vial offset is specific
   to this asset. The state tool has no general object detector.
2. Open to the required aperture in free space; record the new pad midpoint. Home/pre-grasp is about
   0.255 rad, the project's nominal grasp command is -0.1553 rad, and release is about 0.645 rad.
   These are example jaw angles, not universal widths or guaranteed contact settings.
3. Solve a pre-grasp pose 30–50 mm above the cap target, with the correct closing and pad axes. Check the
   full fingers, camera mount, table, and rack along the approach. Pure joint interpolation can dip below
   the endpoints; validate intermediate poses.
4. Move to the elevated pose, inspect actual error, then descend in small Cartesian segments while checking
   the updated object pose. Replan if the vial rolls or is nudged. Do not scrape along the table to converge.
5. Close gradually, holding the arm's absolute targets. Require both `jaw_contacts.fixed` and `.moving`
   (the task threshold is 0.05 N) plus appropriate pad/target alignment. Table contact alone is not a grasp.
6. Lift about 30 mm while retaining the jaw command. Verify the **vial** rises, not just the gripper.
   Require sustained elevation and near-constant object-to-gripper offset over several readings; retain
   bilateral contact where available. If it stays at table height, explicitly report a failed grasp.
7. Keep holding after success. If transport or placement was requested, plan those phases separately.

Sparse demonstrated pre-grasp/lift waypoints in `reset/generator.py` are seeds for solving the current scene.
They do not account for a shifted vial, and copying the entire waypoint list is not a validated path for an
arbitrary scene. The generator already applies the 6.4-degree elbow-coordinate correction; do not add it twice.

## Observed pickup and release on 2026-10-31

Two cap grasps produced bilateral contact but slipped on lifting. A body grasp 30 mm toward the vial root
from the cap point succeeded: target `vial_grasp_point_m - 0.030 * vial_axis_world`. The arm lifted the vial
from about 0.050 m root height to 0.088 m with sustained bilateral contact, turned it upright, and transported
it above the first holder opening at world `[0.180, 0.080]` m. It lowered the root to about 0.096 m before
opening the jaw to 0.645 rad and retracting. All live movement used physics and servo commands; only the
separate headless IK mirror wrote diagnostic poses.

Release was achieved, but the vial leaned after dropping into the holder. Its root settled near 0.070 m
height while the tilt increased. This does **not** demonstrate the stricter upright, seated-and-still task
success criterion. Future precise placements should lower closer to the seat and verify orientation and
stability after release; do not reuse these joint waypoints for a different grasp transform or vial pose.

## Limits of this calibration and verification

- The real STDIO MCP smoke test discovered the absolute-position tool, executed a +0.10 rad pan move
  from startup, verified a maximum joint error of 0.002796 rad (0.160 degrees), and confirmed an
  out-of-limit request was rejected. The 21 manual-controller/MCP unit tests and targeted Ruff checks passed.
- Joint limits, local Jacobian, pulse responses, idle hold, absolute movement, and the geometric sample were
  measured on the loaded simulation. The FK samples teleport joint state in a separate diagnostic run;
  they do not demonstrate physical reachability or a successful manipulation trajectory.
- This task removes non-gripper link colliders; the simulated model does not establish complete real-arm
  collision safety. Review `_remove_non_gripper_colliders` in `env_cfg.py` before transferring any path.
- The initial calibration did not test pickup; the subsequent live pickup/release observations are above.
  A successful MCP return is not task success.
- Recalibrate after asset, physics, actuator, coordinate, or preset changes. `sim2real` uses different action
  scales and actuator behavior. Physical deployment additionally needs hardware calibration and feedback.
- The terminal manual controller still uses relative zero-action idle behavior; fixed-target holding and
  absolute endpoint correction described here belong to the updated MCP worker.

If precise manipulation stalls, inspect actual error/contact and correct the controller or IK. Do not replace
feedback with progressively longer open-loop pulses.
