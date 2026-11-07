# TD SO-101 project

SO-101 robot experiments: an Isaac Lab/Newton simulation, joint control through MCP, measured calibration,
vial pickup training tasks, and experimental LeRobot ACT playback.

## What is implemented

- **Manual and MCP control:** persistent simulator worker, joint state/contact inspection, normalized pulses,
  absolute joint targets, bounded endpoint correction, and fixed targets while idle.
- **Calibration:** joint limits, measured responses, local TCP Jacobian, geometric reach samples, and a
  [control guide](IsaacLabTutorial/docs/SO101_CONTROL_GUIDE.md) describing grasp verification and limitations.
- **Training:** state and wrist-camera variants of vial placement and pickup, pickup rewards/terminations,
  randomized home resets, and an automatic heading curriculum.
- **ACT experiments:** load a public checkpoint, serve actions over STDIO, and try two-camera playback in
  the Newton scene. These are integration experiments, not a verified general-purpose pickup policy.
- **Windows hardware tooling:** a patch adding PowerShell setup instructions and portable path assertions
  to the upstream STS32XX firmware tools. It does not flash motors automatically.

The live simulation demonstrated a body grasp, lift, upright transport, and release onto the holder.
The vial leaned after release; strict upright placement success was **not** established. A high-level
`pick_up("vial")` skill, arbitrary-object training, and a full Isaac Sim installation are future work.

## Run the simulation

Install Git, Python 3.12, `uv`, and a compatible NVIDIA driver. This checkout uses the upstream locked
Isaac Lab 3.0 / Newton environment. The working laptop had an RTX 3060 with 6 GB VRAM; start with a small
environment count. Full Isaac Sim is optional and is not required for these Newton commands.

```powershell
Set-Location IsaacLabTutorial
uv sync --locked
uv run python scripts/manual_control.py --visualizer newton
```

For a reset-pose demonstration, run `Start-SO101.cmd` from the repository root. This launcher cycles
validated reset poses; it does not run a learned policy. See [SO101-SETUP.md](SO101-SETUP.md).

## Connect MCP

From `IsaacLabTutorial`, register the local server with the Python executable in the environment:

```powershell
$python = (Resolve-Path .venv/Scripts/python.exe).Path
codex mcp add so101-sim -- $python -m isaaclab_tutorial.mcp_server
```

Restart the MCP connection after registration or tool changes. The server provides `start_simulator`,
`get_robot_state`, `move_joint_positions`, `pulse_joint`, `apply_joint_actions`, `open_gripper`,
`close_gripper`, `reset_simulator`, and `stop_simulator`. These tools operate the simulation.
An editor configuration template is in [.cursor/mcp.example.json](.cursor/mcp.example.json).

## Train and test

```powershell
Set-Location IsaacLabTutorial
uv run pytest -q
uv run python scripts/mcp_smoke_test.py
uv run isaaclab train --rl_library rsl_rl --task IsaacTutorial-Pickup-Vial-SO101 --num_envs 256 --visualizer none presets=newton_mjwarp
```

The [simulation README](IsaacLabTutorial/README.md) includes camera distillation, placement, evaluation,
and optional dependencies. Policies and run logs are generated locally and excluded from Git.

## ACT checkpoint experiments

The scripts use `samarth-shukla6/act-svla-so101-pickplace` and an independent **LeRobot 0.4.2** environment.
Do not replace the simulation's locked environment with that older LeRobot environment. With its Python
interpreter available, run from this repository root:

```powershell
& C:/path/to/lerobot-venv/Scripts/python.exe scripts/load_act_svla_so101.py --device cpu
IsaacLabTutorial/.venv/Scripts/python.exe IsaacLabTutorial/scripts/play_act_policy.py --policy-python C:/path/to/lerobot-venv/Scripts/python.exe
```

The loader performs a dummy forward pass only. Playback uses rendered views rather than cameras from the
training dataset; matching image names and dimensions does not establish visual or action compatibility.
Physical deployment needs verified action units, hardware calibration, and evaluation on the actual setup.

## Firmware tooling

Fetch the exact upstream firmware revision and apply our Windows patch:

```powershell
./scripts/setup-firmware.ps1
```

This creates `external/STS32XX-Firmware`. Follow its patched `QUICKSTART.md`; run `npm.cmd ci` and its tests
before connecting hardware. The external clone, local backups, motor calibration files, and recordings
are excluded from this project repository.

## Validation of this import

- Simulation suite: **130 passed, 2 skipped** (optional `cv2` and `leapp` dependencies absent).
- Firmware JavaScript suite: **477 passed**.
- Firmware flash-tool Python suite: **56 passed**.
- Earlier headless MCP round trip exercised discovery, absolute movement, and rejection of joint limits.

These results do not establish arbitrary-object generalization, real-arm operation, or successful ACT
pickups. See [THIRD_PARTY.md](THIRD_PARTY.md) for upstream origins and component licenses.
