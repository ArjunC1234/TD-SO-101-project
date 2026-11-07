# SO-101 Isaac Lab setup

The working project is NVIDIA's official `IsaacLabTutorial` SO-101 vial-placement simulation.

## Open the simulation

Double-click `Start-SO101.cmd`, or run this from PowerShell in this folder:

```powershell
.\Start-SO101.ps1
```

The launcher opens the native Newton Viewer and cycles through the task's validated reset poses. Close the
viewer or press `Ctrl+C` in the launching terminal to stop it.

## Verify the headless simulation

```powershell
Set-Location .\IsaacLabTutorial
uv run isaaclab zero_agent --task IsaacTutorial-Place-Vial-SO101 --num_envs 8 --max_steps 100 --visualizer none presets=newton_mjwarp
```

## Start a small training run

The upstream recipe uses 4096 environments. This laptop has 6 GB of VRAM, so start with 256 and scale only
after checking memory use:

```powershell
Set-Location .\IsaacLabTutorial
uv run isaaclab train --rl_library rsl_rl --task IsaacTutorial-Place-Vial-SO101 --num_envs 256 --visualizer none presets=newton_mjwarp
```

Checkpoints are written below `IsaacLabTutorial\logs\rsl_rl\`.

## Installed versions

- Isaac Lab 3.0.0 (`release/3.0.0` source pin)
- PyTorch 2.11.0 with CUDA 13.0
- Newton 1.6.1 / MJWarp physics
- Python 3.12 (managed by `uv`)

The repository's locked environment is in `IsaacLabTutorial\.venv`. Run commands through `uv run` from the
project directory; no global Python or Conda activation is required.
