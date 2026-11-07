param(
    [ValidateRange(1, 100000)]
    [int]$Cycles = 1000,

    [ValidateRange(1, 10000)]
    [int]$StepsPerPose = 45
)

$ErrorActionPreference = "Stop"
$projectRoot = Join-Path $PSScriptRoot "IsaacLabTutorial"

if (-not (Test-Path (Join-Path $projectRoot ".venv"))) {
    throw "Isaac Lab environment not found. Run 'uv sync --locked' in $projectRoot first."
}

Push-Location $projectRoot
try {
    & uv run view-so101-resets `
        --dataset src/isaaclab_tutorial/assets/reset_poses.pt `
        --visualizer newton `
        presets=newton_mjwarp `
        --cycles $Cycles `
        --steps_per_pose $StepsPerPose
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
