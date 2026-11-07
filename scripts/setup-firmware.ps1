param(
    [string]$Destination = (Join-Path $PSScriptRoot "../external/STS32XX-Firmware")
)

$ErrorActionPreference = "Stop"
$revision = "a995f77cf83175060bfa8fb908347448a5ae45eb"
$patch = Join-Path $PSScriptRoot "../patches/sts32xx-windows.patch"

if (Test-Path -LiteralPath $Destination) {
    throw "Destination already exists: $Destination. Use a new destination to preserve local changes."
}

& git clone --no-checkout https://github.com/roboninecom/STS32XX-Firmware.git $Destination
if ($LASTEXITCODE -ne 0) { throw "Firmware clone failed." }
& git -C $Destination checkout --detach $revision
if ($LASTEXITCODE -ne 0) { throw "Could not check out pinned firmware revision." }
& git -C $Destination apply --check $patch
if ($LASTEXITCODE -ne 0) { throw "Windows patch did not validate." }
& git -C $Destination apply $patch
if ($LASTEXITCODE -ne 0) { throw "Windows patch failed." }
Write-Host "Firmware tools ready at $Destination. Follow QUICKSTART.md to install dependencies."
