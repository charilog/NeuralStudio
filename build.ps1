# build.ps1 - NeuralStudio Windows build script
# -----------------------------------------------------------------------------
# Usage:
#   .\build.ps1                          # defaults below
#   .\build.ps1 -QtPath "C:\Qt\6.10.1\msvc2022_64"
#   .\build.ps1 -Config Debug
#   .\build.ps1 -NoBuild                 # configure only
#   .\build.ps1 -NoDeploy                # build but skip windeployqt
#   .\build.ps1 -NoRun                   # don't launch after build
#   .\build.ps1 -NoCli                   # skip the console program neuralstudio-cli.exe
# -----------------------------------------------------------------------------
param(
    [string] $QtPath   = "C:\Qt\6.10.1\msvc2022_64",
    [string] $Config   = "Release",
    [switch] $NoBuild,
    [switch] $NoDeploy,
    [switch] $NoRun,
    [switch] $NoCMake,
    [switch] $NoCli
)

$ErrorActionPreference = "Stop"
$Target   = "neuralstudio"
$ExeName  = "$Target.exe"
$Root     = $PSScriptRoot
$BuildDir = Join-Path $Root "build"
$ExePath  = Join-Path $BuildDir "$Config\$ExeName"

Write-Host ""
Write-Host "  NeuralStudio - Windows build" -ForegroundColor Cyan
Write-Host "  Qt   : $QtPath"               -ForegroundColor Gray
Write-Host "  Cfg  : $Config"               -ForegroundColor Gray
Write-Host "  Out  : $ExePath"              -ForegroundColor Gray
Write-Host ""

# -- 1. Clean + Configure ------------------------------------------------------
if (-not $NoCMake) {
    Write-Host "[1/4] Cleaning build/ ..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $BuildDir -ErrorAction SilentlyContinue

    Write-Host "[1/4] CMake configure ..." -ForegroundColor Yellow
    cmake -S . -B build `
        "-DCMAKE_PROJECT_INCLUDE:FILEPATH=$Root\cmake\neuralstudio.cmake" `
        "-DCMAKE_PREFIX_PATH=$QtPath"

    if ($LASTEXITCODE -ne 0) {
        Write-Host "CMake configure FAILED (exit $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

if ($NoBuild) { Write-Host "NoBuild - done."; exit 0 }

# -- 2. Build ------------------------------------------------------------------
Write-Host "[2/4] Building target '$Target' ($Config) ..." -ForegroundColor Yellow
cmake --build build --config $Config --target $Target

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build FAILED (exit $LASTEXITCODE)" -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "Build OK -> $ExePath" -ForegroundColor Green

# Console program (every mode from the command line, see docs\CLI.md). It is placed next to
# NeuralStudio.exe, so the Qt DLLs deployed below serve both programs.
if (-not $NoCli) {
    Write-Host "[2/4] Building target 'neuralstudio-cli' ($Config) ..." -ForegroundColor Yellow
    cmake --build build --config $Config --target neuralstudio-cli
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Build of neuralstudio-cli FAILED (exit $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host "Build OK -> $(Join-Path $BuildDir "$Config\neuralstudio-cli.exe")" -ForegroundColor Green
}

# -- 3. windeployqt -----------------------------------------------------------
if (-not $NoDeploy) {
    $WinDeployQt = Join-Path $QtPath "bin\windeployqt.exe"
    if (-not (Test-Path $WinDeployQt)) {
        Write-Host "windeployqt not found at $WinDeployQt - skipping." -ForegroundColor Yellow
    } else {
        Write-Host "[3/4] Running windeployqt ..." -ForegroundColor Yellow
        & $WinDeployQt --release --compiler-runtime --no-translations $ExePath
        if ($LASTEXITCODE -ne 0) {
            Write-Host "windeployqt warnings (exit $LASTEXITCODE)" -ForegroundColor Yellow
        } else {
            Write-Host "Deploy OK" -ForegroundColor Green
        }
    }
} else {
    Write-Host "[3/4] Deploy skipped." -ForegroundColor Gray
}

# -- 4. Run --------------------------------------------------------------------
if (-not $NoRun) {
    Write-Host "[4/4] Launching $ExeName ..." -ForegroundColor Yellow
    & $ExePath
} else {
    Write-Host "[4/4] Run skipped." -ForegroundColor Gray
}
