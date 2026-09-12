$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimePython = "C:\Users\ASUS\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$env:YOLO_CONFIG_DIR = Join-Path $projectRoot "artifacts\ultralytics_config"
$env:MPLCONFIGDIR = Join-Path $projectRoot "artifacts\matplotlib_config"
$env:TORCH_HOME = Join-Path $projectRoot "artifacts\torch_cache"
New-Item -ItemType Directory -Force -Path $env:YOLO_CONFIG_DIR,$env:MPLCONFIGDIR,$env:TORCH_HOME | Out-Null

if (-not (Test-Path -LiteralPath $venvPython)) {
    & $runtimePython -m venv (Join-Path $projectRoot ".venv")
}

# Override the machine-wide unreachable NVIDIA index for this process only.
$env:PIP_EXTRA_INDEX_URL = "https://pypi.org/simple"

& $venvPython -m pip install `
    torch==2.7.1 `
    torchvision==0.22.1 `
    --index-url https://download.pytorch.org/whl/cu126

& $venvPython -m pip install ultralytics==8.4.149
& $venvPython -m pip install --editable $projectRoot

& $venvPython scripts/verify_training_environment.py
