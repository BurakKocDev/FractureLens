$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimePython = "C:\Users\ASUS\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

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

& $venvPython -c "import torch, torchvision, ultralytics; print({'torch': torch.__version__, 'torchvision': torchvision.__version__, 'ultralytics': ultralytics.__version__, 'cuda': torch.cuda.is_available(), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})"
