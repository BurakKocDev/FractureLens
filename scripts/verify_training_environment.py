from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / "artifacts" / "ultralytics_config"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "artifacts" / "matplotlib_config"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "artifacts" / "torch_cache"))
for variable in ("YOLO_CONFIG_DIR", "MPLCONFIGDIR", "TORCH_HOME"):
    Path(os.environ[variable]).mkdir(parents=True, exist_ok=True)

import torch  # noqa: E402
import torchvision  # noqa: E402
import ultralytics  # noqa: E402


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    tensor = torch.rand((1024, 1024), device="cuda")
    report = {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_available": True,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_tensor_sum": float(tensor.sum()),
        "allocated_mib": round(torch.cuda.memory_allocated() / 1_048_576, 1),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
