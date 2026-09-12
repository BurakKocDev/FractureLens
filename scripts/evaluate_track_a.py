from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / "artifacts" / "ultralytics_config"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "artifacts" / "matplotlib_config"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "artifacts" / "torch_cache"))
for variable in ("YOLO_CONFIG_DIR", "MPLCONFIGDIR", "TORCH_HOME"):
    Path(os.environ[variable]).mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Track A checkpoint")
    parser.add_argument("--task", choices=("detect", "segment"), default="detect")
    parser.add_argument(
        "--weights",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "runs"
            / "track_a_detect_30ep"
            / "weights"
            / "best.pt"
        ),
    )
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--name", default="track_a_detect_30ep_test")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha() -> str:
    return subprocess.run(
        ["git", "-c", f"safe.directory={PROJECT_ROOT.as_posix()}", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    args = parse_args()
    weights = args.weights.resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)

    import torch
    import ultralytics
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing an accidental CPU evaluation")
    evaluation_root = PROJECT_ROOT / "artifacts" / "evaluations"
    output_root = evaluation_root / args.name
    if output_root.exists():
        raise FileExistsError(f"Evaluation name already exists: {output_root}")

    model = YOLO(str(weights))
    result = model.val(
        data=str(
            PROJECT_ROOT / "data" / "processed" / f"track_a_{args.task}" / "data.yaml"
        ),
        split=args.split,
        imgsz=600,
        batch=args.batch,
        device=0,
        workers=2,
        plots=True,
        save_json=False,
        project=str(evaluation_root),
        name=args.name,
        exist_ok=False,
        verbose=True,
    )
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        "weights": str(weights),
        "weights_sha256": file_sha256(weights),
        "task": args.task,
        "split": args.split,
        "requested_imgsz": 600,
        "effective_imgsz": 608,
        "batch": args.batch,
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "metrics": {key: float(value) for key, value in result.results_dict.items()},
        "speed_ms_per_image": {key: float(value) for key, value in result.speed.items()},
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
