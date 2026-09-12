from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a pinned FractureLens Track A experiment")
    parser.add_argument("--task", choices=("detect", "segment"), default="detect")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--name", default=None)
    return parser.parse_args()


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
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.batch is not None and args.batch < 1:
        raise ValueError("batch must be positive")

    import torch
    import ultralytics
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing an accidental CPU training run")
    if args.task == "segment":
        raise NotImplementedError("Track A segmentation export is intentionally gated after detection")

    config_path = PROJECT_ROOT / "configs" / "experiment" / "track_a_official_reproduction.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_config = next(item for item in config["runs"] if item["task"] == args.task)
    epochs = args.epochs or int(run_config["epochs"])
    batch = args.batch or int(config["local_execution"]["first_batch"])
    name = args.name or f"track_a_{args.task}_{epochs}ep"
    output_root = PROJECT_ROOT / "artifacts" / "runs"
    run_root = output_root / name
    data_yaml = PROJECT_ROOT / "data" / "processed" / "track_a_detect" / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError("Run scripts/prepare_track_a_yolo.py first")
    if run_root.exists():
        raise FileExistsError(f"Run name already exists: {run_root}")

    provenance = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision_cuda": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "task": args.task,
        "model": run_config["model"],
        "epochs": epochs,
        "imgsz": int(run_config["imgsz"]),
        "batch": batch,
        "seed": 0,
    }

    model = YOLO(run_config["model"])
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=int(run_config["imgsz"]),
        batch=batch,
        device=0,
        workers=int(config["local_execution"]["workers"]),
        amp=bool(config["local_execution"]["amp"]),
        seed=0,
        deterministic=True,
        project=str(output_root),
        name=name,
        exist_ok=False,
        plots=True,
        verbose=True,
    )
    actual_run_root = Path(model.trainer.save_dir)
    (actual_run_root / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
