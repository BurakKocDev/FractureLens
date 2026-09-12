from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / "artifacts" / "ultralytics_config"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "artifacts" / "matplotlib_config"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Track B detector with negative images")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--optimizer", choices=("auto", "SGD", "AdamW"), default="auto")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--name", default="track_b_detect_yolov8s_transfer_30ep")
    parser.add_argument(
        "--weights",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts/runs/track_a_detect_fidelity_sgd_b16_30ep/weights/best.pt"
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
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
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing CPU training")
    data_yaml = PROJECT_ROOT / "data/processed/track_b_detect_canonical/data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError("Run scripts/prepare_track_b_yolo.py first")
    weights = args.weights.resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    run_root = PROJECT_ROOT / "artifacts/runs" / args.name
    if run_root.exists():
        raise FileExistsError(run_root)
    model = YOLO(str(weights))
    provenance = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        "task": "detect",
        "track": "B",
        "initial_weights": str(weights),
        "initial_weights_sha256": sha256(weights),
        "data_yaml_sha256": sha256(data_yaml),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": 640,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "seed": 20260912,
        "test_accessed": False,
        "gpu": torch.cuda.get_device_name(0),
    }
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        device=0,
        workers=2,
        amp=True,
        seed=20260912,
        deterministic=True,
        project=str(PROJECT_ROOT / "artifacts/runs"),
        name=args.name,
        exist_ok=False,
        plots=True,
        verbose=True,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
    )
    provenance["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    provenance["peak_cuda_memory_mib"] = round(
        torch.cuda.max_memory_allocated() / 1_048_576, 1
    )
    output = Path(model.trainer.save_dir)
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
