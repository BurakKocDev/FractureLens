from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / "artifacts" / "ultralytics_config"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "artifacts" / "matplotlib_config"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "artifacts" / "torch_cache"))
for variable in ("YOLO_CONFIG_DIR", "MPLCONFIGDIR", "TORCH_HOME"):
    Path(os.environ[variable]).mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a pinned FractureLens Track A experiment")
    parser.add_argument("--task", choices=("detect", "segment"), default="detect")
    parser.add_argument(
        "--profile",
        choices=("modern", "fidelity-sgd"),
        default="modern",
        help=(
            "modern keeps current Ultralytics defaults; fidelity-sgd mirrors the "
            "official optimizer"
        ),
    )
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


def git_dirty() -> bool:
    return bool(
        subprocess.run(
            ["git", "-c", f"safe.directory={PROJECT_ROOT.as_posix()}", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    config_path = PROJECT_ROOT / "configs" / "experiment" / "track_a_official_reproduction.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_config = next(item for item in config["runs"] if item["task"] == args.task)
    epochs = args.epochs or int(run_config["epochs"])
    default_batch_key = "fidelity_batch" if args.profile == "fidelity-sgd" else "first_batch"
    batch = args.batch or int(config["local_execution"][default_batch_key])
    name = args.name or f"track_a_{args.task}_{args.profile}_{epochs}ep"
    output_root = PROJECT_ROOT / "artifacts" / "runs"
    run_root = output_root / name
    data_yaml = PROJECT_ROOT / "data" / "processed" / f"track_a_{args.task}" / "data.yaml"
    if not data_yaml.is_file():
        preparation_script = (
            "prepare_track_a_yolo.py"
            if args.task == "detect"
            else "prepare_track_a_segmentation.py"
        )
        raise FileNotFoundError(f"Run scripts/{preparation_script} first")
    if run_root.exists():
        raise FileExistsError(f"Run name already exists: {run_root}")

    provenance = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "training_script_sha256": sha256(Path(__file__).resolve()),
        "experiment_config_sha256": sha256(config_path),
        "data_yaml_sha256": sha256(data_yaml),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision_cuda": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "task": args.task,
        "profile": args.profile,
        "model": run_config["model"],
        "epochs": epochs,
        "requested_imgsz": int(run_config["imgsz"]),
        "batch": batch,
        "seed": 0,
    }

    train_overrides = {}
    if args.profile == "fidelity-sgd":
        official = config["official_notebook_observed"]
        train_overrides = {
            "optimizer": official["optimizer"],
            "lr0": float(official["lr0"]),
            "lrf": float(official["lrf"]),
            "momentum": float(official["momentum"]),
            "weight_decay": float(official["weight_decay"]),
        }
    provenance["train_overrides"] = train_overrides
    provenance["fidelity_caveat"] = (
        config["data"]["split_warning"] if args.profile == "fidelity-sgd" else None
    )

    model = YOLO(run_config["model"])
    model_path = Path(model.ckpt_path).resolve()
    provenance["model_path"] = str(model_path)
    provenance["model_sha256"] = sha256(model_path)
    torch.cuda.reset_peak_memory_stats()
    started_at = time.perf_counter()
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
        **train_overrides,
    )
    provenance["elapsed_seconds"] = round(time.perf_counter() - started_at, 3)
    provenance["peak_cuda_memory_mib"] = round(
        torch.cuda.max_memory_allocated() / 1_048_576, 1
    )
    provenance["effective_imgsz"] = getattr(model.trainer.train_loader.dataset, "imgsz", None)
    actual_run_root = Path(model.trainer.save_dir)
    (actual_run_root / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
