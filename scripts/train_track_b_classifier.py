from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "artifacts" / "torch_cache"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
Path(os.environ["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Track B mobile classifier baseline")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--name", default="track_b_mobilenet_v3_small_15ep")
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


def seed_everything(seed: int, torch: object) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.batch < 1 or args.workers < 0 or args.patience < 1:
        raise ValueError("Invalid training arguments")

    import torch
    import torchvision
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
    from torchvision.transforms import v2

    from fracturelens.data.classification import ManifestClassificationDataset
    from fracturelens.evaluation.classification import (
        classification_metrics,
        select_youden_threshold,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing an accidental CPU training run")

    seed = 20260912
    seed_everything(seed, torch)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    run_root = PROJECT_ROOT / "artifacts" / "runs" / args.name
    if run_root.exists():
        raise FileExistsError(f"Run name already exists: {run_root}")
    run_root.mkdir(parents=True)

    dataset_root = PROJECT_ROOT / "data" / "raw" / "fracatlas-v7" / "FracAtlas"
    manifest_path = PROJECT_ROOT / "manifests" / "fracatlas_v7_clean_split_seed20260912.csv"
    config_path = PROJECT_ROOT / "configs" / "experiment" / "track_b_clean_benchmark.json"
    weights = MobileNet_V3_Small_Weights.DEFAULT
    normalization = weights.transforms()
    train_transform = v2.Compose(
        [
            v2.RandomResizedCrop(224, scale=(0.8, 1.0), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(7),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=normalization.mean, std=normalization.std),
        ]
    )
    eval_transform = v2.Compose(
        [
            v2.Resize(256, antialias=True),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=normalization.mean, std=normalization.std),
        ]
    )
    train_dataset = ManifestClassificationDataset(
        dataset_root, manifest_path, "train", train_transform
    )
    validation_dataset = ManifestClassificationDataset(
        dataset_root, manifest_path, "validation", eval_transform
    )
    generator = torch.Generator().manual_seed(seed)
    common_loader = {
        "batch_size": args.batch,
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, drop_last=False, **common_loader
    )
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, drop_last=False, **common_loader
    )

    model = mobilenet_v3_small(weights=weights)
    input_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(input_features, 1)
    device = torch.device("cuda:0")
    model.to(device)
    negative_count = len(train_dataset) - sum(train_dataset.labels)
    positive_count = sum(train_dataset.labels)
    pos_weight = torch.tensor([negative_count / positive_count], device=device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    provenance = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "script_sha256": sha256(Path(__file__).resolve()),
        "manifest_sha256": sha256(manifest_path),
        "experiment_config_sha256": sha256(config_path),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "model": "mobilenet_v3_small",
        "pretrained_weights": str(weights),
        "image_size": 224,
        "epochs_requested": args.epochs,
        "batch": args.batch,
        "workers": args.workers,
        "learning_rate": args.lr,
        "optimizer": "AdamW",
        "weight_decay": 1e-4,
        "loss": "BCEWithLogitsLoss",
        "positive_weight": float(pos_weight.item()),
        "seed": seed,
        "split_counts": {"train": len(train_dataset), "validation": len(validation_dataset)},
        "test_accessed": False,
    }
    (run_root / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    history: list[dict[str, float | int]] = []
    best_key = (-float("inf"), -float("inf"))
    epochs_without_improvement = 0
    torch.cuda.reset_peak_memory_stats()
    started_at = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].float().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(images).squeeze(1)
                loss = loss_function(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * labels.numel()
            seen += labels.numel()

        model.eval()
        validation_labels: list[int] = []
        validation_probabilities: list[float] = []
        validation_loss = 0.0
        validation_seen = 0
        with torch.inference_mode():
            for batch in validation_loader:
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].float().to(device, non_blocking=True)
                with torch.amp.autocast("cuda"):
                    logits = model(images).squeeze(1)
                    loss = loss_function(logits, labels)
                probabilities = torch.sigmoid(logits)
                validation_labels.extend(labels.int().cpu().tolist())
                validation_probabilities.extend(probabilities.float().cpu().tolist())
                validation_loss += float(loss.item()) * labels.numel()
                validation_seen += labels.numel()

        threshold = select_youden_threshold(validation_labels, validation_probabilities)
        metrics = classification_metrics(
            validation_labels, validation_probabilities, threshold=threshold
        )
        record = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": running_loss / seen,
            "validation_loss": validation_loss / validation_seen,
            "validation_auroc": float(metrics["auroc"]),
            "validation_auprc": float(metrics["auprc"]),
            "validation_sensitivity": float(metrics["sensitivity"]),
            "validation_specificity": float(metrics["specificity"]),
            "validation_threshold": threshold,
        }
        history.append(record)
        (run_root / "history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(record), flush=True)

        current_key = (record["validation_auroc"], record["validation_auprc"])
        if current_key > best_key:
            best_key = current_key
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_name": "mobilenet_v3_small",
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_metrics": metrics,
                    "validation_threshold": threshold,
                    "manifest_sha256": provenance["manifest_sha256"],
                    "image_size": 224,
                    "normalization_mean": list(normalization.mean),
                    "normalization_std": list(normalization.std),
                },
                run_root / "best.pt",
            )
        else:
            epochs_without_improvement += 1
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, run_root / "last.pt")
        scheduler.step()
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after epoch {epoch}", flush=True)
            break

    provenance["epochs_completed"] = len(history)
    provenance["best_epoch"] = int(torch.load(run_root / "best.pt", map_location="cpu")["epoch"])
    provenance["best_validation_auroc"] = best_key[0]
    provenance["best_validation_auprc"] = best_key[1]
    provenance["elapsed_seconds"] = round(time.perf_counter() - started_at, 3)
    provenance["peak_cuda_memory_mib"] = round(
        torch.cuda.max_memory_allocated() / 1_048_576, 1
    )
    (run_root / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
