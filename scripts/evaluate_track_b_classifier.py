from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "artifacts" / "torch_cache"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a frozen Track B classifier")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--name", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def predict(
    model: object, loader: object, device: object, torch: object
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                probabilities = torch.sigmoid(model(images).squeeze(1)).float().cpu().tolist()
            for index, probability in enumerate(probabilities):
                rows.append(
                    {
                        "image_id": batch["image_id"][index],
                        "label": int(batch["label"][index]),
                        "probability": float(probability),
                        "anatomy": batch["anatomy"][index],
                        "view": batch["view"][index],
                        "hardware": int(batch["hardware"][index]),
                    }
                )
    return rows


def bootstrap_intervals(
    labels: list[int], probabilities: list[float], threshold: float, count: int
) -> dict[str, dict[str, float]]:
    from fracturelens.evaluation.classification import classification_metrics

    if count < 1:
        return {}
    rng = np.random.default_rng(20260912)
    values: dict[str, list[float]] = {
        key: []
        for key in (
            "auroc",
            "auprc",
            "sensitivity",
            "specificity",
            "balanced_accuracy",
            "f1",
            "ece_10_bin",
            "brier",
        )
    }
    y = np.asarray(labels)
    p = np.asarray(probabilities)
    for _ in range(count):
        indices = rng.integers(0, len(y), size=len(y))
        report = classification_metrics(y[indices], p[indices], threshold)
        for key in values:
            value = float(report[key])
            if np.isfinite(value):
                values[key].append(value)
    return {
        key: {
            "lower_95": float(np.percentile(metric_values, 2.5)),
            "upper_95": float(np.percentile(metric_values, 97.5)),
        }
        for key, metric_values in values.items()
        if metric_values
    }


def subgroup_reports(rows: list[dict[str, object]], threshold: float) -> dict[str, object]:
    from fracturelens.evaluation.classification import classification_metrics

    output: dict[str, object] = {}
    for field in ("anatomy", "view", "hardware"):
        field_report = {}
        for value in sorted({str(row[field]) for row in rows}):
            subset = [row for row in rows if str(row[field]) == value]
            labels = [int(row["label"]) for row in subset]
            if len(set(labels)) < 2:
                field_report[value] = {"n": len(subset), "note": "single-class subgroup"}
            else:
                field_report[value] = classification_metrics(
                    labels, [float(row["probability"]) for row in subset], threshold
                )
        output[field] = field_report
    return output


def write_predictions(path: Path, rows: list[dict[str, object]], threshold: float) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "image_id",
                "label",
                "probability",
                "prediction",
                "anatomy",
                "view",
                "hardware",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "prediction": int(float(row["probability"]) >= threshold)})


def main() -> int:
    args = parse_args()
    import torch
    import torchvision
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision.models import mobilenet_v3_small
    from torchvision.transforms import v2

    from fracturelens.data.classification import ManifestClassificationDataset
    from fracturelens.evaluation.classification import (
        classification_metrics,
        select_youden_threshold,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing an accidental CPU evaluation")
    weights_path = args.weights.resolve()
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    manifest_path = PROJECT_ROOT / "manifests" / "fracatlas_v7_clean_split_seed20260912.csv"
    if checkpoint["manifest_sha256"] != sha256(manifest_path):
        raise ValueError("Checkpoint and evaluation manifest hashes differ")
    output_root = PROJECT_ROOT / "artifacts" / "evaluations" / args.name
    if output_root.exists():
        raise FileExistsError(f"Evaluation name already exists: {output_root}")
    output_root.mkdir(parents=True)

    transform = v2.Compose(
        [
            v2.Resize(256, antialias=True),
            v2.CenterCrop(int(checkpoint["image_size"])),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=checkpoint["normalization_mean"], std=checkpoint["normalization_std"]
            ),
        ]
    )
    dataset_root = PROJECT_ROOT / "data" / "raw" / "fracatlas-v7" / "FracAtlas"
    datasets = {
        split: ManifestClassificationDataset(dataset_root, manifest_path, split, transform)
        for split in ("validation", "test")
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
        )
        for split, dataset in datasets.items()
    }
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda:0")
    model.to(device)

    validation_rows = predict(model, loaders["validation"], device, torch)
    threshold = select_youden_threshold(
        [int(row["label"]) for row in validation_rows],
        [float(row["probability"]) for row in validation_rows],
    )
    test_rows = predict(model, loaders["test"], device, torch)
    reports = {}
    for split, rows in (("validation", validation_rows), ("test", test_rows)):
        labels = [int(row["label"]) for row in rows]
        probabilities = [float(row["probability"]) for row in rows]
        reports[split] = {
            "metrics": classification_metrics(labels, probabilities, threshold),
            "subgroups": subgroup_reports(rows, threshold),
        }
        if split == "test":
            reports[split]["bootstrap_95_ci"] = bootstrap_intervals(
                labels, probabilities, threshold, args.bootstrap
            )
        write_predictions(output_root / f"{split}_predictions.csv", rows, threshold)

    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "weights": str(weights_path),
        "weights_sha256": sha256(weights_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "model": checkpoint["model_name"],
        "manifest_sha256": sha256(manifest_path),
        "threshold_selection": "Youden J on validation only; frozen for test",
        "threshold": threshold,
        "bootstrap_replicates": args.bootstrap,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "reports": reports,
    }
    (output_root / "evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["reports"]["test"]["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
