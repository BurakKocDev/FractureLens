from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from fracturelens.evaluation.detection import greedy_match

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / "artifacts/ultralytics_config"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "artifacts/matplotlib_config"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the frozen Track B detector")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--name", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_targets(path: Path, width: int, height: int) -> np.ndarray:
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        _, center_x, center_y, box_width, box_height = map(float, line.split())
        boxes.append(
            [
                (center_x - box_width / 2) * width,
                (center_y - box_height / 2) * height,
                (center_x + box_width / 2) * width,
                (center_y + box_height / 2) * height,
            ]
        )
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def collect(
    model: object, dataset_root: Path, split: str, batch: int
) -> list[dict[str, object]]:
    image_root = dataset_root / "images" / split
    label_root = dataset_root / "labels" / split
    image_paths = sorted(path for path in image_root.iterdir() if path.is_file())
    records = []
    for start in range(0, len(image_paths), batch):
        paths = image_paths[start : start + batch]
        predictions = model.predict(
            source=[str(path) for path in paths],
            imgsz=640,
            conf=0.001,
            batch=len(paths),
            device=0,
            stream=False,
            verbose=False,
        )
        for image_path, result in zip(paths, predictions, strict=True):
            height, width = result.orig_shape
            records.append(
                {
                    "image_id": image_path.name,
                    "targets": load_targets(
                        label_root / f"{image_path.stem}.txt", width, height
                    ),
                    "boxes": result.boxes.xyxy.cpu().numpy(),
                    "confidences": result.boxes.conf.cpu().numpy(),
                }
            )
    return records


def summarize(
    records: list[dict[str, object]], threshold: float, match_iou: float
) -> dict[str, float | int]:
    totals = {"tp": 0, "fp": 0, "fn": 0}
    negative_images = 0
    negative_images_with_fp = 0
    for record in records:
        targets = np.asarray(record["targets"])
        confidences = np.asarray(record["confidences"])
        boxes = np.asarray(record["boxes"])
        keep = confidences >= threshold
        matches = greedy_match(targets, boxes[keep], confidences[keep], match_iou)
        tp = len(matches)
        fp = int(keep.sum()) - tp
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += len(targets) - tp
        if len(targets) == 0:
            negative_images += 1
            negative_images_with_fp += int(fp > 0)
    precision = totals["tp"] / max(1, totals["tp"] + totals["fp"])
    recall = totals["tp"] / max(1, totals["tp"] + totals["fn"])
    return {
        **totals,
        "precision": precision,
        "lesion_sensitivity": recall,
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
        "negative_images": negative_images,
        "false_positives_per_negative_image": totals["fp"] / max(1, negative_images),
        "negative_image_false_alarm_rate": negative_images_with_fp / max(1, negative_images),
    }


def select_threshold(records: list[dict[str, object]], match_iou: float) -> tuple[float, dict]:
    candidates = np.linspace(0.01, 0.80, 160)
    scored = []
    for threshold in candidates:
        metrics = summarize(records, float(threshold), match_iou)
        scored.append({"threshold": float(threshold), **metrics})
    best = max(
        scored,
        key=lambda row: (
            float(row["f1"]),
            -float(row["false_positives_per_negative_image"]),
        ),
    )
    return float(best["threshold"]), {"selected": best, "curve": scored}


def per_image_rows(
    records: list[dict[str, object]], threshold: float, match_iou: float
) -> list[dict[str, object]]:
    rows = []
    for record in records:
        targets = np.asarray(record["targets"])
        confidences = np.asarray(record["confidences"])
        boxes = np.asarray(record["boxes"])
        keep = confidences >= threshold
        kept_confidences = confidences[keep]
        matches = greedy_match(targets, boxes[keep], kept_confidences, match_iou)
        rows.append(
            {
                "image_id": record["image_id"],
                "ground_truth_instances": len(targets),
                "predictions": int(keep.sum()),
                "tp": len(matches),
                "fp": int(keep.sum()) - len(matches),
                "fn": len(targets) - len(matches),
                "max_confidence": float(kept_confidences.max())
                if len(kept_confidences)
                else "",
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    import torch
    import ultralytics
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing CPU evaluation")
    weights = args.weights.resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    output_root = PROJECT_ROOT / "artifacts/evaluations" / args.name
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    dataset_root = PROJECT_ROOT / "data/processed/track_b_detect_canonical"
    model = YOLO(str(weights))

    validation_records = collect(model, dataset_root, "val", args.batch)
    threshold, validation_selection = select_threshold(validation_records, args.match_iou)
    test_records = collect(model, dataset_root, "test", args.batch)
    test_fixed = summarize(test_records, threshold, args.match_iou)
    froc = [
        {"threshold": float(value), **summarize(test_records, float(value), args.match_iou)}
        for value in np.linspace(0.01, 0.80, 160)
    ]
    standard = model.val(
        data=str(dataset_root / "data.yaml"),
        split="test",
        imgsz=640,
        batch=args.batch,
        device=0,
        workers=2,
        plots=True,
        project=str(output_root),
        name="ultralytics_test",
        exist_ok=False,
        verbose=True,
    )
    rows = per_image_rows(test_records, threshold, args.match_iou)
    with (output_root / "test_per_image.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "weights": str(weights),
        "weights_sha256": sha256(weights),
        "threshold_selection": "maximum instance F1 on validation only",
        "match_iou": args.match_iou,
        "confidence_threshold": threshold,
        "validation_selection": validation_selection,
        "test_fixed_threshold": test_fixed,
        "test_froc_curve": froc,
        "test_standard_metrics": {
            key: float(value) for key, value in standard.results_dict.items()
        },
        "speed_ms_per_image": {key: float(value) for key, value in standard.speed.items()},
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
    (output_root / "evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"threshold": threshold, "test": test_fixed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
