from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from fracturelens.evaluation.detection import (
    greedy_match,
    non_max_suppression_indices,
    pairwise_iou_xyxy,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / "artifacts/ultralytics_config"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Track B validation box overlap")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to artifacts/analysis/track_b_overlap_<split>",
    )
    return parser.parse_args()


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


def evaluate_nms(records: list[dict[str, object]], nms_iou: float) -> dict[str, float | int]:
    tp = fp = fn = boxes_before = boxes_after = 0
    negative_images = negative_images_with_alarm = 0
    for record in records:
        targets = np.asarray(record["targets"])
        boxes = np.asarray(record["boxes"])
        confidences = np.asarray(record["confidences"])
        keep = non_max_suppression_indices(boxes, confidences, nms_iou)
        kept_boxes = boxes[keep]
        kept_confidences = confidences[keep]
        matches = greedy_match(targets, kept_boxes, kept_confidences, 0.5)
        image_tp = len(matches)
        image_fp = len(kept_boxes) - image_tp
        tp += image_tp
        fp += image_fp
        fn += len(targets) - image_tp
        boxes_before += len(boxes)
        boxes_after += len(kept_boxes)
        if len(targets) == 0:
            negative_images += 1
            negative_images_with_alarm += int(image_fp > 0)
    precision = tp / max(1, tp + fp)
    sensitivity = tp / max(1, tp + fn)
    return {
        "nms_iou": nms_iou,
        "boxes_before": boxes_before,
        "boxes_after": boxes_after,
        "boxes_removed": boxes_before - boxes_after,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "lesion_sensitivity": sensitivity,
        "f1": 2 * precision * sensitivity / max(1e-12, precision + sensitivity),
        "negative_image_false_alarm_rate": negative_images_with_alarm / max(1, negative_images),
    }


def main() -> int:
    args = parse_args()
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing CPU analysis")
    config = json.loads((PROJECT_ROOT / "configs/inference/track_b_fused.json").read_text())
    detector = config["detector"]
    dataset_root = PROJECT_ROOT / "data/processed/track_b_detect_canonical"
    image_root = dataset_root / "images" / args.split
    label_root = dataset_root / "labels" / args.split
    image_paths = sorted(path for path in image_root.iterdir() if path.is_file())
    model = YOLO(str(PROJECT_ROOT / detector["weights"]))
    records: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    overlap_pair_counts = {"iou_0.30": 0, "iou_0.50": 0, "iou_0.70": 0}
    images_with_overlap = {key: 0 for key in overlap_pair_counts}

    for start in range(0, len(image_paths), args.batch):
        paths = image_paths[start : start + args.batch]
        predictions = model.predict(
            source=[str(path) for path in paths],
            imgsz=int(detector["image_size"]),
            conf=float(detector["confidence_threshold"]),
            iou=0.7,
            batch=len(paths),
            device=0,
            verbose=False,
        )
        for image_path, prediction in zip(paths, predictions, strict=True):
            height, width = prediction.orig_shape
            boxes = prediction.boxes.xyxy.cpu().numpy()
            confidences = prediction.boxes.conf.cpu().numpy()
            targets = load_targets(label_root / f"{image_path.stem}.txt", width, height)
            records.append(
                {
                    "image_id": image_path.name,
                    "targets": targets,
                    "boxes": boxes,
                    "confidences": confidences,
                }
            )
            pair_ious = pairwise_iou_xyxy(boxes, boxes)
            upper = pair_ious[np.triu_indices(len(boxes), k=1)] if len(boxes) > 1 else []
            row: dict[str, object] = {
                "image_id": image_path.name,
                "ground_truth_instances": len(targets),
                "predicted_boxes": len(boxes),
                "max_confidence": float(confidences.max()) if len(confidences) else "",
            }
            for value in (0.30, 0.50, 0.70):
                key = f"iou_{value:.2f}"
                count = int(np.sum(np.asarray(upper) >= value))
                row[f"overlap_pairs_{key}"] = count
                overlap_pair_counts[key] += count
                images_with_overlap[key] += int(count > 0)
            rows.append(row)

    candidate_ious = (0.30, 0.40, 0.50, 0.60, 0.70) if args.split == "val" else (0.50, 0.70)
    nms_results = [evaluate_nms(records, value) for value in candidate_ious]
    relative_output = args.output or Path(f"artifacts/analysis/track_b_overlap_{args.split}")
    output_root = (PROJECT_ROOT / relative_output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "per_image.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "split": args.split,
        "images": len(records),
        "confidence_threshold": detector["confidence_threshold"],
        "source_nms_iou": 0.7,
        "images_with_predictions": sum(int(row["predicted_boxes"] > 0) for row in rows),
        "multi_prediction_images": sum(int(row["predicted_boxes"] > 1) for row in rows),
        "maximum_boxes_on_one_image": max(int(row["predicted_boxes"]) for row in rows),
        "overlap_pair_counts": overlap_pair_counts,
        "images_with_overlap": images_with_overlap,
        "candidate_nms_results": nms_results,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
