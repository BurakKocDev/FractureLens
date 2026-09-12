from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from fracturelens.evaluation.detection import greedy_match

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / "artifacts" / "ultralytics_config"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "artifacts" / "matplotlib_config"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "artifacts" / "torch_cache"))
for variable in ("YOLO_CONFIG_DIR", "MPLCONFIGDIR", "TORCH_HOME"):
    Path(os.environ[variable]).mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create fixed-threshold Track A error analysis")
    parser.add_argument(
        "--weights",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "runs"
            / "track_a_detect_fidelity_sgd_b16_30ep"
            / "weights"
            / "best.pt"
        ),
    )
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument(
        "--name", default="track_a_detect_fidelity_sgd_b16_30ep_test_errors_by_group"
    )
    parser.add_argument("--montage-size", type=int, default=16)
    return parser.parse_args()


def load_yolo_targets(label_path: Path, width: int, height: int) -> np.ndarray:
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        _, center_x, center_y, box_width, box_height = map(float, line.split())
        x1 = (center_x - box_width / 2) * width
        y1 = (center_y - box_height / 2) * height
        x2 = (center_x + box_width / 2) * width
        y2 = (center_y + box_height / 2) * height
        boxes.append([x1, y1, x2, y2])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def load_metadata(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["anatomy"] = "+".join(
            value for value in ("hand", "leg", "hip", "shoulder", "mixed") if row[value] == "1"
        )
        row["view"] = "+".join(
            value for value in ("frontal", "lateral", "oblique") if row[value] == "1"
        )
    return {row["image_id"]: row for row in rows}


def summarize_groups(rows: list[dict], key: str) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    summaries = []
    for value, members in groups.items():
        totals = {
            metric: sum(int(member[metric]) for member in members)
            for metric in ("gt", "pred", "tp", "fp", "fn")
        }
        summaries.append(
            {
                "group": key,
                "value": value,
                "images": len(members),
                **totals,
                "precision": totals["tp"] / max(1, totals["tp"] + totals["fp"]),
                "recall": totals["tp"] / max(1, totals["tp"] + totals["fn"]),
            }
        )
    return sorted(summaries, key=lambda item: (item["recall"], -item["gt"], item["value"]))


def render_overlay(
    image_path: Path,
    targets: np.ndarray,
    predictions: np.ndarray,
    confidences: np.ndarray,
    matched_prediction_indices: set[int],
) -> Image.Image:
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    line_width = max(2, round(min(image.size) / 250))
    for box in targets:
        draw.rectangle(tuple(box), outline="#00d26a", width=line_width)
    for index, (box, confidence) in enumerate(zip(predictions, confidences, strict=True)):
        color = "#2684ff" if index in matched_prediction_indices else "#ff3b30"
        draw.rectangle(tuple(box), outline=color, width=line_width)
        draw.text((box[0] + 3, max(0, box[1] - 14)), f"{confidence:.2f}", fill=color)
    return image


def create_montage(items: list[tuple[dict, Image.Image]], destination: Path) -> None:
    tile_width, tile_height = 420, 300
    columns = 4
    rows = max(1, (len(items) + columns - 1) // columns)
    canvas = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (row, image) in enumerate(items):
        image.thumbnail((tile_width - 10, tile_height - 42))
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        canvas.paste(image, (x + 5, y + 24))
        label = (
            f"{row['image_id']} | GT {row['gt']} P {row['pred']} "
            f"TP {row['tp']} FP {row['fp']} FN {row['fn']}"
        )
        draw.text((x + 5, y + 5), label, fill="black")
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=92)


def main() -> int:
    args = parse_args()
    if not 0 <= args.conf <= 1 or not 0 <= args.match_iou <= 1:
        raise ValueError("conf and match-iou must be within [0, 1]")
    weights = args.weights.resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)

    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing an accidental CPU analysis")
    dataset_root = PROJECT_ROOT / "data" / "processed" / "track_a_detect"
    image_root = dataset_root / "images" / args.split
    label_root = dataset_root / "labels" / args.split
    output_root = PROJECT_ROOT / "artifacts" / "error_analysis" / args.name
    if output_root.exists():
        raise FileExistsError(output_root)
    overlay_root = output_root / "overlays"
    overlay_root.mkdir(parents=True)

    image_paths = sorted(path for path in image_root.iterdir() if path.is_file())
    metadata = load_metadata(
        PROJECT_ROOT / "data" / "raw" / "fracatlas-v7" / "FracAtlas" / "dataset.csv"
    )
    model = YOLO(str(weights))
    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=600,
        conf=args.conf,
        device=0,
        stream=True,
        verbose=False,
    )
    rows = []
    montage_candidates = []
    for image_path, result in zip(image_paths, results, strict=True):
        height, width = result.orig_shape
        targets = load_yolo_targets(label_root / f"{image_path.stem}.txt", width, height)
        predictions = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        matches = greedy_match(targets, predictions, confidences, args.match_iou)
        matched_prediction_indices = {match.prediction_index for match in matches}
        row = {
            "image_id": image_path.name,
            "anatomy": metadata[image_path.name]["anatomy"],
            "view": metadata[image_path.name]["view"],
            "hardware": metadata[image_path.name]["hardware"],
            "multiscan": metadata[image_path.name]["multiscan"],
            "gt": len(targets),
            "pred": len(predictions),
            "tp": len(matches),
            "fp": len(predictions) - len(matches),
            "fn": len(targets) - len(matches),
            "mean_matched_iou": (
                round(sum(match.iou for match in matches) / len(matches), 6) if matches else ""
            ),
            "max_confidence": round(float(confidences.max()), 6) if len(confidences) else "",
        }
        rows.append(row)
        overlay = render_overlay(
            image_path, targets, predictions, confidences, matched_prediction_indices
        )
        overlay.save(overlay_root / image_path.name, quality=92)
        montage_candidates.append((row, overlay))

    rows.sort(key=lambda row: (-int(row["fn"]), -int(row["fp"]), str(row["image_id"])))
    with (output_root / "per_image_errors.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    group_summaries = [
        *summarize_groups(rows, "anatomy"),
        *summarize_groups(rows, "view"),
        *summarize_groups(rows, "hardware"),
    ]
    with (output_root / "group_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(group_summaries[0]))
        writer.writeheader()
        writer.writerows(group_summaries)
    ordered_overlays = {row["image_id"]: image for row, image in montage_candidates}
    worst = [(row, ordered_overlays[row["image_id"]]) for row in rows[: args.montage_size]]
    create_montage(worst, output_root / "worst_cases.jpg")

    totals = {key: sum(int(row[key]) for row in rows) for key in ("gt", "pred", "tp", "fp", "fn")}
    summary = {
        "weights": str(weights),
        "split": args.split,
        "confidence_threshold": args.conf,
        "match_iou_threshold": args.match_iou,
        "image_count": len(rows),
        **totals,
        "fixed_threshold_precision": totals["tp"] / max(1, totals["tp"] + totals["fp"]),
        "fixed_threshold_recall": totals["tp"] / max(1, totals["tp"] + totals["fn"]),
        "by_anatomy": summarize_groups(rows, "anatomy"),
        "by_hardware": summarize_groups(rows, "hardware"),
        "legend": {"ground_truth": "green", "matched_prediction": "blue", "false_positive": "red"},
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
