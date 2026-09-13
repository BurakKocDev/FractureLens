from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from fracturelens.data.audit import load_display_image
from fracturelens.inference.pipeline import FractureLensPipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = (
    "classifier_miss_detector_rescue",
    "detector_miss_classifier_rescue",
    "both_miss",
    "classifier_only_false_alarm",
    "detector_only_false_alarm",
    "both_false_alarm",
    "consensus_true_positive",
)
COLORS = {"low": "#ffd477", "medium": "#ffad66", "high": "#ff5252"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a fused Track B error gallery")
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/error_gallery/fused_track_b"),
    )
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def category(row: dict[str, object]) -> str:
    label = int(row["label"])
    classifier = int(row["classifier_positive"])
    detector = int(row["detector_positive"])
    if label and not classifier and detector:
        return "classifier_miss_detector_rescue"
    if label and classifier and not detector:
        return "detector_miss_classifier_rescue"
    if label and not classifier and not detector:
        return "both_miss"
    if not label and classifier and detector:
        return "both_false_alarm"
    if not label and classifier:
        return "classifier_only_false_alarm"
    if not label and detector:
        return "detector_only_false_alarm"
    if label and classifier and detector:
        return "consensus_true_positive"
    return "consensus_true_negative"


def ranking(row: dict[str, object], group: str) -> float:
    classifier_score = float(row["classifier_probability"])
    detector_score = float(row.get("detector_max_confidence") or 0)
    if group in {
        "classifier_miss_detector_rescue",
        "detector_only_false_alarm",
        "both_false_alarm",
    }:
        return detector_score
    if group == "both_miss":
        return -classifier_score
    return classifier_score


def load_ground_truth(path: Path, width: int, height: int) -> list[list[float]]:
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
    return boxes


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def draw_overlay(
    image: Image.Image,
    ground_truth: list[list[float]],
    predictions: list[dict[str, object]],
) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    width = max(2, round(min(image.size) / 220))
    for box in ground_truth:
        draw.rectangle(tuple(box), outline="#39e58c", width=width)
    for index, prediction in enumerate(predictions, start=1):
        level = str(prediction["confidence_level"])
        color = COLORS[level]
        box = tuple(float(value) for value in prediction["xyxy"])
        draw.rectangle(box, outline=color, width=width)
        draw.text(
            (box[0] + 3, max(0, box[1] - 15)),
            f"P{index} {float(prediction['confidence']):.2f}",
            fill=color,
            font=font(max(10, width * 3), bold=True),
        )
    return annotated


def make_tile(
    row: dict[str, object],
    result: dict[str, object],
    annotated: Image.Image,
    size: tuple[int, int] = (320, 390),
) -> Image.Image:
    tile = Image.new("RGB", size, "#10212e")
    draw = ImageDraw.Draw(tile)
    image_area = (size[0] - 20, size[1] - 92)
    fitted = ImageOps.contain(annotated, image_area)
    tile.paste(fitted, ((size[0] - fitted.width) // 2, 10))
    y = size[1] - 74
    draw.text((10, y), str(row["image_id"]), fill="white", font=font(15, bold=True))
    y += 20
    classifier_score = float(result["classification"]["fracture_probability"])
    boxes = int(result["localization"]["count"])
    draw.text(
        (10, y),
        f"GT={row['label']}  Class={classifier_score:.2f}  Boxes={boxes}",
        fill="#b8cbd6",
        font=font(12),
    )
    y += 18
    draw.text(
        (10, y),
        f"{row['anatomy']} | {row['view']}",
        fill="#7f9aaa",
        font=font(11),
    )
    return tile


def main() -> int:
    args = parse_args()
    consistency_root = PROJECT_ROOT / "artifacts/consistency/track_b_densenet_yolov8s"
    detector_root = (
        PROJECT_ROOT
        / "artifacts/evaluations/track_b_detect_yolov8s_finetune_adamw_15ep_test_final"
    )
    joined = load_csv(consistency_root / "joined_predictions.csv")
    detector = {row["image_id"]: row for row in load_csv(detector_root / "test_per_image.csv")}
    manifest = {
        row["image_id"]: row
        for row in load_csv(PROJECT_ROOT / "manifests/fracatlas_v7_clean_split_seed20260912.csv")
        if row["split"] == "test"
    }
    rows: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    for source in joined:
        image_id = source["image_id"]
        row: dict[str, object] = {
            **source,
            **{f"detector_{key}": value for key, value in detector[image_id].items()},
            "anatomy": manifest[image_id]["anatomy"],
            "view": manifest[image_id]["view"],
        }
        row["category"] = category(row)
        counts[str(row["category"])] += 1
        rows.append(row)

    selected = []
    for group in CATEGORIES:
        candidates = [row for row in rows if row["category"] == group]
        candidates.sort(key=lambda row: ranking(row, group), reverse=True)
        selected.extend(candidates[: args.examples])

    pipeline = FractureLensPipeline(
        PROJECT_ROOT,
        PROJECT_ROOT / "configs/inference/track_b_fused.json",
    )
    output_root = (PROJECT_ROOT / args.output).resolve()
    image_output = output_root / "images"
    image_output.mkdir(parents=True, exist_ok=True)
    tiles: list[tuple[str, Image.Image]] = []
    selected_rows = []
    data_root = PROJECT_ROOT / "data/processed/track_b_detect_canonical"
    for row in selected:
        image_id = str(row["image_id"])
        image_path = data_root / "images/test" / image_id
        image = load_display_image(image_path).convert("RGB")
        result = pipeline.predict(image, image_id=image_id)
        ground_truth = load_ground_truth(
            data_root / "labels/test" / f"{Path(image_id).stem}.txt",
            image.width,
            image.height,
        )
        annotated = draw_overlay(image, ground_truth, result["localization"]["boxes"])
        annotated.save(image_output / image_id, quality=92)
        tiles.append((str(row["category"]), make_tile(row, result, annotated)))
        selected_rows.append(
            {
                "image_id": image_id,
                "category": row["category"],
                "label": int(row["label"]),
                "anatomy": row["anatomy"],
                "view": row["view"],
                "classifier_probability": result["classification"]["fracture_probability"],
                "predicted_boxes": result["localization"]["count"],
                "ground_truth_instances": len(ground_truth),
            }
        )

    tile_width, tile_height = tiles[0][1].size
    columns = 3
    heading_height = 44
    rows_needed = (len(tiles) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * tile_width, heading_height + rows_needed * tile_height),
        "#071018",
    )
    sheet_draw = ImageDraw.Draw(sheet)
    sheet_draw.text(
        (14, 11),
        "FractureLens Track B | Green=ground truth, yellow/orange/red=prediction",
        fill="white",
        font=font(16, bold=True),
    )
    for index, (group, tile) in enumerate(tiles):
        x = index % columns * tile_width
        y = heading_height + index // columns * tile_height
        sheet.paste(tile, (x, y))
        sheet_draw.text((x + 10, y + 10), group, fill="#42dbc5", font=font(11, bold=True))
    sheet.save(output_root / "gallery.jpg", quality=92)

    with (output_root / "selected_examples.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected_rows[0]))
        writer.writeheader()
        writer.writerows(selected_rows)
    summary = {
        "split": "test",
        "category_counts": dict(counts),
        "examples_per_category": args.examples,
        "selected_images": len(selected_rows),
        "legend": {
            "ground_truth": "green",
            "prediction_low": "yellow",
            "prediction_medium": "orange",
            "prediction_high": "red",
        },
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
