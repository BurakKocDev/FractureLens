from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "artifacts" / "matplotlib_config"))
os.environ.setdefault("MPLBACKEND", "Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Track B classifier errors")
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "evaluations"
            / "track_b_mobilenet_v3_small_15ep_test_corrected"
        ),
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "runs"
            / "track_b_mobilenet_v3_small_15ep"
            / "history.json"
        ),
    )
    parser.add_argument("--examples", type=int, default=20)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_boxes(dataset_root: Path) -> dict[str, list[list[float]]]:
    coco_paths = list((dataset_root / "Annotations" / "COCO JSON").glob("*.json"))
    if len(coco_paths) != 1:
        raise ValueError(f"Expected one COCO annotation file, found {len(coco_paths)}")
    coco = json.loads(coco_paths[0].read_text(encoding="utf-8"))
    image_names = {int(item["id"]): item["file_name"] for item in coco["images"]}
    boxes: dict[str, list[list[float]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        boxes[image_names[int(annotation["image_id"])]].append(annotation["bbox"])
    return boxes


def make_contact_sheet(
    rows: list[dict[str, object]],
    output: Path,
    dataset_root: Path,
    source_paths: dict[str, str],
    boxes_by_image: dict[str, list[list[float]]],
    title: str,
) -> None:
    from fracturelens.data.audit import load_display_image

    columns = 4
    tile_width, tile_height = 300, 280
    title_height = 34
    row_count = max(1, (len(rows) + columns - 1) // columns)
    sheet_size = (columns * tile_width, title_height + row_count * tile_height)
    sheet = Image.new("RGB", sheet_size, "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((8, 10), title, fill="black", font=font)
    for index, row in enumerate(rows):
        column = index % columns
        sheet_row = index // columns
        x0 = column * tile_width
        y0 = title_height + sheet_row * tile_height
        image_id = str(row["image_id"])
        source = load_display_image(dataset_root / source_paths[image_id]).convert("RGB")
        annotated = source.copy()
        annotation_draw = ImageDraw.Draw(annotated)
        for x, y, width, height in boxes_by_image.get(image_id, []):
            annotation_draw.rectangle((x, y, x + width, y + height), outline="#ff2d2d", width=5)
        fitted = ImageOps.contain(annotated, (tile_width - 12, tile_height - 52))
        image_x = x0 + (tile_width - fitted.width) // 2
        image_y = y0 + 3
        sheet.paste(fitted, (image_x, image_y))
        label = (
            f"{image_id}  y={row['label']} p={float(row['probability']):.3f}\n"
            f"{row['anatomy']} | {row['view']} | HW={row['hardware']}"
        )
        draw.multiline_text((x0 + 5, y0 + tile_height - 45), label, fill="black", font=font)
        draw.rectangle(
            (x0, y0, x0 + tile_width - 1, y0 + tile_height - 1), outline="#aaaaaa"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def plot_diagnostics(
    rows: list[dict[str, object]], history: list[dict[str, float]], output_root: Path
) -> None:
    import matplotlib.pyplot as plt

    labels = np.asarray([int(row["label"]) for row in rows])
    probabilities = np.asarray([float(row["probability"]) for row in rows])
    figure, axis = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 1, 21)
    axis.hist(probabilities[labels == 0], bins=bins, alpha=0.65, label="Non-fractured")
    axis.hist(probabilities[labels == 1], bins=bins, alpha=0.65, label="Fractured")
    axis.set(
        xlabel="Predicted fracture probability",
        ylabel="Images",
        title="Test score distribution",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_root / "score_distribution.png", dpi=180)
    plt.close(figure)

    bin_ids = np.minimum((probabilities * 10).astype(int), 9)
    confidence, observed, counts = [], [], []
    for bin_id in range(10):
        mask = bin_ids == bin_id
        if mask.any():
            confidence.append(float(probabilities[mask].mean()))
            observed.append(float(labels[mask].mean()))
            counts.append(int(mask.sum()))
    figure, axis = plt.subplots(figsize=(5.5, 5.5))
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    axis.plot(confidence, observed, marker="o", label="MobileNetV3-Small")
    for x, y, count in zip(confidence, observed, counts, strict=True):
        axis.annotate(str(count), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean predicted probability",
        ylabel="Observed fracture rate",
        title="Test reliability diagram (labels show bin size)",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_root / "reliability_diagram.png", dpi=180)
    plt.close(figure)

    epochs = [int(item["epoch"]) for item in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [item["train_loss"] for item in history], label="Train")
    axes[0].plot(epochs, [item["validation_loss"] for item in history], label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Weighted BCE", title="Loss")
    axes[0].legend()
    axes[1].plot(epochs, [item["validation_auroc"] for item in history], label="AUROC")
    axes[1].plot(epochs, [item["validation_auprc"] for item in history], label="AUPRC")
    axes[1].set(xlabel="Epoch", ylabel="Score", title="Validation ranking metrics")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_root / "training_curves.png", dpi=180)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    evaluation_root = args.evaluation.resolve()
    predictions = load_rows(evaluation_root / "test_predictions.csv")
    report = json.loads((evaluation_root / "evaluation.json").read_text(encoding="utf-8"))
    threshold = float(report["threshold"])
    rows: list[dict[str, object]] = [
        {
            **row,
            "label": int(row["label"]),
            "probability": float(row["probability"]),
            "prediction": int(row["prediction"]),
            "hardware": int(row["hardware"]),
            "error_type": (
                "false_negative"
                if int(row["label"]) == 1 and int(row["prediction"]) == 0
                else "false_positive"
                if int(row["label"]) == 0 and int(row["prediction"]) == 1
                else "correct"
            ),
        }
        for row in predictions
    ]
    false_negatives = sorted(
        (row for row in rows if row["error_type"] == "false_negative"),
        key=lambda row: float(row["probability"]),
    )
    false_positives = sorted(
        (row for row in rows if row["error_type"] == "false_positive"),
        key=lambda row: float(row["probability"]),
        reverse=True,
    )
    manifest_rows = load_rows(
        PROJECT_ROOT / "manifests" / "fracatlas_v7_clean_split_seed20260912.csv"
    )
    source_paths = {row["image_id"]: row["source_relative_path"] for row in manifest_rows}
    dataset_root = PROJECT_ROOT / "data" / "raw" / "fracatlas-v7" / "FracAtlas"
    boxes_by_image = load_boxes(dataset_root)
    output_root = evaluation_root / "error_analysis"
    output_root.mkdir(parents=True, exist_ok=True)
    make_contact_sheet(
        false_negatives[: args.examples],
        output_root / "worst_false_negatives.jpg",
        dataset_root,
        source_paths,
        boxes_by_image,
        "Most confident false negatives (red = ground-truth fracture box)",
    )
    make_contact_sheet(
        false_positives[: args.examples],
        output_root / "worst_false_positives.jpg",
        dataset_root,
        source_paths,
        boxes_by_image,
        "Most confident false positives",
    )
    error_rows = false_negatives + false_positives
    with (output_root / "ranked_errors.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(error_rows[0]))
        writer.writeheader()
        writer.writerows(error_rows)

    summary = {
        "threshold": threshold,
        "total_errors": len(error_rows),
        "false_negatives": len(false_negatives),
        "false_positives": len(false_positives),
        "false_negative_anatomy": dict(Counter(str(row["anatomy"]) for row in false_negatives)),
        "false_positive_anatomy": dict(Counter(str(row["anatomy"]) for row in false_positives)),
        "false_negative_view": dict(Counter(str(row["view"]) for row in false_negatives)),
        "false_positive_view": dict(Counter(str(row["view"]) for row in false_positives)),
        "most_confident_false_negative": false_negatives[0] if false_negatives else None,
        "most_confident_false_positive": false_positives[0] if false_positives else None,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    history = json.loads(args.history.read_text(encoding="utf-8"))
    plot_diagnostics(rows, history, output_root)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
