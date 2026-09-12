from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

OFFICIAL_SPLIT_FILES = {
    "train": "train.csv",
    "val": "valid.csv",
    "test": "test.csv",
}


def read_image_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "image_id" not in rows[0]:
        raise ValueError(f"Missing image_id column or rows in {path}")
    image_ids = [row["image_id"] for row in rows]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError(f"Duplicate image id inside {path}")
    return image_ids


def normalized_text_sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        digest.update(b"\0")
    return digest.hexdigest()


def validate_yolo_label(path: Path) -> int:
    count = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Expected five YOLO fields in {path}:{line_number}")
        class_id = int(parts[0])
        values = [float(value) for value in parts[1:]]
        if class_id != 0:
            raise ValueError(f"Unexpected class id {class_id} in {path}:{line_number}")
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"Out-of-range coordinate in {path}:{line_number}")
        if values[2] <= 0.0 or values[3] <= 0.0:
            raise ValueError(f"Non-positive box size in {path}:{line_number}")
        count += 1
    if count == 0:
        raise ValueError(f"Positive-only Track A label is empty: {path}")
    return count


def coco_polygon_to_yolo(points: list[float], width: int, height: int) -> str:
    if len(points) < 6 or len(points) % 2:
        raise ValueError("A polygon must contain at least three xy points")
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    normalized = []
    for index, value in enumerate(points):
        denominator = width if index % 2 == 0 else height
        coordinate = float(value) / denominator
        if not -1e-9 <= coordinate <= 1.0 + 1e-9:
            raise ValueError(f"Polygon coordinate outside image bounds: {coordinate}")
        coordinate = min(1.0, max(0.0, coordinate))
        normalized.append(f"{coordinate:.8f}")
    return "0 " + " ".join(normalized)


def ensure_hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not os.path.samefile(source, destination):
            raise FileExistsError(f"Unexpected existing output: {destination}")
        return
    os.link(source, destination)


def prepare_track_a_detection(
    dataset_root: Path,
    output_root: Path,
    portable_manifest_path: Path | None = None,
) -> dict:
    split_root = dataset_root / "Utilities" / "Fracture Split"
    image_root = dataset_root / "images" / "Fractured"
    label_root = dataset_root / "Annotations" / "YOLO"
    expected = {"train": 574, "val": 82, "test": 63}
    split_ids: dict[str, list[str]] = {}
    manifest_rows: list[dict[str, str | int]] = []
    instance_counts: Counter[str] = Counter()

    for split, filename in OFFICIAL_SPLIT_FILES.items():
        image_ids = read_image_ids(split_root / filename)
        if len(image_ids) != expected[split]:
            raise ValueError(
                f"Official {split} count changed: expected {expected[split]}, got {len(image_ids)}"
            )
        split_ids[split] = image_ids
        for image_id in image_ids:
            source_image = image_root / image_id
            source_label = label_root / Path(image_id).with_suffix(".txt")
            if not source_image.is_file() or not source_label.is_file():
                raise FileNotFoundError(f"Missing Track A pair for {image_id}")
            instance_count = validate_yolo_label(source_label)
            ensure_hardlink(source_image, output_root / "images" / split / image_id)
            ensure_hardlink(
                source_label,
                output_root / "labels" / split / Path(image_id).with_suffix(".txt"),
            )
            instance_counts[split] += instance_count
            manifest_rows.append(
                {
                    "image_id": image_id,
                    "split": split,
                    "instance_count": instance_count,
                    "image_relative_path": f"images/{split}/{image_id}",
                    "label_relative_path": (
                        f"labels/{split}/{Path(image_id).with_suffix('.txt').name}"
                    ),
                }
            )

    all_ids = [image_id for values in split_ids.values() for image_id in values]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Official Track A splits overlap")

    output_root.mkdir(parents=True, exist_ok=True)
    yaml_text = "\n".join(
        [
            f"path: {output_root.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: fracture",
            "",
        ]
    )
    (output_root / "data.yaml").write_text(yaml_text, encoding="utf-8", newline="\n")

    summary = {
        "dataset": "FracAtlas v7 official positive-only split",
        "task": "detection",
        "link_mode": "hardlink",
        "image_counts": {split: len(ids) for split, ids in split_ids.items()},
        "instance_counts": dict(instance_counts),
        "unique_images": len(set(all_ids)),
        "split_overlap": 0,
        "source_split_sha256": normalized_text_sha256(
            split_root / name for name in OFFICIAL_SPLIT_FILES.values()
        ),
        "source_label_sha256": normalized_text_sha256(
            label_root / Path(image_id).with_suffix(".txt") for image_id in all_ids
        ),
    }
    (output_root / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    manifest_paths = [output_root / "official_split_manifest.csv"]
    if portable_manifest_path is not None:
        portable_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_paths.append(portable_manifest_path)
    for manifest_path in manifest_paths:
        with manifest_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
            writer.writeheader()
            writer.writerows(manifest_rows)
    return summary


def prepare_track_a_segmentation(
    dataset_root: Path,
    output_root: Path,
    portable_manifest_path: Path | None = None,
) -> dict:
    split_root = dataset_root / "Utilities" / "Fracture Split"
    image_root = dataset_root / "images" / "Fractured"
    coco_path = dataset_root / "Annotations" / "COCO JSON" / "COCO_fracture_masks.json"
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    if coco.get("categories") != [{"id": 1, "name": "fractured"}]:
        raise ValueError("Unexpected COCO category definition")

    images_by_id = {int(item["id"]): item for item in coco["images"]}
    image_ids_by_name = {item["file_name"]: int(item["id"]) for item in coco["images"]}
    if len(images_by_id) != len(image_ids_by_name):
        raise ValueError("Duplicate COCO image id or filename")
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in coco["annotations"]:
        if int(annotation["category_id"]) != 1 or int(annotation["iscrowd"]) != 0:
            raise ValueError(f"Unsupported COCO annotation: {annotation['id']}")
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    expected = {"train": 574, "val": 82, "test": 63}
    split_ids = {
        split: read_image_ids(split_root / filename)
        for split, filename in OFFICIAL_SPLIT_FILES.items()
    }
    for split, image_ids in split_ids.items():
        if len(image_ids) != expected[split]:
            raise ValueError(
                f"Official {split} count changed: expected {expected[split]}, got {len(image_ids)}"
            )
    all_ids = [image_id for image_ids in split_ids.values() for image_id in image_ids]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Official Track A splits overlap")
    if set(all_ids) != set(image_ids_by_name):
        raise ValueError("COCO segmentation image membership differs from official splits")

    manifest_rows = []
    instance_counts: Counter[str] = Counter()
    polygon_point_counts: list[int] = []
    for split, image_ids in split_ids.items():
        for image_name in image_ids:
            coco_image = images_by_id[image_ids_by_name[image_name]]
            width, height = int(coco_image["width"]), int(coco_image["height"])
            annotations = annotations_by_image[int(coco_image["id"])]
            if not annotations:
                raise ValueError(f"Positive image has no COCO polygon: {image_name}")
            label_lines = []
            for annotation in sorted(annotations, key=lambda item: int(item["id"])):
                segmentation = annotation["segmentation"]
                if len(segmentation) != 1:
                    raise ValueError(f"Expected one polygon for annotation {annotation['id']}")
                points = segmentation[0]
                label_lines.append(coco_polygon_to_yolo(points, width, height))
                polygon_point_counts.append(len(points) // 2)

            source_image = image_root / image_name
            destination_image = output_root / "images" / split / image_name
            ensure_hardlink(source_image, destination_image)
            label_path = output_root / "labels" / split / Path(image_name).with_suffix(".txt")
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8", newline="\n")
            instance_counts[split] += len(label_lines)
            manifest_rows.append(
                {
                    "image_id": image_name,
                    "split": split,
                    "instance_count": len(label_lines),
                    "polygon_points": ";".join(
                        str(len(annotation["segmentation"][0]) // 2)
                        for annotation in annotations
                    ),
                }
            )

    output_root.mkdir(parents=True, exist_ok=True)
    yaml_text = "\n".join(
        [
            f"path: {output_root.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: fracture",
            "",
        ]
    )
    (output_root / "data.yaml").write_text(yaml_text, encoding="utf-8", newline="\n")
    summary = {
        "dataset": "FracAtlas v7 official positive-only split",
        "task": "instance_segmentation",
        "annotation_source": "COCO_fracture_masks.json",
        "link_mode": "hardlink",
        "image_counts": {split: len(image_ids) for split, image_ids in split_ids.items()},
        "instance_counts": dict(instance_counts),
        "unique_images": len(set(all_ids)),
        "split_overlap": 0,
        "polygon_points": {
            "minimum": min(polygon_point_counts),
            "maximum": max(polygon_point_counts),
            "mean": sum(polygon_point_counts) / len(polygon_point_counts),
        },
        "source_split_sha256": normalized_text_sha256(
            split_root / name for name in OFFICIAL_SPLIT_FILES.values()
        ),
        "source_coco_sha256": hashlib.sha256(coco_path.read_bytes()).hexdigest(),
    }
    (output_root / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    manifest_paths = [output_root / "official_split_manifest.csv"]
    if portable_manifest_path is not None:
        portable_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_paths.append(portable_manifest_path)
    for manifest_path in manifest_paths:
        with manifest_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
            writer.writeheader()
            writer.writerows(manifest_rows)
    return summary
