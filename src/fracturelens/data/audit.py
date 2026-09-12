from __future__ import annotations

import csv
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFile, ImageFont, ImageOps


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    path: str
    fractured: int
    fracture_count: int
    anatomy: str
    view: str
    hardware: int
    multiscan: int
    raw_width: int
    raw_height: int
    width: int
    height: int
    mode: str
    exif_orientation: int
    decode_status: str
    size_bytes: int
    file_md5: str
    pixel_md5: str
    phash: str
    dhash: str


def file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_bits_to_hex(bits: np.ndarray) -> str:
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bit)
    width = math.ceil(bits.size / 4)
    return f"{value:0{width}x}"


def _dct_matrix(size: int) -> np.ndarray:
    positions = np.arange(size, dtype=np.float64) + 0.5
    frequencies = np.arange(size, dtype=np.float64)[:, None]
    matrix = np.cos((math.pi / size) * frequencies * positions)
    matrix[0] *= math.sqrt(1 / size)
    matrix[1:] *= math.sqrt(2 / size)
    return matrix


_DCT_32 = _dct_matrix(32)


def perceptual_hash(image: Image.Image) -> str:
    pixels = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
    transformed = _DCT_32 @ pixels @ _DCT_32.T
    low = transformed[:8, :8]
    threshold = np.median(low.reshape(-1)[1:])
    return _hash_bits_to_hex(low > threshold)


def difference_hash(image: Image.Image) -> str:
    pixels = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    return _hash_bits_to_hex(pixels[:, 1:] > pixels[:, :-1])


def hamming_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def global_ssim(left: Path, right: Path, size: int = 256) -> float:
    with load_display_image(left) as left_image, load_display_image(right) as right_image:
        x = np.asarray(
            ImageOps.fit(left_image.convert("L"), (size, size), method=Image.Resampling.BILINEAR),
            dtype=np.float64,
        )
        y = np.asarray(
            ImageOps.fit(right_image.convert("L"), (size, size), method=Image.Resampling.BILINEAR),
            dtype=np.float64,
        )

    mean_x = x.mean()
    mean_y = y.mean()
    variance_x = x.var()
    variance_y = y.var()
    covariance = ((x - mean_x) * (y - mean_y)).mean()
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    numerator = (2 * mean_x * mean_y + c1) * (2 * covariance + c2)
    denominator = (mean_x**2 + mean_y**2 + c1) * (variance_x + variance_y + c2)
    return float(numerator / denominator) if denominator else 0.0


def load_display_image(path: Path) -> Image.Image:
    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        with Image.open(path) as source:
            source.load()
            return ImageOps.exif_transpose(source).copy()
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous


def locate_dataset_root(search_root: Path) -> Path:
    candidates = [path.parent for path in search_root.rglob("dataset.csv")]
    valid = [
        path
        for path in candidates
        if (path / "images").is_dir() and (path / "Annotations").is_dir()
    ]
    if len(valid) != 1:
        raise ValueError(f"Expected one FracAtlas root, found {len(valid)} under {search_root}")
    return valid[0]


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def dominant_label(row: dict[str, str], columns: tuple[str, ...]) -> str:
    active = [column for column in columns if int(row[column]) == 1]
    return "+".join(active) if active else "unknown"


def canonical_image_path(dataset_root: Path, row: dict[str, str]) -> Path:
    folder = "Fractured" if int(row["fractured"]) == 1 else "Non_fractured"
    return dataset_root / "images" / folder / row["image_id"]


def inventory_images(
    dataset_root: Path,
    rows: list[dict[str, str]],
) -> tuple[list[ImageRecord], list[str], list[str]]:
    records: list[ImageRecord] = []
    errors: list[str] = []
    warnings: list[str] = []

    for row in rows:
        path = canonical_image_path(dataset_root, row)
        if not path.is_file():
            errors.append(f"missing canonical image: {row['image_id']} -> {path}")
            continue

        decode_status = "strict"
        strict_error = ""
        try:
            with Image.open(path) as source:
                raw_width, raw_height = source.size
                mode = source.mode
                exif_orientation = int(source.getexif().get(274, 1))
                source.load()
                image = ImageOps.exif_transpose(source).copy()
        except OSError as exc:
            strict_error = str(exc)
            decode_status = "truncated_recovered"
            try:
                image = load_display_image(path)
                with Image.open(path) as metadata_source:
                    raw_width, raw_height = metadata_source.size
                    mode = metadata_source.mode
                    exif_orientation = int(metadata_source.getexif().get(274, 1))
                warnings.append(f"{row['image_id']}: {strict_error}")
            except Exception as recovery_exc:  # noqa: BLE001 - audit captures corrupt files
                errors.append(
                    f"cannot decode {row['image_id']}: strict={strict_error}; "
                    f"recovery={type(recovery_exc).__name__}: {recovery_exc}"
                )
                continue
        except Exception as exc:  # noqa: BLE001 - audit must capture all corrupt-file errors
            errors.append(f"cannot decode {row['image_id']}: {type(exc).__name__}: {exc}")
            continue

        grayscale = np.asarray(image.convert("L"))
        record = ImageRecord(
            image_id=row["image_id"],
            path=str(path),
            fractured=int(row["fractured"]),
            fracture_count=int(row["fracture_count"]),
            anatomy=dominant_label(row, ("hand", "leg", "hip", "shoulder", "mixed")),
            view=dominant_label(row, ("frontal", "lateral", "oblique")),
            hardware=int(row["hardware"]),
            multiscan=int(row["multiscan"]),
            raw_width=raw_width,
            raw_height=raw_height,
            width=image.width,
            height=image.height,
            mode=mode,
            exif_orientation=exif_orientation,
            decode_status=decode_status,
            size_bytes=path.stat().st_size,
            file_md5=file_md5(path),
            pixel_md5=hashlib.md5(grayscale.tobytes(), usedforsecurity=False).hexdigest(),
            phash=perceptual_hash(image),
            dhash=difference_hash(image),
        )
        records.append(record)

    return records, errors, warnings


def duplicate_groups(records: list[ImageRecord], attribute: str) -> list[dict[str, Any]]:
    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        groups[getattr(record, attribute)].append(record)

    output = []
    for digest, members in groups.items():
        if len(members) > 1:
            output.append(
                {
                    "digest_type": attribute,
                    "digest": digest,
                    "count": len(members),
                    "redundant_count": len(members) - 1,
                    "image_ids": [member.image_id for member in members],
                    "paths": [member.path for member in members],
                    "fractured_labels": sorted({member.fractured for member in members}),
                    "fracture_counts": sorted({member.fracture_count for member in members}),
                    "anatomies": sorted({member.anatomy for member in members}),
                    "views": sorted({member.view for member in members}),
                    "label_conflict": len({member.fractured for member in members}) > 1,
                }
            )
    return sorted(output, key=lambda item: (-item["count"], item["digest"]))


def physical_duplicate_groups(dataset_root: Path) -> list[dict[str, Any]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted((dataset_root / "images").rglob("*.jpg")):
        groups[file_md5(path)].append(path)
    return [
        {
            "digest": digest,
            "count": len(paths),
            "redundant_count": len(paths) - 1,
            "paths": [str(path) for path in paths],
        }
        for digest, paths in groups.items()
        if len(paths) > 1
    ]


def near_duplicate_candidates(
    records: list[ImageRecord],
    phash_threshold: int = 4,
    dhash_threshold: int = 2,
    ssim_threshold: float = 0.92,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    by_anatomy: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_anatomy[record.anatomy].append(record)

    for anatomy, members in by_anatomy.items():
        for index, left in enumerate(members):
            for right in members[index + 1 :]:
                if left.file_md5 == right.file_md5 or left.pixel_md5 == right.pixel_md5:
                    continue
                p_distance = hamming_distance(left.phash, right.phash)
                d_distance = hamming_distance(left.dhash, right.dhash)
                if p_distance <= phash_threshold or d_distance <= dhash_threshold:
                    similarity = global_ssim(Path(left.path), Path(right.path))
                    if similarity < ssim_threshold and not (p_distance <= 2 and d_distance <= 2):
                        continue
                    candidates.append(
                        {
                            "anatomy": anatomy,
                            "left_image_id": left.image_id,
                            "right_image_id": right.image_id,
                            "left_path": left.path,
                            "right_path": right.path,
                            "left_fractured": left.fractured,
                            "right_fractured": right.fractured,
                            "left_fracture_count": left.fracture_count,
                            "right_fracture_count": right.fracture_count,
                            "label_conflict": left.fractured != right.fractured,
                            "phash_distance": p_distance,
                            "dhash_distance": d_distance,
                            "global_ssim": round(similarity, 6),
                        }
                    )

    return sorted(
        candidates,
        key=lambda item: (
            item["phash_distance"],
            item["dhash_distance"],
            -item["global_ssim"],
        ),
    )


def load_coco(dataset_root: Path) -> dict[str, Any]:
    paths = list((dataset_root / "Annotations" / "COCO JSON").glob("*.json"))
    if len(paths) != 1:
        raise ValueError(f"Expected one COCO JSON, found {len(paths)}")
    with paths[0].open("r", encoding="utf-8") as stream:
        return json.load(stream)


def validate_coco(
    coco: dict[str, Any],
    records: list[ImageRecord],
) -> dict[str, Any]:
    record_by_name = {record.image_id: record for record in records}
    image_by_id = {int(image["id"]): image for image in coco["images"]}
    positive_names = {record.image_id for record in records if record.fractured == 1}
    coco_names = {image["file_name"] for image in coco["images"]}
    errors: list[str] = []

    for image in coco["images"]:
        name = image["file_name"]
        record = record_by_name.get(name)
        if record is None:
            errors.append(f"COCO image missing from canonical records: {name}")
            continue
        if (int(image["width"]), int(image["height"])) != (record.width, record.height):
            errors.append(
                f"COCO dimension mismatch {name}: coco={image['width']}x{image['height']} "
                f"actual={record.width}x{record.height}"
            )

    invalid_bbox = 0
    invalid_segmentation = 0
    annotations_per_image: Counter[int] = Counter()
    for annotation in coco["annotations"]:
        image = image_by_id.get(int(annotation["image_id"]))
        if image is None:
            errors.append(f"annotation references missing image id: {annotation['id']}")
            continue
        annotations_per_image[int(annotation["image_id"])] += 1
        x, y, width, height = map(float, annotation["bbox"])
        if (
            width <= 0
            or height <= 0
            or x < 0
            or y < 0
            or x + width > float(image["width"]) + 1e-3
            or y + height > float(image["height"]) + 1e-3
        ):
            invalid_bbox += 1

        segmentations = annotation.get("segmentation", [])
        if not segmentations:
            invalid_segmentation += 1
        for polygon in segmentations:
            if len(polygon) < 6 or len(polygon) % 2:
                invalid_segmentation += 1
                continue
            xs = polygon[0::2]
            ys = polygon[1::2]
            if (
                min(xs) < 0
                or min(ys) < 0
                or max(xs) > float(image["width"]) + 1e-3
                or max(ys) > float(image["height"]) + 1e-3
            ):
                invalid_segmentation += 1

    return {
        "images": len(coco["images"]),
        "annotations": len(coco["annotations"]),
        "categories": coco.get("categories", []),
        "positive_images_missing_from_coco": sorted(positive_names - coco_names),
        "coco_images_not_positive": sorted(coco_names - positive_names),
        "images_without_annotations": sorted(
            image["file_name"]
            for image_id, image in image_by_id.items()
            if annotations_per_image[image_id] == 0
        ),
        "invalid_bbox_count": invalid_bbox,
        "invalid_segmentation_count": invalid_segmentation,
        "errors": errors,
    }


def validate_yolo(dataset_root: Path, records: list[ImageRecord]) -> dict[str, Any]:
    root = dataset_root / "Annotations" / "YOLO"
    missing: list[str] = []
    invalid_lines: list[str] = []
    line_counts: dict[str, int] = {}

    for record in records:
        path = root / f"{Path(record.image_id).stem}.txt"
        if not path.is_file():
            missing.append(record.image_id)
            continue
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        line_counts[record.image_id] = len(lines)
        for line_number, line in enumerate(lines, start=1):
            parts = line.split()
            try:
                class_id, x_center, y_center, width, height = map(float, parts)
            except (ValueError, TypeError):
                invalid_lines.append(f"{record.image_id}:{line_number}: malformed")
                continue
            if len(parts) != 5 or class_id != 0 or not all(
                0 <= value <= 1 for value in (x_center, y_center, width, height)
            ) or width <= 0 or height <= 0:
                invalid_lines.append(f"{record.image_id}:{line_number}: out of range")

    fracture_count_mismatches = sorted(
        record.image_id
        for record in records
        if line_counts.get(record.image_id, -1) != record.fracture_count
    )
    return {
        "label_files": len(list(root.glob("*.txt"))) - int((root / "classes.txt").exists()),
        "missing_label_files": missing,
        "invalid_lines": invalid_lines,
        "fracture_count_mismatches": fracture_count_mismatches,
    }


def validate_voc(dataset_root: Path, records: list[ImageRecord]) -> dict[str, Any]:
    root = dataset_root / "Annotations" / "PASCAL VOC"
    missing: list[str] = []
    parse_errors: list[str] = []
    dimension_mismatches: list[str] = []
    object_count_mismatches: list[str] = []

    for record in records:
        path = root / f"{Path(record.image_id).stem}.xml"
        if not path.is_file():
            missing.append(record.image_id)
            continue
        try:
            tree = ET.parse(path)
            root_node = tree.getroot()
            width = int(root_node.findtext("size/width", "-1"))
            height = int(root_node.findtext("size/height", "-1"))
            if (width, height) != (record.width, record.height):
                dimension_mismatches.append(record.image_id)
            objects = root_node.findall("object")
            if len(objects) != record.fracture_count:
                object_count_mismatches.append(record.image_id)
        except (ET.ParseError, ValueError) as exc:
            parse_errors.append(f"{record.image_id}: {exc}")

    return {
        "xml_files": len(list(root.glob("*.xml"))),
        "missing_xml_files": missing,
        "parse_errors": parse_errors,
        "dimension_mismatches": dimension_mismatches,
        "object_count_mismatches": object_count_mismatches,
    }


def validate_official_splits(dataset_root: Path, records: list[ImageRecord]) -> dict[str, Any]:
    split_root = dataset_root / "Utilities" / "Fracture Split"
    positive_names = {record.image_id for record in records if record.fractured == 1}
    split_members: dict[str, list[str]] = {}
    all_members: list[str] = []

    for split in ("train", "valid", "test"):
        rows = load_csv_rows(split_root / f"{split}.csv")
        members = [row["image_id"] for row in rows]
        split_members[split] = members
        all_members.extend(members)

    counts = Counter(all_members)
    return {
        "counts": {split: len(members) for split, members in split_members.items()},
        "duplicates_across_or_within_splits": sorted(name for name, count in counts.items() if count > 1),
        "positive_images_missing_from_splits": sorted(positive_names - set(all_members)),
        "split_images_not_positive": sorted(set(all_members) - positive_names),
    }


def _best_bbox_alignment_error(
    reference: list[tuple[float, float, float, float]],
    candidate: list[tuple[float, float, float, float]],
) -> float:
    if len(reference) != len(candidate):
        return float("inf")
    if not reference:
        return 0.0
    best = float("inf")
    for ordering in permutations(candidate):
        error = max(
            abs(reference_box[index] - candidate_box[index])
            for reference_box, candidate_box in zip(reference, ordering, strict=True)
            for index in range(4)
        )
        best = min(best, error)
    return best


def validate_cross_format_boxes(
    dataset_root: Path,
    records: list[ImageRecord],
    coco: dict[str, Any],
    tolerance_pixels: float = 2.0,
) -> dict[str, Any]:
    record_by_name = {record.image_id: record for record in records}
    coco_image_by_id = {int(item["id"]): item for item in coco["images"]}
    coco_boxes: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        name = coco_image_by_id[int(annotation["image_id"])]["file_name"]
        x, y, width, height = map(float, annotation["bbox"])
        coco_boxes[name].append((x, y, x + width, y + height))

    yolo_root = dataset_root / "Annotations" / "YOLO"
    voc_root = dataset_root / "Annotations" / "PASCAL VOC"
    yolo_errors: dict[str, float] = {}
    voc_errors: dict[str, float] = {}

    for name, reference_boxes in coco_boxes.items():
        record = record_by_name[name]
        yolo_path = yolo_root / f"{Path(name).stem}.txt"
        yolo_boxes = []
        for line in yolo_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            _, x_center, y_center, width, height = map(float, line.split())
            x_center *= record.width
            y_center *= record.height
            width *= record.width
            height *= record.height
            yolo_boxes.append(
                (
                    x_center - width / 2,
                    y_center - height / 2,
                    x_center + width / 2,
                    y_center + height / 2,
                )
            )
        yolo_errors[name] = _best_bbox_alignment_error(reference_boxes, yolo_boxes)

        tree = ET.parse(voc_root / f"{Path(name).stem}.xml")
        voc_boxes = [
            (
                float(node.findtext("bndbox/xmin", "nan")),
                float(node.findtext("bndbox/ymin", "nan")),
                float(node.findtext("bndbox/xmax", "nan")),
                float(node.findtext("bndbox/ymax", "nan")),
            )
            for node in tree.getroot().findall("object")
        ]
        voc_errors[name] = _best_bbox_alignment_error(reference_boxes, voc_boxes)

    return {
        "positive_images_checked": len(coco_boxes),
        "tolerance_pixels": tolerance_pixels,
        "max_coco_yolo_error_pixels": max(yolo_errors.values(), default=0.0),
        "max_coco_voc_error_pixels": max(voc_errors.values(), default=0.0),
        "coco_yolo_over_tolerance": sorted(
            name for name, error in yolo_errors.items() if error > tolerance_pixels
        ),
        "coco_voc_over_tolerance": sorted(
            name for name, error in voc_errors.items() if error > tolerance_pixels
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def select_qa_records(records: list[ImageRecord], count: int = 50) -> list[ImageRecord]:
    ordered = sorted(records, key=lambda record: record.image_id)
    buckets: dict[tuple[Any, ...], list[ImageRecord]] = defaultdict(list)
    for record in ordered:
        key = (record.fractured, record.anatomy, record.view, record.hardware, record.multiscan)
        buckets[key].append(record)

    selected: list[ImageRecord] = []
    keys = sorted(buckets, key=str)
    while len(selected) < count and keys:
        remaining_keys = []
        for key in keys:
            if buckets[key] and len(selected) < count:
                selected.append(buckets[key].pop(len(buckets[key]) // 2))
            if buckets[key]:
                remaining_keys.append(key)
        keys = remaining_keys
    return selected


def make_contact_sheet(records: list[ImageRecord], output: Path, columns: int = 5) -> None:
    tile_width, tile_height = 300, 300
    rows = math.ceil(len(records) / columns)
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for index, record in enumerate(records):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        with load_display_image(Path(record.path)) as source:
            image = ImageOps.contain(source.convert("RGB"), (tile_width - 12, tile_height - 54))
        image_x = x + (tile_width - image.width) // 2
        image_y = y + 4
        sheet.paste(image, (image_x, image_y))
        label = (
            f"{record.image_id} F={record.fractured} n={record.fracture_count}\n"
            f"{record.anatomy} | {record.view} | HW={record.hardware} M={record.multiscan}"
        )
        draw.multiline_text((x + 5, y + tile_height - 46), label, fill="black", font=font, spacing=2)
        draw.rectangle((x, y, x + tile_width - 1, y + tile_height - 1), outline="#a0a0a0")

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def make_pair_contact_sheet(
    candidates: list[dict[str, Any]],
    output: Path,
    pair_limit: int = 30,
) -> None:
    chosen = candidates[:pair_limit]
    tile_width, tile_height = 320, 260
    sheet = Image.new("RGB", (tile_width * 2, tile_height * max(len(chosen), 1)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for row_index, candidate in enumerate(chosen):
        for column_index, side in enumerate(("left", "right")):
            path = Path(candidate[f"{side}_path"])
            with load_display_image(path) as source:
                image = ImageOps.contain(source.convert("RGB"), (tile_width - 12, tile_height - 46))
            x = column_index * tile_width + (tile_width - image.width) // 2
            y = row_index * tile_height + 4
            sheet.paste(image, (x, y))
            fracture_label = candidate.get(f"{side}_fractured", "?")
            label = f"{candidate[f'{side}_image_id']} F={fracture_label}"
            draw.text((column_index * tile_width + 5, y + image.height + 4), label, fill="black", font=font)
        metrics = (
            f"p={candidate['phash_distance']} d={candidate['dhash_distance']} "
            f"ssim={candidate['global_ssim']:.3f}"
        )
        draw.text((5, (row_index + 1) * tile_height - 16), metrics, fill="#9b1c1c", font=font)

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def run_audit(search_root: Path, output_root: Path) -> dict[str, Any]:
    dataset_root = locate_dataset_root(search_root)
    rows = load_csv_rows(dataset_root / "dataset.csv")
    records, decode_errors, decode_warnings = inventory_images(dataset_root, rows)
    record_by_name = {record.image_id: record for record in records}

    physical_images = list((dataset_root / "images").rglob("*.jpg"))
    physical_name_counts = Counter(path.name for path in physical_images)
    duplicate_physical_names = {
        name: count for name, count in physical_name_counts.items() if count > 1
    }

    file_duplicates = duplicate_groups(records, "file_md5")
    pixel_duplicates = duplicate_groups(records, "pixel_md5")
    near_candidates = near_duplicate_candidates(records)
    coco = load_coco(dataset_root)
    coco_report = validate_coco(coco, records)
    yolo_report = validate_yolo(dataset_root, records)
    voc_report = validate_voc(dataset_root, records)
    split_report = validate_official_splits(dataset_root, records)

    csv_ids = [row["image_id"] for row in rows]
    summary = {
        "dataset_root": str(dataset_root),
        "csv_rows": len(rows),
        "csv_unique_image_ids": len(set(csv_ids)),
        "canonical_images": len(records),
        "physical_jpg_files": len(physical_images),
        "duplicate_physical_names": duplicate_physical_names,
        "fractured": sum(record.fractured for record in records),
        "non_fractured": sum(1 - record.fractured for record in records),
        "fracture_instances_from_csv": sum(record.fracture_count for record in records),
        "anatomy_counts": dict(Counter(record.anatomy for record in records)),
        "view_counts": dict(Counter(record.view for record in records)),
        "hardware": sum(record.hardware for record in records),
        "multiscan": sum(record.multiscan for record in records),
        "decode_errors": decode_errors,
        "decode_warnings": decode_warnings,
        "truncated_recovered": sum(
            record.decode_status == "truncated_recovered" for record in records
        ),
        "exif_oriented_images": sum(record.exif_orientation not in (0, 1) for record in records),
        "exact_file_duplicate_groups": len(file_duplicates),
        "exact_file_redundant_images": sum(group["redundant_count"] for group in file_duplicates),
        "exact_pixel_duplicate_groups": len(pixel_duplicates),
        "exact_pixel_redundant_images": sum(group["redundant_count"] for group in pixel_duplicates),
        "exact_file_label_conflict_groups": sum(group["label_conflict"] for group in file_duplicates),
        "exact_pixel_label_conflict_groups": sum(group["label_conflict"] for group in pixel_duplicates),
        "near_duplicate_candidates": len(near_candidates),
        "near_duplicate_label_conflict_pairs": sum(
            candidate["label_conflict"] for candidate in near_candidates
        ),
        "coco": coco_report,
        "yolo": yolo_report,
        "voc": voc_report,
        "official_splits": split_report,
        "cross_format_boxes": validate_cross_format_boxes(dataset_root, records, coco),
    }

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_csv(output_root / "image_inventory.csv", [asdict(record) for record in records])
    write_csv(output_root / "exact_file_duplicates.csv", file_duplicates)
    write_csv(output_root / "exact_pixel_duplicates.csv", pixel_duplicates)
    write_csv(output_root / "physical_duplicate_groups.csv", physical_duplicate_groups(dataset_root))
    write_csv(output_root / "near_duplicate_candidates.csv", near_candidates)

    qa_records = select_qa_records(records)
    write_csv(output_root / "qa_manifest.csv", [asdict(record) for record in qa_records])
    make_contact_sheet(qa_records, output_root / "qa_contact_sheet.jpg")
    if near_candidates:
        make_pair_contact_sheet(near_candidates, output_root / "near_duplicate_contact_sheet.jpg")

    unresolved = [
        record.image_id
        for record in records
        if record.image_id not in record_by_name  # defensive; always false unless inventory changes
    ]
    if unresolved:
        summary["unresolved_inventory_records"] = unresolved
    return summary
