from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fracturelens.data.audit import (  # noqa: E402
    canonical_image_path,
    load_coco,
    load_csv_rows,
    load_display_image,
    locate_dataset_root,
    make_pair_contact_sheet,
)


def load_conflict_groups(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [row for row in rows if row["label_conflict"].lower() == "true"]


def draw_coco_annotations(image: Image.Image, annotations: list[dict]) -> Image.Image:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output, "RGBA")
    for annotation in annotations:
        for polygon in annotation.get("segmentation", []):
            points = list(zip(polygon[0::2], polygon[1::2], strict=True))
            draw.polygon(points, fill=(255, 0, 0, 70), outline=(255, 0, 0, 255), width=3)
        x, y, width, height = annotation["bbox"]
        draw.rectangle((x, y, x + width, y + height), outline=(255, 255, 0, 255), width=3)
    return output


def render_conflicts(dataset_root: Path, audit_root: Path) -> Path:
    csv_rows = load_csv_rows(dataset_root / "dataset.csv")
    row_by_id = {row["image_id"]: row for row in csv_rows}
    coco = load_coco(dataset_root)
    coco_image_by_id = {int(item["id"]): item for item in coco["images"]}
    annotations_by_name: dict[str, list[dict]] = defaultdict(list)
    for annotation in coco["annotations"]:
        name = coco_image_by_id[int(annotation["image_id"])]["file_name"]
        annotations_by_name[name].append(annotation)

    groups = load_conflict_groups(audit_root / "exact_pixel_duplicates.csv")
    columns = max(len(json.loads(group["image_ids"])) for group in groups)
    tile_width, tile_height = 420, 390
    sheet = Image.new("RGB", (columns * tile_width, len(groups) * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for row_index, group in enumerate(groups):
        for column_index, image_id in enumerate(json.loads(group["image_ids"])):
            metadata = row_by_id[image_id]
            path = canonical_image_path(dataset_root, metadata)
            with load_display_image(path) as image:
                annotated = draw_coco_annotations(image, annotations_by_name.get(image_id, []))
                preview = ImageOps.contain(
                    annotated,
                    (tile_width - 14, tile_height - 72),
                    method=Image.Resampling.LANCZOS,
                )
            x = column_index * tile_width + (tile_width - preview.width) // 2
            y = row_index * tile_height + 5
            sheet.paste(preview, (x, y))
            label = (
                f"{image_id}\nfractured={metadata['fractured']} count={metadata['fracture_count']} "
                f"{metadata['hand']=} {metadata['leg']=} {metadata['hip']=} "
                f"{metadata['shoulder']=} view=({metadata['frontal']},{metadata['lateral']},"
                f"{metadata['oblique']})"
            )
            draw.multiline_text(
                (column_index * tile_width + 5, (row_index + 1) * tile_height - 62),
                label,
                fill="black",
                font=font,
                spacing=2,
            )
            draw.rectangle(
                (
                    column_index * tile_width,
                    row_index * tile_height,
                    (column_index + 1) * tile_width - 1,
                    (row_index + 1) * tile_height - 1,
                ),
                outline="#999999",
            )

    output = audit_root / "conflicting_label_duplicates.jpg"
    sheet.save(output, quality=94)
    return output


def main() -> int:
    data_search_root = PROJECT_ROOT / "data" / "raw" / "fracatlas-v7"
    audit_root = PROJECT_ROOT / "artifacts" / "data_audit"
    dataset_root = locate_dataset_root(data_search_root)
    conflict_path = render_conflicts(dataset_root, audit_root)

    with (audit_root / "near_duplicate_candidates.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        candidates = list(csv.DictReader(stream))
    for candidate in candidates:
        candidate["phash_distance"] = int(candidate["phash_distance"])
        candidate["dhash_distance"] = int(candidate["dhash_distance"])
        candidate["global_ssim"] = float(candidate["global_ssim"])
    near_path = audit_root / "near_duplicate_top10.jpg"
    make_pair_contact_sheet(candidates, near_path, pair_limit=10)
    for start in range(0, len(candidates), 10):
        chunk = candidates[start : start + 10]
        chunk_path = audit_root / f"near_duplicate_review_{start // 10 + 1:02d}.jpg"
        make_pair_contact_sheet(chunk, chunk_path, pair_limit=len(chunk))
    conflict_candidates = [
        candidate for candidate in candidates if candidate["label_conflict"] == "True"
    ]
    near_conflict_path = audit_root / "near_duplicate_label_conflicts.jpg"
    make_pair_contact_sheet(
        conflict_candidates,
        near_conflict_path,
        pair_limit=len(conflict_candidates),
    )

    print(conflict_path)
    print(near_path)
    print(near_conflict_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
