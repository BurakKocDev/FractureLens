from __future__ import annotations

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
)


def choose_examples(rows: list[dict[str, str]], count: int = 20) -> list[dict[str, str]]:
    positives = [row for row in rows if int(row["fractured"]) == 1]
    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in positives:
        anatomy = next(
            (name for name in ("hand", "leg", "hip", "shoulder", "mixed") if int(row[name])),
            "unknown",
        )
        buckets[anatomy].append(row)

    selected: list[dict[str, str]] = []
    names = sorted(buckets)
    offset = 0
    while len(selected) < count and names:
        name = names[offset % len(names)]
        bucket = buckets[name]
        position = offset // len(names)
        if position < len(bucket):
            selected.append(bucket[position])
        offset += 1
        if offset > count * len(names) * 2:
            break
    return selected


def main() -> int:
    dataset_root = locate_dataset_root(PROJECT_ROOT / "data" / "raw" / "fracatlas-v7")
    rows = load_csv_rows(dataset_root / "dataset.csv")
    coco = load_coco(dataset_root)
    image_names = {int(item["id"]): item["file_name"] for item in coco["images"]}
    annotations: dict[str, list[dict]] = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations[image_names[int(annotation["image_id"])]].append(annotation)

    tile_width, tile_height = 420, 360
    columns, rows_count = 4, 5
    sheet = Image.new("RGB", (columns * tile_width, rows_count * tile_height), "white")
    sheet_draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    examples = choose_examples(rows, columns * rows_count)
    for index, row in enumerate(examples):
        image_id = row["image_id"]
        image = load_display_image(canonical_image_path(dataset_root, row)).convert("RGB")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")
        for annotation in annotations[image_id]:
            for polygon in annotation.get("segmentation", []):
                points = list(zip(polygon[0::2], polygon[1::2], strict=True))
                draw.polygon(points, fill=(255, 0, 0, 90), outline=(255, 64, 64, 255), width=4)
            x, y, width, height = annotation["bbox"]
            draw.rectangle((x, y, x + width, y + height), outline=(255, 255, 0, 255), width=4)
        composited = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
        preview = ImageOps.contain(
            composited, (tile_width - 12, tile_height - 44), Image.Resampling.LANCZOS
        )
        column, row_index = index % columns, index // columns
        x = column * tile_width + (tile_width - preview.width) // 2
        y = row_index * tile_height + 4
        sheet.paste(preview, (x, y))
        anatomy = next(
            (name for name in ("hand", "leg", "hip", "shoulder", "mixed") if int(row[name])),
            "unknown",
        )
        label = f"{image_id} | {anatomy} | instances={row['fracture_count']}"
        sheet_draw.text(
            (column * tile_width + 6, (row_index + 1) * tile_height - 34),
            label,
            fill="black",
            font=font,
        )
        sheet_draw.rectangle(
            (
                column * tile_width,
                row_index * tile_height,
                (column + 1) * tile_width - 1,
                (row_index + 1) * tile_height - 1,
            ),
            outline="#999999",
        )

    output = PROJECT_ROOT / "artifacts" / "data_audit" / "mask_overlay_qa.jpg"
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=94)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
