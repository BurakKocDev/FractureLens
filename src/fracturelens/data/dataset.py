from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from PIL import Image, ImageDraw

from fracturelens.data.audit import load_display_image


@dataclass(frozen=True)
class ManifestSample:
    image_id: str
    split: str
    image: np.ndarray
    fractured: int
    boxes_xywh: np.ndarray
    masks: np.ndarray
    metadata: dict[str, str]


class FracAtlasManifestDataset:
    """Read the frozen manifest while applying the audited image-decoding rules."""

    def __init__(self, dataset_root: Path, manifest_path: Path, split: str) -> None:
        self.dataset_root = dataset_root
        with manifest_path.open("r", encoding="utf-8", newline="") as stream:
            self.rows = [row for row in csv.DictReader(stream) if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No manifest rows found for split={split!r}")

        coco_paths = list((dataset_root / "Annotations" / "COCO JSON").glob("*.json"))
        if len(coco_paths) != 1:
            raise ValueError(f"Expected one COCO file, found {len(coco_paths)}")
        with coco_paths[0].open("r", encoding="utf-8") as stream:
            coco = json.load(stream)

        names = {int(item["id"]): item["file_name"] for item in coco["images"]}
        self.annotations: dict[str, list[dict]] = defaultdict(list)
        for annotation in coco["annotations"]:
            self.annotations[names[int(annotation["image_id"])]].append(annotation)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> ManifestSample:
        row = self.rows[index]
        path = self.dataset_root / Path(row["source_relative_path"])
        image = load_display_image(path).convert("RGB")
        width, height = image.size

        annotations = self.annotations.get(row["image_id"], [])
        boxes = np.asarray(
            [annotation["bbox"] for annotation in annotations], dtype=np.float32
        ).reshape(-1, 4)
        masks = []
        for annotation in annotations:
            mask = Image.new("L", (width, height), 0)
            draw = ImageDraw.Draw(mask)
            for polygon in annotation.get("segmentation", []):
                points = list(zip(polygon[0::2], polygon[1::2], strict=True))
                draw.polygon(points, fill=1)
            masks.append(np.asarray(mask, dtype=np.uint8))

        mask_array = (
            np.stack(masks, axis=0)
            if masks
            else np.zeros((0, height, width), dtype=np.uint8)
        )
        image_array = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)
        return ManifestSample(
            image_id=row["image_id"],
            split=row["split"],
            image=image_array,
            fractured=int(row["fractured"]),
            boxes_xywh=boxes,
            masks=mask_array,
            metadata=row,
        )

    def batches(self, batch_size: int) -> Iterator[list[ManifestSample]]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        for start in range(0, len(self), batch_size):
            yield [self[index] for index in range(start, min(start + batch_size, len(self)))]


def validate_batch(samples: Sequence[ManifestSample]) -> None:
    if not samples:
        raise ValueError("Batch is empty")
    for sample in samples:
        _, height, width = sample.image.shape
        if sample.image.shape[0] != 3 or sample.image.dtype != np.uint8:
            raise ValueError(f"Unexpected image tensor for {sample.image_id}")
        if sample.boxes_xywh.shape != (sample.masks.shape[0], 4):
            raise ValueError(f"Box/mask count mismatch for {sample.image_id}")
        if sample.masks.shape[1:] != (height, width):
            raise ValueError(f"Mask/image dimension mismatch for {sample.image_id}")
        if sample.fractured == 0 and (len(sample.boxes_xywh) or len(sample.masks)):
            raise ValueError(f"Negative image has a local annotation: {sample.image_id}")
        if sample.fractured == 1 and not len(sample.boxes_xywh):
            raise ValueError(f"Positive image lacks a local annotation: {sample.image_id}")
