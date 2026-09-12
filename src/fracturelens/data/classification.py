from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from fracturelens.data.audit import load_display_image


class ManifestClassificationDataset:
    """Manifest-backed binary classifier dataset with audited image decoding."""

    def __init__(
        self,
        dataset_root: Path,
        manifest_path: Path,
        split: str,
        transform: Callable[[Image.Image], object] | None = None,
    ) -> None:
        with manifest_path.open("r", encoding="utf-8", newline="") as stream:
            self.rows = [row for row in csv.DictReader(stream) if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No manifest rows found for split={split!r}")
        self.dataset_root = dataset_root
        self.split = split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        path = self.dataset_root / Path(row["source_relative_path"])
        image = load_display_image(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image_id": row["image_id"],
            "image": image,
            "label": int(row["fractured"]),
            "anatomy": row["anatomy"],
            "view": row["view"],
            "hardware": int(row["hardware"]),
        }

    @property
    def labels(self) -> list[int]:
        return [int(row["fractured"]) for row in self.rows]
