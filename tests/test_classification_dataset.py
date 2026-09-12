from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from fracturelens.data.classification import ManifestClassificationDataset


def test_manifest_classification_dataset(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    image_path = dataset_root / "images" / "Fractured" / "sample.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("L", (8, 6), 128).save(image_path)
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "image_id",
                "split",
                "source_relative_path",
                "fractured",
                "anatomy",
                "view",
                "hardware",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_id": "sample.jpg",
                "split": "train",
                "source_relative_path": "images/Fractured/sample.jpg",
                "fractured": 1,
                "anatomy": "hand",
                "view": "frontal",
                "hardware": 0,
            }
        )

    dataset = ManifestClassificationDataset(dataset_root, manifest_path, "train")
    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["image"].mode == "RGB"
    assert sample["label"] == 1
    assert dataset.labels == [1]
