from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fracturelens.data.audit import locate_dataset_root  # noqa: E402
from fracturelens.data.dataset import (  # noqa: E402
    FracAtlasManifestDataset,
    validate_batch,
)


def main() -> int:
    dataset_root = locate_dataset_root(PROJECT_ROOT / "data" / "raw" / "fracatlas-v7")
    manifest = PROJECT_ROOT / "manifests" / "fracatlas_v7_clean_split_seed20260912.csv"

    for split in ("train", "validation", "test"):
        dataset = FracAtlasManifestDataset(dataset_root, manifest, split)
        positive_indices = [
            index for index, row in enumerate(dataset.rows) if int(row["fractured"]) == 1
        ]
        negative_indices = [
            index for index, row in enumerate(dataset.rows) if int(row["fractured"]) == 0
        ]
        indices = positive_indices[:4] + negative_indices[:4]
        batch = [dataset[index] for index in indices]
        validate_batch(batch)
        positive_count = sum(sample.fractured for sample in batch)
        print(
            f"{split}: rows={len(dataset)}, batch={len(batch)}, "
            f"positives={positive_count}, first={batch[0].image_id}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
