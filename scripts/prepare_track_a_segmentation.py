from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fracturelens.data.audit import locate_dataset_root  # noqa: E402
from fracturelens.data.yolo import prepare_track_a_segmentation  # noqa: E402


def main() -> int:
    dataset_root = locate_dataset_root(PROJECT_ROOT / "data" / "raw" / "fracatlas-v7")
    output_root = PROJECT_ROOT / "data" / "processed" / "track_a_segment"
    summary = prepare_track_a_segmentation(
        dataset_root,
        output_root,
        PROJECT_ROOT / "manifests" / "fracatlas_v7_official_positive_segment_split.csv",
    )
    print(json.dumps(summary, indent=2))
    print(output_root / "data.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
