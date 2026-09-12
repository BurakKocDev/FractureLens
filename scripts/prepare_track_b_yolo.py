from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fracturelens.data.audit import locate_dataset_root  # noqa: E402
from fracturelens.data.yolo import prepare_track_b_detection  # noqa: E402


def main() -> int:
    dataset_root = locate_dataset_root(PROJECT_ROOT / "data" / "raw" / "fracatlas-v7")
    output_root = PROJECT_ROOT / "data" / "processed" / "track_b_detect_canonical"
    summary = prepare_track_b_detection(
        dataset_root,
        PROJECT_ROOT / "manifests" / "fracatlas_v7_clean_split_seed20260912.csv",
        output_root,
    )
    print(json.dumps(summary, indent=2))
    print(output_root / "data.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
