from __future__ import annotations

import csv
import hashlib
import json
import unittest
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = PROJECT_ROOT / "manifests"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class ManifestTests(unittest.TestCase):
    def test_frozen_manifest_invariants(self) -> None:
        manifest = MANIFEST_ROOT / "fracatlas_v7_clean_split_seed20260912.csv"
        summary = json.loads(
            (MANIFEST_ROOT / "fracatlas_v7_clean_split_summary.json").read_text(
                encoding="utf-8"
            )
        )
        rows = read_rows(manifest)

        self.assertEqual(len(rows), 3978)
        self.assertEqual({row["split"] for row in rows}, {"train", "validation", "test"})
        self.assertEqual(len({row["image_id"] for row in rows}), len(rows))

        group_splits: dict[str, set[str]] = defaultdict(set)
        pixel_splits: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            group_splits[row["split_group"]].add(row["split"])
            pixel_splits[row["pixel_md5"]].add(row["split"])
        self.assertTrue(all(len(splits) == 1 for splits in group_splits.values()))
        self.assertTrue(all(len(splits) == 1 for splits in pixel_splits.values()))

        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        self.assertEqual(digest, summary["manifest_sha256"])

    def test_exclusions_are_disjoint(self) -> None:
        included = {
            row["image_id"]
            for row in read_rows(MANIFEST_ROOT / "fracatlas_v7_clean_split_seed20260912.csv")
        }
        excluded_rows = read_rows(MANIFEST_ROOT / "fracatlas_v7_exclusions.csv")
        excluded = {row["image_id"] for row in excluded_rows}

        self.assertEqual(len(excluded_rows), 105)
        self.assertFalse(included & excluded)
        self.assertEqual(len(included | excluded), 4083)

