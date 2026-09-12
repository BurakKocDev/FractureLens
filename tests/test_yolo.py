from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fracturelens.data.yolo import coco_polygon_to_yolo, read_image_ids, validate_yolo_label


class YoloPreparationTests(unittest.TestCase):
    def test_valid_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "label.txt"
            path.write_text("0 0.5 0.4 0.2 0.1\n", encoding="utf-8")
            self.assertEqual(validate_yolo_label(path), 1)

    def test_invalid_box_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "label.txt"
            path.write_text("0 0.5 0.4 0.0 0.1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Non-positive"):
                validate_yolo_label(path)

    def test_duplicate_split_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.csv"
            path.write_text("image_id\na.jpg\na.jpg\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_image_ids(path)

    def test_coco_polygon_is_normalized_for_yolo_segmentation(self) -> None:
        label = coco_polygon_to_yolo([0, 0, 100, 0, 100, 50], width=200, height=100)
        self.assertEqual(
            label,
            "0 0.00000000 0.00000000 0.50000000 0.00000000 0.50000000 0.50000000",
        )

    def test_invalid_coco_polygon_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "three xy points"):
            coco_polygon_to_yolo([0, 0, 1, 1], width=10, height=10)

    def test_coco_polygon_clamps_floating_point_boundary_noise(self) -> None:
        label = coco_polygon_to_yolo(
            [0, 0, 100.00000000000001, 0, 100, 50], width=100, height=100
        )
        self.assertIn("1.00000000", label)


if __name__ == "__main__":
    unittest.main()
