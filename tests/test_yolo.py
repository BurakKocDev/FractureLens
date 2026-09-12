from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fracturelens.data.yolo import read_image_ids, validate_yolo_label


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


if __name__ == "__main__":
    unittest.main()
