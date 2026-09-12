from __future__ import annotations

import unittest

import numpy as np

from fracturelens.data.dataset import ManifestSample, validate_batch


class DatasetTests(unittest.TestCase):
    def test_valid_negative_batch(self) -> None:
        sample = ManifestSample(
            image_id="negative.jpg",
            split="train",
            image=np.zeros((3, 8, 10), dtype=np.uint8),
            fractured=0,
            boxes_xywh=np.zeros((0, 4), dtype=np.float32),
            masks=np.zeros((0, 8, 10), dtype=np.uint8),
            metadata={},
        )
        validate_batch([sample])

    def test_positive_without_annotation_is_rejected(self) -> None:
        sample = ManifestSample(
            image_id="positive.jpg",
            split="train",
            image=np.zeros((3, 8, 10), dtype=np.uint8),
            fractured=1,
            boxes_xywh=np.zeros((0, 4), dtype=np.float32),
            masks=np.zeros((0, 8, 10), dtype=np.uint8),
            metadata={},
        )
        with self.assertRaisesRegex(ValueError, "lacks a local annotation"):
            validate_batch([sample])


if __name__ == "__main__":
    unittest.main()
