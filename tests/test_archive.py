from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fracturelens.data.archive import md5_file, safe_extract_zip, verify_archive  # noqa: E402
from fracturelens.data.audit import hamming_distance  # noqa: E402


class ArchiveTests(unittest.TestCase):
    def test_md5_and_size_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "sample.bin"
            archive.write_bytes(b"fracturelens")
            expected = hashlib.md5(b"fracturelens", usedforsecurity=False).hexdigest()

            verify_archive(archive, len(b"fracturelens"), expected)
            self.assertEqual(md5_file(archive), expected)

    def test_size_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "sample.bin"
            archive.write_bytes(b"fracturelens")

            with self.assertRaisesRegex(ValueError, "size mismatch"):
                verify_archive(archive, 1, "unused")

    def test_zip_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("../escape.txt", "nope")

            with self.assertRaisesRegex(ValueError, "Unsafe archive member"):
                safe_extract_zip(archive, root / "output")

    def test_hamming_distance(self) -> None:
        self.assertEqual(hamming_distance("0f", "0f"), 0)
        self.assertEqual(hamming_distance("00", "ff"), 8)


if __name__ == "__main__":
    unittest.main()
