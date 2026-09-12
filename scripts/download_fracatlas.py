from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fracturelens.data.archive import (  # noqa: E402
    download_file,
    load_config,
    md5_file,
    safe_extract_zip,
    verify_archive,
)


def progress(downloaded: int, total: int) -> None:
    percent = min(downloaded / total * 100, 100) if total else 0
    print(f"\rDownloading: {downloaded:,}/{total:,} bytes ({percent:5.1f}%)", end="", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and verify pinned FracAtlas v7")
    parser.add_argument("--force", action="store_true", help="Replace an existing invalid archive")
    parser.add_argument("--no-extract", action="store_true", help="Verify but do not extract")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = PROJECT_ROOT / "configs" / "data" / "fracatlas_v7.json"
    config = load_config(config_path)
    file_config = config["file"]

    raw_root = PROJECT_ROOT / "data" / "raw"
    archive_path = raw_root / file_config["name"]
    extract_root = raw_root / "fracatlas-v7"

    if archive_path.exists():
        try:
            verify_archive(archive_path, file_config["size_bytes"], file_config["md5"])
            print(f"Using verified archive: {archive_path}")
        except ValueError:
            if not args.force:
                raise
            archive_path.unlink()

    if not archive_path.exists():
        download_file(
            file_config["download_url"],
            archive_path,
            file_config["size_bytes"],
            progress=progress,
        )
        print()
        verify_archive(archive_path, file_config["size_bytes"], file_config["md5"])

    receipt = {
        "downloaded_at": datetime.now(UTC).isoformat(),
        "article_id": config["article_id"],
        "version": config["version"],
        "doi": config["doi"],
        "file_id": file_config["id"],
        "archive": str(archive_path.relative_to(PROJECT_ROOT)),
        "size_bytes": archive_path.stat().st_size,
        "md5": md5_file(archive_path),
        "source_url": file_config["download_url"],
    }
    receipt_path = raw_root / "fracatlas_v7_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"Verified receipt: {receipt_path}")

    if not args.no_extract:
        if extract_root.exists() and any(extract_root.iterdir()):
            print(f"Extraction directory already populated: {extract_root}")
        else:
            safe_extract_zip(archive_path, extract_root)
            print(f"Extracted dataset: {extract_root}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

