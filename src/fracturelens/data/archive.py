from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

CHUNK_SIZE = 1024 * 1024


def md5_file(path: Path, chunk_size: int = CHUNK_SIZE) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def verify_archive(path: Path, expected_size: int, expected_md5: str) -> None:
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"Archive size mismatch: expected {expected_size}, got {actual_size}")

    actual_md5 = md5_file(path)
    if actual_md5.lower() != expected_md5.lower():
        raise ValueError(f"Archive MD5 mismatch: expected {expected_md5}, got {actual_md5}")


def download_file(
    url: str,
    destination: Path,
    expected_size: int,
    progress: Callable[[int, int], None] | None = None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    request = urllib.request.Request(url, headers={"User-Agent": "FractureLens/0.0.1"})
    downloaded = 0
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
        while chunk := response.read(CHUNK_SIZE):
            output.write(chunk)
            downloaded += len(chunk)
            if progress:
                progress(downloaded, expected_size)

    partial.replace(destination)


def safe_extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()

    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if destination_root != target and destination_root not in target.parents:
                raise ValueError(f"Unsafe archive member: {member.filename}")

        for member in bundle.infolist():
            target = destination / member.filename
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)

