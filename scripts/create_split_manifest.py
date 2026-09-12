from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = PROJECT_ROOT / "artifacts" / "data_audit"
MANIFEST_ROOT = PROJECT_ROOT / "manifests"
SEED = 20260912
RATIOS = {"train": 0.70, "validation": 0.10, "test": 0.20}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_groups(rows: list[dict[str, str]]) -> dict[str, str]:
    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_group[row["split_group"]].append(row)

    by_stratum: dict[tuple[str, str], list[tuple[str, list[dict[str, str]]]]] = defaultdict(list)
    for group_id, members in by_group.items():
        fracture_labels = {member["fractured"] for member in members}
        if len(fracture_labels) != 1:
            raise ValueError(f"Mixed fracture labels remain in split group {group_id}")
        anatomies = Counter(member["anatomy"] for member in members)
        anatomy = anatomies.most_common(1)[0][0]
        by_stratum[(next(iter(fracture_labels)), anatomy)].append((group_id, members))

    assignments: dict[str, str] = {}
    for stratum_index, (stratum, groups) in enumerate(sorted(by_stratum.items())):
        rng = random.Random(SEED + stratum_index)
        rng.shuffle(groups)
        total = sum(len(members) for _, members in groups)
        targets = {split: total * ratio for split, ratio in RATIOS.items()}
        counts = Counter()

        for group_id, members in groups:
            deficits = {
                split: targets[split] - counts[split]
                for split in ("train", "validation", "test")
            }
            split = max(deficits, key=lambda name: (deficits[name], RATIOS[name], name))
            assignments[group_id] = split
            counts[split] += len(members)

    return assignments


def main() -> int:
    cleaning_rows = read_csv(AUDIT_ROOT / "cleaning_manifest_v1.csv")
    included = [row for row in cleaning_rows if row["include_provisional"] == "True"]
    excluded = [row for row in cleaning_rows if row["include_provisional"] != "True"]
    assignments = assign_groups(included)

    output_rows = []
    for row in sorted(included, key=lambda item: item["image_id"]):
        source_folder = "Fractured" if row["fractured"] == "1" else "Non_fractured"
        output_rows.append(
            {
                "image_id": row["image_id"],
                "split": assignments[row["split_group"]],
                "split_group": row["split_group"],
                "source_relative_path": f"images/{source_folder}/{row['image_id']}",
                "fractured": row["fractured"],
                "fracture_count": row["fracture_count"],
                "anatomy": row["anatomy"],
                "view": row["view"],
                "hardware": row["hardware"],
                "multiscan": row["multiscan"],
                "width": row["width"],
                "height": row["height"],
                "decode_status": row["decode_status"],
                "exif_orientation": row["exif_orientation"],
                "file_md5": row["file_md5"],
                "pixel_md5": row["pixel_md5"],
            }
        )

    manifest_path = MANIFEST_ROOT / "fracatlas_v7_clean_split_seed20260912.csv"
    write_csv(manifest_path, output_rows)
    write_csv(
        MANIFEST_ROOT / "fracatlas_v7_exclusions.csv",
        [
            {
                "image_id": row["image_id"],
                "reason": row["reason"],
                "exact_group": row["exact_group"],
                "near_group": row["near_group"],
                "fractured": row["fractured"],
                "fracture_count": row["fracture_count"],
                "pixel_md5": row["pixel_md5"],
            }
            for row in sorted(excluded, key=lambda item: item["image_id"])
        ],
    )

    group_splits: dict[str, set[str]] = defaultdict(set)
    pixel_splits: dict[str, set[str]] = defaultdict(set)
    for row in output_rows:
        group_splits[row["split_group"]].add(row["split"])
        pixel_splits[row["pixel_md5"]].add(row["split"])
    if any(len(splits) > 1 for splits in group_splits.values()):
        raise AssertionError("A duplicate-aware split group crossed partitions")
    if any(len(splits) > 1 for splits in pixel_splits.values()):
        raise AssertionError("An exact pixel duplicate crossed partitions")

    split_counts = Counter(row["split"] for row in output_rows)
    positive_counts = Counter(row["split"] for row in output_rows if row["fractured"] == "1")
    summary = {
        "seed": SEED,
        "ratios": RATIOS,
        "source_records": len(cleaning_rows),
        "included_records": len(output_rows),
        "excluded_records": len(excluded),
        "split_counts": dict(split_counts),
        "positive_counts": dict(positive_counts),
        "negative_counts": {
            split: split_counts[split] - positive_counts[split] for split in split_counts
        },
        "unique_split_groups": len(group_splits),
        "cross_split_group_violations": 0,
        "cross_split_exact_pixel_duplicate_violations": 0,
        "manifest_sha256": sha256_file(manifest_path),
        "limitations": [
            "FracAtlas has no patient identifiers; this is not a patient-level split.",
            "Near-duplicate candidates are conservatively co-located, but repeated patients may remain.",
            "Label-conflicting exact and near-duplicate components are excluded from model development.",
        ],
    }
    summary_path = MANIFEST_ROOT / "fracatlas_v7_clean_split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

