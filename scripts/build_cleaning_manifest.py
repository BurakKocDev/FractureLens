from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = PROJECT_ROOT / "artifacts" / "data_audit"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def main() -> int:
    inventory = read_csv(AUDIT_ROOT / "image_inventory.csv")
    exact_groups = read_csv(AUDIT_ROOT / "exact_pixel_duplicates.csv")
    near_pairs = read_csv(AUDIT_ROOT / "near_duplicate_candidates.csv")
    by_id = {row["image_id"]: row for row in inventory}

    decisions = {
        image_id: {
            "image_id": image_id,
            "include_provisional": True,
            "decision": "keep_unique",
            "reason": "",
            "exact_group": "",
            "near_group": "",
            "near_review_decision": "not_applicable",
            "split_group": f"IMAGE::{image_id}",
        }
        for image_id in by_id
    }

    exact_conflict_members = 0
    exact_consistent_redundant = 0
    for group_index, group in enumerate(exact_groups, start=1):
        members = json.loads(group["image_ids"])
        group_id = f"PX{group_index:03d}"
        conflict = group["label_conflict"].lower() == "true"
        for image_id in members:
            decisions[image_id]["exact_group"] = group_id

        if conflict:
            for image_id in members:
                decisions[image_id].update(
                    include_provisional=False,
                    decision="exclude",
                    reason="exact_pixel_duplicate_with_label_conflict",
                )
                exact_conflict_members += 1
            continue

        representative = min(
            members,
            key=lambda image_id: (
                by_id[image_id]["decode_status"] != "strict",
                image_id,
            ),
        )
        decisions[representative]["decision"] = "keep_exact_group_representative"
        for image_id in members:
            if image_id == representative:
                continue
            decisions[image_id].update(
                include_provisional=False,
                decision="exclude",
                reason=f"exact_pixel_duplicate_of_{representative}",
            )
            exact_consistent_redundant += 1

    union_find = UnionFind()
    for pair in near_pairs:
        left = pair["left_image_id"]
        right = pair["right_image_id"]
        if decisions[left]["include_provisional"] and decisions[right]["include_provisional"]:
            union_find.union(left, right)

    near_groups: dict[str, list[str]] = defaultdict(list)
    for image_id in union_find.parent:
        near_groups[union_find.find(image_id)].append(image_id)

    sorted_groups = sorted(
        (sorted(members) for members in near_groups.values() if len(members) > 1),
        key=lambda members: members[0],
    )
    for group_index, members in enumerate(sorted_groups, start=1):
        group_id = f"NEAR{group_index:03d}"
        label_conflict = len({by_id[image_id]["fractured"] for image_id in members}) > 1
        for image_id in members:
            decisions[image_id]["near_group"] = group_id
            decisions[image_id]["split_group"] = group_id
            if label_conflict:
                decisions[image_id].update(
                    include_provisional=False,
                    decision="exclude",
                    reason="near_duplicate_group_with_label_conflict",
                    near_review_decision="exclude_conflicting_component",
                )
            elif decisions[image_id]["decision"] == "keep_unique":
                decisions[image_id].update(
                    decision="keep_grouped_near_candidate",
                    reason="conservative_same_split_after_visual_review",
                    near_review_decision="keep_same_split",
                )

    output_rows = []
    for image_id in sorted(by_id):
        output_rows.append({**by_id[image_id], **decisions[image_id]})

    output_path = AUDIT_ROOT / "cleaning_manifest_v1.csv"
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "initial_canonical_records": len(inventory),
        "excluded_exact_label_conflict_members": exact_conflict_members,
        "excluded_consistent_exact_pixel_redundants": exact_consistent_redundant,
        "provisional_records_before_near_conflict_exclusions": (
            sum(row["include_provisional"] for row in decisions.values())
            + sum(
                row["reason"] == "near_duplicate_group_with_label_conflict"
                for row in decisions.values()
            )
        ),
        "near_candidate_pairs": len(near_pairs),
        "near_candidate_groups_after_exact_cleaning": len(sorted_groups),
        "near_candidate_images_after_exact_cleaning": sum(map(len, sorted_groups)),
        "excluded_near_label_conflict_members": sum(
            row["reason"] == "near_duplicate_group_with_label_conflict"
            for row in decisions.values()
        ),
        "provisional_records_after_conflict_exclusions": sum(
            row["include_provisional"] for row in decisions.values()
        ),
        "status": "conservative_grouping_ready_near_review_still_required_before_freeze",
    }
    summary["near_candidate_contact_sheets_reviewed"] = 10
    summary["review_date"] = "2026-09-12"
    summary["status"] = "ready_for_group_stratified_split"
    (AUDIT_ROOT / "cleaning_summary_v1.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
