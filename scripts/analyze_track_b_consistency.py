from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze global-to-local Track B consistency")
    parser.add_argument(
        "--classification",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts/evaluations"
            / "track_b_densenet121_letterbox_auprc_15ep_test_calibrated_b8"
            / "test_predictions.csv"
        ),
    )
    parser.add_argument(
        "--detection",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts/evaluations"
            / "track_b_detect_yolov8s_finetune_adamw_15ep_test_final"
            / "test_per_image.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts/consistency/track_b_densenet_yolov8s",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def confusion(labels: list[int], predictions: list[int]) -> dict[str, float | int]:
    pairs = list(zip(labels, predictions, strict=True))
    tp = sum(label == 1 and prediction == 1 for label, prediction in pairs)
    tn = sum(label == 0 and prediction == 0 for label, prediction in pairs)
    fp = sum(label == 0 and prediction == 1 for label, prediction in pairs)
    fn = sum(label == 1 and prediction == 0 for label, prediction in pairs)
    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    return {
        "n": len(labels),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / max(1, len(labels)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2,
    }


def main() -> int:
    args = parse_args()
    classification = {row["image_id"]: row for row in read_csv(args.classification)}
    detection = {row["image_id"]: row for row in read_csv(args.detection)}
    if set(classification) != set(detection):
        raise ValueError("Classification and detection image memberships differ")
    joined = []
    for image_id in sorted(classification):
        classifier = classification[image_id]
        detector = detection[image_id]
        label = int(classifier["label"])
        classifier_positive = int(classifier["prediction"])
        detector_positive = int(int(detector["predictions"]) > 0)
        joined.append(
            {
                "image_id": image_id,
                "label": label,
                "classifier_probability": float(classifier["probability"]),
                "classifier_positive": classifier_positive,
                "detector_positive": detector_positive,
                "agreement": int(classifier_positive == detector_positive),
                "or_prediction": int(classifier_positive or detector_positive),
                "and_prediction": int(classifier_positive and detector_positive),
                "ground_truth_instances": int(detector["ground_truth_instances"]),
                "detector_predictions": int(detector["predictions"]),
                "detector_tp": int(detector["tp"]),
                "detector_fp": int(detector["fp"]),
                "detector_fn": int(detector["fn"]),
            }
        )

    labels = [int(row["label"]) for row in joined]
    classifier_predictions = [int(row["classifier_positive"]) for row in joined]
    detector_predictions = [int(row["detector_positive"]) for row in joined]
    agreements = [row for row in joined if row["agreement"]]
    disagreement_count = len(joined) - len(agreements)
    consensus = confusion(
        [int(row["label"]) for row in agreements],
        [int(row["classifier_positive"]) for row in agreements],
    )

    gated_tp = sum(
        int(row["detector_tp"]) for row in joined if row["classifier_positive"] == 1
    )
    gated_fp = sum(
        int(row["detector_fp"]) for row in joined if row["classifier_positive"] == 1
    )
    total_gt = sum(int(row["ground_truth_instances"]) for row in joined)
    gated_fn = total_gt - gated_tp
    report = {
        "images": len(joined),
        "global_to_local_agreement": len(agreements) / len(joined),
        "disagreements": disagreement_count,
        "global_positive_without_local_evidence": sum(
            row["classifier_positive"] == 1 and row["detector_positive"] == 0
            for row in joined
        ),
        "global_negative_with_local_evidence": sum(
            row["classifier_positive"] == 0 and row["detector_positive"] == 1
            for row in joined
        ),
        "classifier": confusion(labels, classifier_predictions),
        "detector_image_level": confusion(labels, detector_predictions),
        "or_fusion": confusion(labels, [int(row["or_prediction"]) for row in joined]),
        "and_fusion": confusion(labels, [int(row["and_prediction"]) for row in joined]),
        "consensus_only": {
            "coverage": len(agreements) / len(joined),
            "abstention_rate": disagreement_count / len(joined),
            **consensus,
        },
        "classifier_gated_detection": {
            "tp": gated_tp,
            "fp": gated_fp,
            "fn": gated_fn,
            "precision": gated_tp / max(1, gated_tp + gated_fp),
            "lesion_sensitivity": gated_tp / max(1, gated_tp + gated_fn),
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "joined_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(joined[0]))
        writer.writeheader()
        writer.writerows(joined)
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
