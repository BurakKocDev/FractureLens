# Track A Detection Error Analysis

## Scope

This report analyzes the best checkpoint from
`track_a_detect_fidelity_sgd_b16_30ep` on the 63-image, 69-instance positive-only
test split. It uses a fixed confidence threshold of 0.25 and one-to-one greedy
matching at IoU 0.50. These fixed-threshold counts explain individual errors;
they are not substitutes for confidence-swept mAP.

## Fixed-threshold result

- True positives: 30
- False positives: 22
- False negatives: 39
- Precision: 0.577
- Recall: 0.435

The thresholded prediction count is 52 for 69 annotated fractures. Missed
fractures are therefore the primary failure mode at this operating point.

## Subgroup signals

| Group | Images | GT | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hand | 33 | 37 | 16 | 13 | 21 | 0.552 | 0.432 |
| Leg | 25 | 26 | 13 | 9 | 13 | 0.591 | 0.500 |
| Hip | 2 | 3 | 1 | 0 | 2 | 1.000 | 0.333 |
| Hardware present | 4 | 5 | 2 | 5 | 3 | 0.286 | 0.400 |
| Hardware absent | 59 | 64 | 28 | 17 | 36 | 0.622 | 0.438 |

Hip, hardware, and mixed-anatomy subsets are too small for strong statistical
claims. Hardware is nevertheless a useful audit target: four images generate
five of the 22 false-positive boxes.

## Visual review

The worst-case montage shows four two-fracture images with both annotations
missed. Other recurring cases include small distal extremity boxes, casts,
metal hardware, and predictions close to but insufficiently overlapping the
annotated box. Green denotes ground truth, blue a matched prediction, and red
an unmatched prediction.

Generated artifacts:

- `artifacts/error_analysis/track_a_detect_fidelity_sgd_b16_30ep_test_errors_by_group/per_image_errors.csv`
- `artifacts/error_analysis/track_a_detect_fidelity_sgd_b16_30ep_test_errors_by_group/group_metrics.csv`
- `artifacts/error_analysis/track_a_detect_fidelity_sgd_b16_30ep_test_errors_by_group/worst_cases.jpg`

## Decision

Do not optimize only against this positive-only split. It cannot measure false
alarms on healthy radiographs. Complete the Track A segmentation baseline next,
then use the duplicate-aware full-dataset Track B split to measure normal-image
specificity, false positives per image, calibration, and selective prediction.
