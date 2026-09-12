# Track B Detection and Consistency Log

## Detector

The Track B detector uses all 3,978 cleaned images, including empty YOLO label
files for negative radiographs. Fifty-four audited truncated JPEGs were decoded
with the project recovery path and re-encoded before Ultralytics ingestion. The
final scan reported zero corrupt or excluded images.

YOLOv8s was initialized from the Track A positive-only checkpoint, trained for
30 epochs, and then fine-tuned for 15 epochs with AdamW at learning rate 1e-4.
The fine-tune improved validation mAP50 from 0.4324 to 0.4630 and mAP50-95 from
0.1786 to 0.1998. Epoch 14 was frozen before test access.

## Frozen test

The 799-image test set contains 141 positive and 658 negative images with 176
fracture instances. Standard test results are:

| Metric | Result |
| --- | ---: |
| Precision | 0.573 |
| Recall | 0.464 |
| mAP50 | 0.479 |
| mAP50-95 | 0.195 |

The fixed confidence threshold 0.2435 was selected by maximum instance F1 on
validation only. At IoU 0.5 it gives 79 TP, 55 FP, and 97 FN on test: precision
0.590, lesion sensitivity 0.449, and F1 0.510. Only five of 658 negative images
contain a false alarm, with six boxes total: 0.0091 false positives per negative
image and a 0.76% negative-image false-alarm rate.

For presentation only, retained boxes are assigned validation-derived detector
confidence tiers. Scores from 0.2435 to below 0.3031 are `low`; 0.3031 to below
0.4075 are `medium`; and 0.4075 or greater are `high`. The medium and high
cutoffs are the first validation curve points reaching cumulative lesion
precision of at least 0.70 and 0.80. These labels describe detector confidence,
not disease severity or clinical certainty, and do not change the frozen
operating point.

## Global-to-local consistency

DenseNet121 global classification was joined with YOLOv8s local evidence by
image ID at their independently validation-frozen thresholds.

| Policy | Sensitivity | Specificity | Balanced accuracy |
| --- | ---: | ---: | ---: |
| DenseNet classifier | 0.773 | 0.881 | 0.827 |
| YOLO image-level evidence | 0.532 | 0.992 | 0.762 |
| OR fusion | 0.865 | 0.880 | 0.873 |
| AND fusion | 0.440 | 0.994 | 0.717 |

The models agree on 83.1% of test images. There are 121 global-positive/no-local
cases and 14 global-negative/local-evidence cases. Treating disagreements as
abstentions gives 83.1% coverage and 96.5% accuracy on covered images, with
76.5% sensitivity and 99.3% specificity. These are retrospective technical
results and do not establish clinical safety.

Classifier-gating the detector is rejected: it reduces lesion sensitivity from
44.9% to 37.5% and does not improve lesion precision. The next prototype should
use OR fusion for sensitive triage and expose disagreements for review rather
than suppressing local evidence.

Canonical artifacts:

- `artifacts/evaluations/track_b_detect_yolov8s_finetune_adamw_15ep_test_final/`
- `artifacts/consistency/track_b_densenet_yolov8s/`
