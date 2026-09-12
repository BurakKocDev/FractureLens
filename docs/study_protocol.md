# FractureLens Study Protocol

## Status

- Protocol version: 0.1
- Dataset gate: passed with documented limitations
- Dataset release: FracAtlas v7
- Model training: not started; Track A configuration frozen
- Clinical claim: none; research prototype only

## Primary question

Under one duplicate-aware evaluation protocol, does a shared-encoder model for
classification, detection, and instance segmentation improve global-to-local
consistency and probability calibration relative to independent task-specific
models?

## Evaluation tracks

### Track A — official reproduction

Reproduce the published positive-only YOLOv8s and YOLOv8s-seg technical
validation using the official split files and recorded training arguments.

### Track B — FractureLens benchmark

Use all unique positive and negative images. Exact duplicates are removed and
verified near-duplicate groups remain within one split. The intended split is
approximately 70% training, 10% validation, and 20% frozen test, stratified by
fracture label and dominant anatomy when the audited labels permit it.

The absence of patient identifiers prevents a true patient-level split and must
remain a primary limitation in every report.

## Baselines

- Classification: DenseNet121
- Detection: YOLOv8s and Faster R-CNN R50-FPN
- Segmentation: YOLOv8s-seg and DeepLabV3+ R50
- Main candidate: Mask R-CNN R50-FPN plus global binary classification head

## Primary outcomes

- Classification: AUROC, AUPRC, sensitivity, specificity, balanced accuracy,
  F1, ECE, and Brier score
- Detection: mAP50, mAP50-95, lesion sensitivity, FROC, and false positives per
  negative image
- Segmentation: Dice, IoU, boundary F1, HD95, and false-positive mask burden on
  negative images
- Consistency: global-positive/no-local-evidence and
  global-negative/high-confidence-local-evidence rates
- Selective prediction: risk-coverage curves after validation-only calibration

All primary metrics will receive 95% nonparametric bootstrap confidence
intervals. Thresholds and model selection are validation-only decisions. The
test manifest stays frozen until the model and thresholds are locked.

## Data gate completion criteria

- Pinned archive size and MD5 match Figshare metadata.
- Dataset CSV rows, image files, COCO polygons, YOLO boxes, and VOC boxes are
  cross-referenced.
- Invalid polygons, out-of-bounds boxes, missing files, and empty or corrupt
  images are reported.
- Exact byte and decoded-pixel duplicates are identified.
- Near-duplicate candidates are generated with perceptual hashes and manually
  reviewable contact sheets.
- At least 50 stratified examples cover positive/negative, anatomy, view,
  hardware, and multiscan labels.
- A one-batch data loader smoke test passes before training.
