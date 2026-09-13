# FractureLens Model Card

## Model details

FractureLens Track B fused bundle `v1.1` combines two independently trained
models for musculoskeletal radiographs:

| Component | Architecture | Input | Frozen operating point |
| --- | --- | --- | --- |
| Global classifier | DenseNet121 | RGB 224×224 letterbox | Temperature 1.1858; threshold 0.3595 |
| Local detector | YOLOv8s | RGB, 640 px inference | Confidence 0.2435; NMS IoU 0.50 |

The classifier weight SHA-256 is
`394bd5296d456abebafcdb12e04c3018477f05bce78a5745206588d6aef94bf9`.
The detector weight SHA-256 is
`761b616a8d5dc453b7416a6e19ec09291cb320c6886269ddbd0010499de31ce5`.
The complete executable configuration is frozen in
`configs/inference/track_b_fused.json`.

## Intended use

The bundle is intended for retrospective machine-learning research,
educational demonstration, portfolio review, and analysis of agreement between
global fracture classification and local bounding-box evidence.

It is **not a medical device**. It must not be used for diagnosis, treatment,
triage of real patients, or autonomous clinical decision-making. Every output
is a model result rather than medical advice.

## Training and evaluation data

The source is FracAtlas v7, pinned to DOI
`10.6084/m9.figshare.22363012.v7` under CC BY 4.0. A direct audit found 4,083
CSV records. Duplicate handling and label-conflict exclusions produced 3,978
images in the frozen development manifest:

| Split | Images | Positive | Negative |
| --- | ---: | ---: | ---: |
| Train | 2,784 | 485 | 2,299 |
| Validation | 395 | 68 | 327 |
| Test | 799 | 141 | 658 |

Exact-pixel and conservatively grouped near-duplicate components do not cross
splits. Patient-level separation is impossible because patient identifiers are
not supplied.

## Training procedure

The DenseNet121 classifier used ImageNet-1K initialization, weighted binary
cross-entropy, AdamW at learning rate `3e-4`, batch size 8, cosine scheduling,
and 15 epochs. Epoch 13 was selected by validation AUPRC. Temperature scaling
and the Youden-J threshold were fitted on validation only. Training took 669
seconds on an NVIDIA GeForce RTX 3050 Ti Laptop GPU.

The YOLOv8s detector was initialized from the positive-only Track A detector,
trained for 30 epochs on the full negative-aware split, and fine-tuned for 15
epochs with AdamW at learning rate `1e-4`. Epoch 14 of the fine-tune was frozen
before test evaluation. The confidence threshold maximizes instance F1 on
validation. NMS IoU 0.50 was selected in a separate validation-only overlap
audit for the local presentation bundle.

## Frozen test results

### DenseNet121 classifier

| Metric | Estimate | Bootstrap 95% CI |
| --- | ---: | ---: |
| AUROC | 0.912 | 0.883–0.939 |
| AUPRC | 0.781 | 0.718–0.838 |
| Sensitivity | 0.773 | 0.697–0.843 |
| Specificity | 0.881 | 0.857–0.904 |
| Balanced accuracy | 0.827 | 0.787–0.864 |
| ECE, 10 bins | 0.057 | 0.043–0.079 |
| Brier score | 0.084 | 0.071–0.099 |

The fixed confusion matrix is TP=109, TN=580, FP=78, and FN=32.

### YOLOv8s detector

Canonical Ultralytics metrics are precision 0.573, recall 0.464, mAP50 0.479,
and mAP50-95 0.195. At the validation-selected confidence threshold and IoU
0.50 matching rule, lesion precision is 0.590 and lesion sensitivity is 0.449.
Five of 658 negative images contain a false alarm (0.76%).

The `v1.1` presentation NMS removes duplicate candidates. At NMS IoU 0.50,
fixed-threshold test precision is 0.738, lesion sensitivity is 0.432, and F1 is
0.545. These post-processed values do not replace the canonical mAP report.

### Fusion

| Policy | Sensitivity | Specificity | Balanced accuracy |
| --- | ---: | ---: | ---: |
| DenseNet classifier | 0.773 | 0.881 | 0.827 |
| Detector image-level evidence | 0.532 | 0.992 | 0.762 |
| OR fusion | 0.865 | 0.880 | 0.873 |
| AND fusion | 0.440 | 0.994 | 0.717 |

The models agree on 83.1% of test images. Agreement-only reporting covers
83.1% at 96.5% accuracy, but this retrospective result does not establish
clinical safety. The interface exposes disagreement instead of suppressing one
model with the other.

## Output interpretation

- `fracture_probability` is a calibrated model score, not an individual
  patient's probability of disease.
- Bounding boxes are candidate regions, not confirmed lesions.
- Low/medium/high box tiers are derived from cumulative validation precision
  thresholds and do not encode fracture severity or clinical certainty.
- An absent box does not rule out a fracture; fixed-threshold lesion sensitivity
  is limited.
- Model agreement does not remove the need for qualified human review.

## Limitations and risks

- Evaluation uses a single JPG dataset without external validation.
- Patient IDs, demographics, acquisition metadata, and original DICOM depth are
  unavailable.
- Repeated patients may cross splits despite duplicate grouping.
- Performance varies by anatomy and view; very small subgroups cannot support
  stable conclusions.
- Borders, text, multi-view panels, and fixation hardware may create shortcuts.
- The detector misses more than half of annotated instances at the fixed
  operating point.
- Distribution shift, image quality failures, pediatric/adult imbalance, and
  demographic performance are not quantified.

## Reproducibility and reporting

The dataset card, manifest hash, experiment logs, overlap audit, fused error
gallery, and source attribution are committed under `docs/`. Model binaries and
raw data are intentionally excluded from Git; the inference configuration
contains their expected paths and SHA-256 digests.
