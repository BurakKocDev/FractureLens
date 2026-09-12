# FractureLens

FractureLens is a research and decision-support prototype for jointly evaluating
fracture classification, localization, instance segmentation, calibration, and
selective prediction on musculoskeletal radiographs.

The project is not a medical device and must not be used for diagnosis or
treatment decisions.

## Current status

**Phase 3 — frozen Track A/Track B benchmarks and fused local prototype complete.**
The pinned FracAtlas v7 archive has been downloaded, verified, audited, and
converted into a duplicate-aware frozen split. The closest Track A detection
reproduction reaches test mAP50 0.492 and mAP50-95 0.208. The YOLOv8s-seg test
mask scores are mAP50 0.407 and mAP50-95 0.127. On Track B, the mobile-first
MobileNetV3-Small classifier reaches test AUROC 0.895 and AUPRC 0.723 on 799
fractured and non-fractured images.

The final Track B DenseNet121 classifier reaches test AUROC 0.912 and AUPRC
0.781. The negative-aware YOLOv8s detector reaches test mAP50 0.479 and
mAP50-95 0.195, with a 0.76% false-alarm rate on negative images at the
validation-selected operating point. OR fusion raises image-level sensitivity
to 86.5%; agreement-only selective prediction covers 83.1% of images at 96.5%
accuracy. A local FastAPI service and browser interface now run the frozen
DenseNet121 and YOLOv8s models together, show model disagreements, annotate
candidate regions, and export a visual PNG report.

Important dataset facts:

- Figshare article: `22363012`
- Pinned release: `10.6084/m9.figshare.22363012.v7`
- Pinned file: `FracAtlas.zip` (`file_id=65518038`)
- Expected size: `338436460` bytes
- Expected MD5: `fe9da2c7c285915ebee69dfdab8fd396`
- License: CC BY 4.0
- The v7 record describes 4,073 images, while the 2023 paper describes 4,083.
  Counts must therefore come from the pinned archive rather than the paper.

## Phase 0 commands

Use Python 3.11 or later.

```powershell
python scripts/download_fracatlas.py
python scripts/inspect_fracatlas.py
python scripts/render_duplicate_review.py
python scripts/build_cleaning_manifest.py
python scripts/create_split_manifest.py
python scripts/smoke_test_dataset.py
python scripts/render_mask_qa.py
python -m unittest discover -s tests -v
```

The download command refuses to extract an archive whose size or MD5 differs
from the pinned metadata. Raw data and generated artifacts are excluded from
Git.

## Planned experiment tracks

1. Official positive-only YOLOv8s/YOLOv8s-seg reproduction.
2. Duplicate-aware full-dataset benchmark using both positive and negative
   images on a single frozen test manifest.
3. FractureLens-MTL: Mask R-CNN R50-FPN with an additional global fracture
   classification head.

The detailed study protocol lives in `docs/study_protocol.md`; verified data
findings and decisions are in `docs/data_audit_report.md` and
`docs/dataset_card.md`.

## Track A detection

The training code stays in the repository; only source images, linked training
trees, downloaded weights, and run artifacts stay outside Git.

```powershell
./scripts/setup_training.ps1
python scripts/prepare_track_a_yolo.py
python scripts/train_track_a.py --task detect --epochs 1 --batch 2 --name track_a_detect_smoke
python scripts/train_track_a.py --task detect --epochs 30 --batch 2 --name track_a_detect_30ep
python scripts/evaluate_track_a.py --split test --name track_a_detect_30ep_test
python scripts/train_track_a.py --task detect --profile fidelity-sgd --epochs 30 --batch 16 --name track_a_detect_fidelity_sgd_b16_30ep
python scripts/evaluate_track_a.py --weights artifacts/runs/track_a_detect_fidelity_sgd_b16_30ep/weights/best.pt --split test --name track_a_detect_fidelity_sgd_b16_30ep_test
python scripts/analyze_track_a_errors.py
python scripts/prepare_track_a_segmentation.py
python scripts/train_track_a.py --task segment --profile fidelity-sgd --epochs 30 --batch 8 --name track_a_segment_fidelity_sgd_b8_30ep
python scripts/evaluate_track_a.py --task segment --weights artifacts/runs/track_a_segment_fidelity_sgd_b8_30ep/weights/best.pt --split test --name track_a_segment_fidelity_sgd_b8_30ep_test
```

The trainer refuses to fall back silently to CPU, records environment and Git
provenance, and refuses to overwrite an existing named run.

The official notebook output scans 608 training images, but the current v7
split contains 574. Exact reproduction is therefore not claimed; see
`docs/track_a_experiment_log.md` for the audited comparison and limitations.
Fixed-threshold per-image and subgroup findings are recorded in
`docs/track_a_error_analysis.md`.

## Track B classification

Track B uses the frozen duplicate-aware split and never resplits inside the
trainer. The positive class receives a training-only loss weight. The decision
threshold is selected on validation, stored in the checkpoint, and then applied
unchanged to test.

```powershell
python scripts/train_track_b_classifier.py --epochs 1 --batch 32 --patience 1 --name track_b_mobilenet_v3_small_smoke_1ep
python scripts/train_track_b_classifier.py --epochs 15 --batch 32 --patience 5 --name track_b_mobilenet_v3_small_15ep
python scripts/evaluate_track_b_classifier.py --weights artifacts/runs/track_b_mobilenet_v3_small_15ep/best.pt --batch 64 --bootstrap 1000 --name track_b_mobilenet_v3_small_15ep_test_corrected
python scripts/analyze_track_b_errors.py
```

See `docs/track_b_experiment_log.md` for the complete configuration, bootstrap
confidence intervals, subgroup results, threshold audit note, and next ablation.
Detection and global-to-local consistency results are recorded in
`docs/track_b_detection_log.md`.

## Fused inference prototype

Run both frozen models and return calibrated classification, fracture boxes,
OR/AND decisions, and a disagreement review flag:

```powershell
python scripts/predict_fracture.py path/to/radiograph.jpg --output-json result.json --overlay overlay.jpg
```

Install the optional service dependencies and start the local API:

```powershell
pip install -e ".[service]"
python scripts/run_api.py --host 127.0.0.1 --port 8000
curl.exe -X POST http://127.0.0.1:8000/v1/predict -H "Content-Type: image/jpeg" --data-binary "@radiograph.jpg"
```

Open `http://127.0.0.1:8000/` for the local browser interface. The API accepts
raw or multipart JPEG, PNG, and WebP requests up to 25 MiB. Localization boxes
receive validation-derived low/medium/high display tiers; these describe model
confidence rather than disease severity or clinical certainty. The application
is a research prototype and must not be used for diagnosis or treatment
decisions.

## Source and attribution

FracAtlas is distributed under CC BY 4.0. Cite:

> Abedeen I, Rahman MA, Prottyasha FZ, et al. FracAtlas: A Dataset for
> Fracture Classification, Localization and Segmentation of Musculoskeletal
> Radiographs. Scientific Data. 2023;10:521.
> https://doi.org/10.1038/s41597-023-02432-4

Dataset record: https://doi.org/10.6084/m9.figshare.22363012.v7
