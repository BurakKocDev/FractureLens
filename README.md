# FractureLens

FractureLens is a research and decision-support prototype for jointly evaluating
fracture classification, localization, instance segmentation, calibration, and
selective prediction on musculoskeletal radiographs.

The project is not a medical device and must not be used for diagnosis or
treatment decisions.

## Current status

**Phase 1 — Track A detection baseline complete.** The pinned FracAtlas v7
archive has been downloaded, verified, audited, and converted into a
duplicate-aware frozen split. Two 30-epoch YOLOv8s detection runs and explicit
held-out test evaluation are complete. The closest official-configuration run
uses batch 16 and SGD; its test mAP50 is 0.492 and mAP50-95 is 0.208.

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
```

The trainer refuses to fall back silently to CPU, records environment and Git
provenance, and refuses to overwrite an existing named run.

The official notebook output scans 608 training images, but the current v7
split contains 574. Exact reproduction is therefore not claimed; see
`docs/track_a_experiment_log.md` for the audited comparison and limitations.

## Source and attribution

FracAtlas is distributed under CC BY 4.0. Cite:

> Abedeen I, Rahman MA, Prottyasha FZ, et al. FracAtlas: A Dataset for
> Fracture Classification, Localization and Segmentation of Musculoskeletal
> Radiographs. Scientific Data. 2023;10:521.
> https://doi.org/10.1038/s41597-023-02432-4

Dataset record: https://doi.org/10.6084/m9.figshare.22363012.v7
