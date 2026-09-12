# Track A Experiment Log

## Controlled environment

- Python: 3.12.14
- PyTorch: 2.7.1+cu126
- torchvision: 0.22.1+cu126
- Ultralytics: 8.4.149
- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4 GB
- Pretrained `yolov8s.pt` SHA-256:
  `1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a`

The official notebook requests 600-pixel input. Ultralytics enforces a
stride-32-compatible size and therefore trains at an effective 608 pixels. Both
the requested and effective values are retained in run provenance.

## Detection smoke run

- Run: `track_a_detect_smoke`
- Source commit: `090d81154c1e54ae819250cbc5b9acd412d27cf9`
- Epochs: 1
- Batch: 2
- Train/validation images: 574/82
- Train/validation instances: 764/91
- Corrupt or missing samples reported by Ultralytics: 0
- Peak trainer-reported GPU memory: approximately 0.54 GB
- Training plus validation completed and produced `best.pt` and `last.pt`.

The one-epoch metrics are not model-quality results. This run exists only to
verify CUDA execution, parsing, augmentation, finite loss, validation, plots,
and checkpoint creation before the 30-epoch reproduction.

## Modern 30-epoch baseline

- Run: `track_a_detect_30ep`
- Epochs/batch/optimizer: 30 / 2 / AdamW selected by Ultralytics `auto`
- Validation (82 images, 91 instances): precision 0.547, recall 0.385,
  mAP50 0.438, mAP50-95 0.179
- Held-out test (63 images, 69 instances): precision 0.560, recall 0.377,
  mAP50 0.375, mAP50-95 0.145
- Training time: 814.5 seconds
- Peak CUDA allocation: 1262.2 MiB
- Best checkpoint SHA-256:
  `311d4aaef390b1419315cbde6899cbc40ebfbdde739fcf795589ccc3fc5fa8d2`

The held-out test result is our measurement; the official notebook reports
validation only, so the two rows must not be compared as if they used the same
split.

## Reproduction audit of the official notebook

Cached output in `notebooks/Train_8s.ipynb` records Python 3.8.12, PyTorch
1.10.2, Ultralytics 8.0.49, an RTX 3070 Laptop GPU with 8 GB VRAM, batch 16,
SGD, learning rate 0.01, momentum 0.937, weight decay 0.0005, seed 0, and 8
workers. It reports validation precision 0.807, recall 0.473, mAP50 0.562, and
mAP50-95 0.276.

The same output scans 608 training images and 82 validation images. The current
FracAtlas v7 `train.csv` contains 574 images. We will not fabricate the missing
historical 34-image membership. Consequently, the next local run is a
configuration-fidelity experiment, not a claim of exact reproduction: it uses
the current v7 split and explicit SGD settings. One-epoch smoke tests passed at
both batch 8 and the notebook's batch 16 setting; batch 16 is therefore used
for the controlled 30-epoch comparison despite the local GPU having half the
notebook GPU's VRAM.

## SGD fidelity run

- Run: `track_a_detect_fidelity_sgd_b16_30ep`
- Epochs/batch/optimizer: 30 / 16 / SGD
- Explicit optimizer settings: lr0 0.01, lrf 0.01, momentum 0.937,
  weight decay 0.0005
- Best epoch: 27 by mAP50-95
- Best-epoch CSV metrics: precision 0.579, recall 0.495, mAP50 0.470,
  mAP50-95 0.215
- Held-out test (63 images, 69 instances): precision 0.607, recall 0.470,
  mAP50 0.492, mAP50-95 0.208
- Training time: 367.6 seconds
- Peak CUDA allocation: 3054.2 MiB
- Best checkpoint SHA-256:
  `f50328dea3b64ef249416f2b507fa05a8f62e2a4f7a994261bb28f7bac54daae`

The run used training script SHA-256
`65d26d553635c08f2b989461624146c4dccc21ff03d42d2854b05a0ef08b2a56`
and experiment configuration SHA-256
`c80f505b99e1dde285d7436e9d8f6a6de3fe975ad0831e44b5db6a000e5b8ec4`.
It ran from Git commit `40279a8203838e8e61ce7dc88a1f39747ede59b1` with
uncommitted experiment-profile changes, which are captured by the two hashes
above and committed immediately after evaluation.

Compared with the modern batch-2/AdamW baseline, the SGD fidelity run improves
validation mAP50-95 from 0.179 to 0.215 and test mAP50-95 from 0.145 to 0.208.
It also improves test mAP50 from 0.375 to 0.492. This supports the conclusion
that optimizer and batch configuration explain a material part of the gap.
The remaining official-validation gap cannot be attributed cleanly while the
historical 608-image training membership and legacy Ultralytics behavior remain
different.

## Segmentation data gate

The COCO polygon source contains 719 positive images and 924 fracture
instances. Conversion to YOLO segmentation format preserves the official v7
574/82/63 image split and 764/91/69 instance counts. Every annotation contains
one polygon with 3 to 17 points (mean 6.17). Split overlap is zero. A source
coordinate equal to `1.0000000000000002` after normalization was accepted as
floating-point boundary noise and clamped to 1.0 with a `1e-9` tolerance.

## SGD segmentation run

- Run: `track_a_segment_fidelity_sgd_b8_30ep`
- Source commit: `6d2b22868dd5b3a10688ec48586f73f6f385424b`
- Epochs/batch/optimizer: 30 / 8 / SGD
- Requested/effective image size: 600 / 608
- Training/validation images: 574 / 82
- Training/validation instances: 764 / 91
- Best joint checkpoint: epoch 25 by Ultralytics fitness
- Validation box metrics: precision 0.646, recall 0.481, mAP50 0.530,
  mAP50-95 0.258
- Validation mask metrics: precision 0.763, recall 0.424, mAP50 0.509,
  mAP50-95 0.192
- Held-out test box metrics: precision 0.720, recall 0.420, mAP50 0.437,
  mAP50-95 0.193
- Held-out test mask metrics: precision 0.670, recall 0.391, mAP50 0.407,
  mAP50-95 0.127
- Training time: 552.4 seconds
- Peak CUDA allocation: 2015.4 MiB
- Best checkpoint SHA-256:
  `5abb8c3593ee39c01371e6b6775a89948f82be125a22d9adaf8ecc131a145369`

The segmentation model is precise when it emits a mask, but its test recall is
only 0.391. The validation-to-test mask mAP50 drop from 0.509 to 0.407 also
shows that the 82-image validation split is optimistic for this run. The
dedicated detection checkpoint remains the better localizer on test
(mAP50 0.492 versus the segmentation model's box mAP50 0.437). Segment masks
therefore add explanatory shape, but do not replace the detection baseline.
