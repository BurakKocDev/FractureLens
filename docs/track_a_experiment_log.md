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
