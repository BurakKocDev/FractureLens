from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from fracturelens.data.classification import SquareLetterbox
from fracturelens.evaluation.classification import temperature_scale
from fracturelens.inference.confidence import localization_confidence_level
from fracturelens.inference.decision import fuse_decisions


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FractureLensPipeline:
    """Load the frozen Track B models and produce one fused inference response."""

    def __init__(self, project_root: Path, config_path: Path, device: str | None = None) -> None:
        os.environ.setdefault(
            "YOLO_CONFIG_DIR", str(project_root.resolve() / "artifacts/ultralytics_config")
        )
        import torch
        from torch import nn
        from torchvision.models import densenet121
        from torchvision.transforms import v2
        from ultralytics import YOLO

        self.project_root = project_root.resolve()
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        classifier_config = self.config["classifier"]
        detector_config = self.config["detector"]
        classifier_path = self.project_root / classifier_config["weights"]
        detector_path = self.project_root / detector_config["weights"]
        for path, expected in (
            (classifier_path, classifier_config["weights_sha256"]),
            (detector_path, detector_config["weights_sha256"]),
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
            if file_sha256(path) != expected:
                raise ValueError(f"Model hash mismatch: {path}")

        checkpoint = torch.load(classifier_path, map_location="cpu", weights_only=False)
        if checkpoint["model_name"] != "densenet121":
            raise ValueError("Fused v1 expects a DenseNet121 classifier")
        classifier = densenet121(weights=None)
        classifier.classifier = nn.Linear(classifier.classifier.in_features, 1)
        classifier.load_state_dict(checkpoint["model_state_dict"])
        classifier.to(self.device).eval()
        self.classifier = classifier
        self.torch = torch
        self.classifier_transform = v2.Compose(
            [
                SquareLetterbox(int(classifier_config["image_size"])),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=checkpoint["normalization_mean"],
                    std=checkpoint["normalization_std"],
                ),
            ]
        )
        self.detector = YOLO(str(detector_path))

    def predict(self, image: Image.Image, image_id: str | None = None) -> dict[str, Any]:
        image = image.convert("RGB")
        classifier_config = self.config["classifier"]
        detector_config = self.config["detector"]
        tensor = self.classifier_transform(image).unsqueeze(0).to(self.device)
        with (
            self.torch.inference_mode(),
            self.torch.amp.autocast("cuda", enabled=self.device.startswith("cuda")),
        ):
            raw_logit = self.classifier(tensor).squeeze().float().cpu().item()
        raw_probability = float(1 / (1 + np.exp(-raw_logit)))
        probability = float(
            temperature_scale([raw_probability], float(classifier_config["temperature"]))[0]
        )
        classifier_positive = probability >= float(classifier_config["threshold"])

        detection = self.detector.predict(
            source=image,
            imgsz=int(detector_config["image_size"]),
            conf=float(detector_config["confidence_threshold"]),
            device=self.device,
            verbose=False,
        )[0]
        boxes = []
        confidence_tiers = detector_config["confidence_tiers"]
        for xyxy, confidence in zip(
            detection.boxes.xyxy.cpu().tolist(),
            detection.boxes.conf.cpu().tolist(),
            strict=True,
        ):
            boxes.append(
                {
                    "xyxy": [round(float(value), 2) for value in xyxy],
                    "confidence": round(float(confidence), 6),
                    "confidence_level": localization_confidence_level(
                        float(confidence),
                        float(confidence_tiers["medium_min"]),
                        float(confidence_tiers["high_min"]),
                    ),
                }
            )
        decision = fuse_decisions(classifier_positive, len(boxes))
        return {
            "schema_version": "1.0",
            "model_bundle": self.config["name"],
            "image_id": image_id,
            "image": {"width": image.width, "height": image.height},
            "classification": {
                "fracture_probability": probability,
                "raw_probability": raw_probability,
                "threshold": float(classifier_config["threshold"]),
                "positive": classifier_positive,
            },
            "localization": {
                "threshold": float(detector_config["confidence_threshold"]),
                "count": len(boxes),
                "max_confidence_level": (
                    max(boxes, key=lambda box: float(box["confidence"]))["confidence_level"]
                    if boxes
                    else None
                ),
                "confidence_tiers": confidence_tiers,
                "boxes": boxes,
            },
            "decision": decision.to_dict(),
            "disclaimer": self.config["disclaimer"],
        }
