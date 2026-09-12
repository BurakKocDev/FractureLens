from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectionMatch:
    prediction_index: int
    target_index: int
    iou: float


def pairwise_iou_xyxy(targets: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Return pairwise IoU for target and prediction boxes in xyxy pixels."""
    targets = np.asarray(targets, dtype=np.float32).reshape(-1, 4)
    predictions = np.asarray(predictions, dtype=np.float32).reshape(-1, 4)
    if len(targets) == 0 or len(predictions) == 0:
        return np.zeros((len(targets), len(predictions)), dtype=np.float32)

    top_left = np.maximum(targets[:, None, :2], predictions[None, :, :2])
    bottom_right = np.minimum(targets[:, None, 2:], predictions[None, :, 2:])
    intersection_size = np.clip(bottom_right - top_left, 0, None)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]

    target_size = np.clip(targets[:, 2:] - targets[:, :2], 0, None)
    prediction_size = np.clip(predictions[:, 2:] - predictions[:, :2], 0, None)
    target_area = target_size[:, 0] * target_size[:, 1]
    prediction_area = prediction_size[:, 0] * prediction_size[:, 1]
    union = target_area[:, None] + prediction_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def non_max_suppression_indices(
    boxes: np.ndarray,
    confidences: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Return confidence-ordered indices retained by class-agnostic NMS."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    confidences = np.asarray(confidences, dtype=np.float32).reshape(-1)
    if len(boxes) != len(confidences):
        raise ValueError("boxes and confidences must have equal length")
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be between 0 and 1")
    order = np.argsort(-confidences)
    kept: list[int] = []
    while len(order):
        current = int(order[0])
        kept.append(current)
        if len(order) == 1:
            break
        remaining = order[1:]
        overlaps = pairwise_iou_xyxy(boxes[[current]], boxes[remaining])[0]
        order = remaining[overlaps <= iou_threshold]
    return np.asarray(kept, dtype=np.int64)


def greedy_match(
    targets: np.ndarray,
    predictions: np.ndarray,
    confidences: np.ndarray,
    iou_threshold: float = 0.5,
) -> list[DetectionMatch]:
    """Match predictions to targets once each, processing high confidence first."""
    predictions = np.asarray(predictions, dtype=np.float32).reshape(-1, 4)
    confidences = np.asarray(confidences, dtype=np.float32).reshape(-1)
    if len(predictions) != len(confidences):
        raise ValueError("predictions and confidences must have equal length")
    ious = pairwise_iou_xyxy(targets, predictions)
    unmatched_targets = set(range(len(ious)))
    matches: list[DetectionMatch] = []
    for prediction_index in np.argsort(-confidences):
        candidates = sorted(
            unmatched_targets,
            key=lambda target_index: float(ious[target_index, prediction_index]),
            reverse=True,
        )
        if not candidates:
            continue
        target_index = candidates[0]
        iou = float(ious[target_index, prediction_index])
        if iou >= iou_threshold:
            matches.append(DetectionMatch(int(prediction_index), target_index, iou))
            unmatched_targets.remove(target_index)
    return matches
