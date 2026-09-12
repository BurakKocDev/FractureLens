from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def _arrays(labels: Iterable[int], probabilities: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(list(labels), dtype=np.int64)
    p = np.asarray(list(probabilities), dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1 or y.size == 0:
        raise ValueError("Labels and probabilities must be non-empty one-dimensional arrays")
    if not np.isin(y, (0, 1)).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Expected binary labels and finite probabilities in [0, 1]")
    return y, p


def roc_auc(labels: Iterable[int], probabilities: Iterable[float]) -> float:
    y, p = _arrays(labels, probabilities)
    positives = int(y.sum())
    negatives = int(y.size - positives)
    if not positives or not negatives:
        return math.nan
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    ranks = np.empty(y.size, dtype=np.float64)
    start = 0
    while start < y.size:
        stop = start + 1
        while stop < y.size and sorted_p[stop] == sorted_p[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2
        start = stop
    positive_rank_sum = ranks[y == 1].sum()
    return float((positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def average_precision(labels: Iterable[int], probabilities: Iterable[float]) -> float:
    y, p = _arrays(labels, probabilities)
    positives = int(y.sum())
    if not positives:
        return math.nan
    order = np.argsort(-p, kind="mergesort")
    sorted_y = y[order]
    cumulative_tp = np.cumsum(sorted_y)
    precision = cumulative_tp / np.arange(1, y.size + 1)
    return float(precision[sorted_y == 1].sum() / positives)


def select_youden_threshold(labels: Iterable[int], probabilities: Iterable[float]) -> float:
    y, p = _arrays(labels, probabilities)
    positives = int(y.sum())
    negatives = int(y.size - positives)
    if not positives or not negatives:
        raise ValueError("Threshold selection requires both classes")
    candidates = np.unique(np.concatenate(([0.0], p, [1.0])))
    best_key = (-math.inf, -math.inf, -math.inf)
    best_threshold = 0.5
    for threshold in candidates:
        predicted = p >= threshold
        sensitivity = float(((predicted == 1) & (y == 1)).sum() / positives)
        specificity = float(((predicted == 0) & (y == 0)).sum() / negatives)
        key = (sensitivity + specificity - 1, sensitivity, float(threshold))
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def expected_calibration_error(
    labels: Iterable[int], probabilities: Iterable[float], bins: int = 10
) -> float:
    y, p = _arrays(labels, probabilities)
    if bins < 1:
        raise ValueError("bins must be positive")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (p >= edges[index]) & (p <= edges[index + 1])
        else:
            mask = (p >= edges[index]) & (p < edges[index + 1])
        if mask.any():
            total += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return total


def classification_metrics(
    labels: Iterable[int], probabilities: Iterable[float], threshold: float
) -> dict[str, float | int]:
    y, p = _arrays(labels, probabilities)
    predicted = (p >= threshold).astype(np.int64)
    tp = int(((predicted == 1) & (y == 1)).sum())
    tn = int(((predicted == 0) & (y == 0)).sum())
    fp = int(((predicted == 1) & (y == 0)).sum())
    fn = int(((predicted == 0) & (y == 1)).sum())
    sensitivity = tp / (tp + fn) if tp + fn else math.nan
    specificity = tn / (tn + fp) if tn + fp else math.nan
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "n": int(y.size),
        "positives": int(y.sum()),
        "negatives": int(y.size - y.sum()),
        "threshold": float(threshold),
        "auroc": roc_auc(y, p),
        "auprc": average_precision(y, p),
        "accuracy": float((predicted == y).mean()),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": float((sensitivity + specificity) / 2),
        "precision": precision,
        "f1": float(2 * precision * sensitivity / (precision + sensitivity))
        if precision + sensitivity
        else 0.0,
        "ece_10_bin": expected_calibration_error(y, p, bins=10),
        "brier": float(np.mean((p - y) ** 2)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
