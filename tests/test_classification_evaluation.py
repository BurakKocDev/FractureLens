from __future__ import annotations

import math

import pytest

from fracturelens.evaluation.classification import (
    average_precision,
    classification_metrics,
    expected_calibration_error,
    roc_auc,
    select_youden_threshold,
    temperature_scale,
)


def test_perfect_ranking_metrics() -> None:
    labels = [0, 0, 1, 1]
    probabilities = [0.1, 0.2, 0.8, 0.9]

    assert roc_auc(labels, probabilities) == pytest.approx(1.0)
    assert average_precision(labels, probabilities) == pytest.approx(1.0)
    assert select_youden_threshold(labels, probabilities) == pytest.approx(0.8)


def test_metrics_at_frozen_threshold() -> None:
    metrics = classification_metrics([0, 0, 1, 1], [0.1, 0.7, 0.6, 0.9], 0.6)

    assert metrics["tp"] == 2
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 0
    assert metrics["sensitivity"] == pytest.approx(1.0)
    assert metrics["specificity"] == pytest.approx(0.5)
    assert metrics["brier"] == pytest.approx(0.1675)


def test_ties_and_calibration_are_handled() -> None:
    assert roc_auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)
    assert expected_calibration_error([0, 1], [0.5, 0.5]) == pytest.approx(0.0)
    assert math.isnan(roc_auc([1, 1], [0.2, 0.8]))
    assert temperature_scale([0.2, 0.8], 1.0).tolist() == pytest.approx([0.2, 0.8])


@pytest.mark.parametrize(
    ("labels", "probabilities"),
    [([], []), ([0, 2], [0.1, 0.2]), ([0, 1], [0.1, 1.1])],
)
def test_invalid_inputs_raise(labels: list[int], probabilities: list[float]) -> None:
    with pytest.raises(ValueError):
        roc_auc(labels, probabilities)
