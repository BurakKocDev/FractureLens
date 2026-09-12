import numpy as np
import pytest

from fracturelens.evaluation.classification import (
    average_precision,
    classification_metrics,
    expected_calibration_error,
    roc_auc,
    select_youden_threshold,
)


def test_perfect_ranking_metrics() -> None:
    labels = [0, 0, 1, 1]
    probabilities = [0.1, 0.2, 0.8, 0.9]
    assert roc_auc(labels, probabilities) == pytest.approx(1.0)
    assert average_precision(labels, probabilities) == pytest.approx(1.0)
    assert select_youden_threshold(labels, probabilities) == pytest.approx(0.8)


def test_metrics_confusion_and_calibration() -> None:
    report = classification_metrics([0, 0, 1, 1], [0.1, 0.7, 0.8, 0.4], threshold=0.5)
    assert (report["tp"], report["tn"], report["fp"], report["fn"]) == (1, 1, 1, 1)
    assert report["balanced_accuracy"] == pytest.approx(0.5)
    assert report["brier"] == pytest.approx(np.mean([0.01, 0.49, 0.04, 0.36]))
    assert 0 <= expected_calibration_error([0, 1], [0.1, 0.9]) <= 1


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        roc_auc([0, 1], [0.2])
    with pytest.raises(ValueError):
        classification_metrics([0, 2], [0.2, 0.8], threshold=0.5)
