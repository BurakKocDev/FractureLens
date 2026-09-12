import numpy as np
import pytest

from fracturelens.evaluation.detection import greedy_match, pairwise_iou_xyxy


def test_pairwise_iou_handles_overlap_and_empty_arrays() -> None:
    targets = np.array([[0, 0, 10, 10]], dtype=np.float32)
    predictions = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=np.float32)

    result = pairwise_iou_xyxy(targets, predictions)

    assert result.shape == (1, 2)
    assert result[0, 0] == pytest.approx(1.0)
    assert result[0, 1] == pytest.approx(25 / 175)
    assert pairwise_iou_xyxy(np.empty((0, 4)), predictions).shape == (0, 2)


def test_greedy_match_uses_confidence_order_and_one_to_one_targets() -> None:
    targets = np.array([[0, 0, 10, 10]], dtype=np.float32)
    predictions = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)

    matches = greedy_match(targets, predictions, np.array([0.4, 0.9]))

    assert len(matches) == 1
    assert matches[0].prediction_index == 1
    assert matches[0].target_index == 0
    assert matches[0].iou == pytest.approx(1.0)


def test_greedy_match_rejects_mismatched_confidence_count() -> None:
    with pytest.raises(ValueError, match="equal length"):
        greedy_match(np.empty((0, 4)), np.empty((1, 4)), np.empty((0,)))
