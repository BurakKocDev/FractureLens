import pytest

from fracturelens.inference.confidence import localization_confidence_level


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.243522, "low"),
        (0.30314465408805, "medium"),
        (0.40, "medium"),
        (0.40748427672956, "high"),
        (0.95, "high"),
    ],
)
def test_localization_confidence_levels(score: float, expected: str) -> None:
    assert localization_confidence_level(score, 0.30314465408805, 0.40748427672956) == expected


@pytest.mark.parametrize("score", [-0.01, 1.01])
def test_invalid_score_is_rejected(score: float) -> None:
    with pytest.raises(ValueError):
        localization_confidence_level(score, 0.30, 0.40)


def test_invalid_threshold_order_is_rejected() -> None:
    with pytest.raises(ValueError):
        localization_confidence_level(0.50, 0.60, 0.40)
