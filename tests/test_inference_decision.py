import pytest

from fracturelens.inference.decision import fuse_decisions


@pytest.mark.parametrize(
    ("classifier", "regions", "triage", "strict", "review"),
    [
        (False, 0, False, False, False),
        (True, 1, True, True, False),
        (True, 0, True, False, True),
        (False, 2, True, False, True),
    ],
)
def test_fusion_truth_table(
    classifier: bool, regions: int, triage: bool, strict: bool, review: bool
) -> None:
    result = fuse_decisions(classifier, regions)

    assert result.triage_positive is triage
    assert result.strict_positive is strict
    assert result.review_required is review


def test_negative_region_count_is_rejected() -> None:
    with pytest.raises(ValueError):
        fuse_decisions(False, -1)
