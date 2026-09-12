from __future__ import annotations


def localization_confidence_level(
    score: float,
    medium_min: float,
    high_min: float,
) -> str:
    """Map a detector score to validation-derived display tiers."""
    if not 0 <= score <= 1:
        raise ValueError("score must be between 0 and 1")
    if not 0 <= medium_min <= high_min <= 1:
        raise ValueError("confidence tier thresholds must be ordered within [0, 1]")
    if score >= high_min:
        return "high"
    if score >= medium_min:
        return "medium"
    return "low"
