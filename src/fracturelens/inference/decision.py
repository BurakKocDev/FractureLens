from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FusionDecision:
    classifier_positive: bool
    local_evidence: bool
    triage_positive: bool
    strict_positive: bool
    agreement: bool
    review_required: bool
    status: str

    def to_dict(self) -> dict[str, bool | str]:
        return asdict(self)


def fuse_decisions(classifier_positive: bool, detected_regions: int) -> FusionDecision:
    if detected_regions < 0:
        raise ValueError("detected_regions cannot be negative")
    local_evidence = detected_regions > 0
    agreement = classifier_positive == local_evidence
    if agreement and classifier_positive:
        status = "fracture_signal_with_local_evidence"
    elif agreement:
        status = "no_fracture_signal"
    elif classifier_positive:
        status = "global_signal_without_local_evidence"
    else:
        status = "local_evidence_without_global_signal"
    return FusionDecision(
        classifier_positive=classifier_positive,
        local_evidence=local_evidence,
        triage_positive=classifier_positive or local_evidence,
        strict_positive=classifier_positive and local_evidence,
        agreement=agreement,
        review_required=not agreement,
        status=status,
    )
