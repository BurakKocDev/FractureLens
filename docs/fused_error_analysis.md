# Fused Track B Error Analysis

## Scope

This report audits the frozen 799-image duplicate-aware test split by joining
the calibrated DenseNet121 classifier with YOLOv8s local evidence at thresholds
selected on validation. It is a retrospective technical analysis, not a
clinical validation study.

## Outcome taxonomy

The 141 positive and 658 negative images divide into eight mutually exclusive
outcomes:

| Ground truth and model outcome | Images |
| --- | ---: |
| Both models identify a positive image | 62 |
| Classifier identifies it; detector has no local evidence | 47 |
| Detector identifies it; classifier is negative | 13 |
| Both models miss a positive image | 19 |
| Both models correctly reject a negative image | 579 |
| Classifier-only false alarm | 74 |
| Detector-only false alarm | 1 |
| Both models false alarm | 4 |

This decomposition explains the fusion results. OR fusion recovers 13 positive
images missed by the classifier and reaches 122/141 sensitivity (86.5%), but it
also accepts every classifier false alarm and reaches 579/658 specificity
(88.0%). AND fusion accepts only the 62 consensus-positive images, yielding
44.0% sensitivity and 99.4% specificity. Neither policy is a clinical decision
rule; the prototype exposes disagreement instead of hiding it.

## Qualitative gallery

The gallery contains up to three deterministic examples from each informative
category. Green boxes are FracAtlas ground truth. Yellow, orange, and red boxes
are low-, medium-, and high-confidence detector candidates. The confidence
colors express model confidence only, not fracture severity.

![Fused Track B error gallery](assets/fused_error_gallery.jpg)

The examples highlight three recurring limitations:

1. A global classifier can correctly flag an image without locating every
   annotated lesion.
2. The detector can localize a valid fracture when the classifier score is
   below its threshold, motivating the disagreement state and OR triage view.
3. Hardware, multi-view composites, and subtle small lesions occur in both
   misses and false alarms and need explicit qualitative review.

The gallery is generated from the frozen split and current inference bundle:

```powershell
python scripts/build_fused_error_gallery.py --examples 3
```

Generated tables and individual overlays are written to
`artifacts/error_gallery/fused_track_b/`. The committed contact sheet is a
compact, attributed derivative of the CC BY 4.0 FracAtlas data; dataset citation
and source details are provided in the project README and dataset card.

## Limitations

- FracAtlas does not provide patient identifiers, so the test split cannot be
  guaranteed patient-independent.
- Ground-truth boxes reflect the dataset annotations and may not encode every
  clinically relevant finding.
- Confidence tiers are validation-derived presentation aids and are not
  calibrated probabilities of a correct localization.
- The error taxonomy is based on one dataset and one frozen operating point;
  external-domain performance remains unknown.
