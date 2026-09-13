# Track B Classification Experiment Log

## Scope

Track B evaluates fracture versus non-fracture classification on the frozen,
duplicate-group-aware FracAtlas v7 split. It contains the MobileNetV3-Small
baseline and the final calibrated DenseNet121 comparator. This is a technical
research benchmark, not a clinical diagnostic model.

## Data protocol

- Frozen manifest: `manifests/fracatlas_v7_clean_split_seed20260912.csv`
- Manifest SHA-256: `416b599c6026c9fcaaa8edbe94e65cf8e657b230cca9a6b8ff6684b02c3c40c1`
- Train: 2,784 images (485 positive, 2,299 negative)
- Validation: 395 images (68 positive, 327 negative)
- Test: 799 images (141 positive, 658 negative)
- Cross-split duplicate-group violations: 0
- Cross-split exact-pixel duplicate violations: 0

The split is not patient-level because FracAtlas does not provide patient IDs.
Repeated patients may therefore remain across splits despite duplicate grouping.

## MobileNetV3-Small baseline

The first Track B model is deliberately mobile-oriented. DenseNet121 remains a
planned literature-style comparator rather than the deployment baseline.

- Architecture: torchvision MobileNetV3-Small, ImageNet-1K initialization
- Input: RGB 224 x 224
- Training augmentation: random resized crop, horizontal flip, +/-7 degree rotation
- Evaluation preprocessing: resize to 256, center crop to 224
- Loss: weighted binary cross entropy, positive weight 4.7402
- Optimizer: AdamW, learning rate 3e-4, weight decay 1e-4
- Schedule: cosine annealing
- Batch: 32
- Requested/completed epochs: 15/15
- Model selection: validation AUROC, then validation AUPRC as tie-breaker
- Seed: 20260912
- Best checkpoint: epoch 13
- Elapsed training time: 322.983 seconds
- Peak allocated CUDA memory: 361.2 MiB
- Hardware: NVIDIA GeForce RTX 3050 Ti Laptop GPU
- Training Git revision: `1fbd72c42e586ffbd970125ccba6dc948ff2e44a`

The test split was not loaded by the training script. The decision threshold was
selected by maximizing Youden's J on validation during training and stored in
the checkpoint. The frozen threshold is 0.66845703125.

## Validation results at the frozen threshold

| Metric | Result |
| --- | ---: |
| AUROC | 0.9288 |
| AUPRC | 0.7995 |
| Sensitivity | 0.8529 |
| Specificity | 0.9235 |
| Balanced accuracy | 0.8882 |
| F1 | 0.7682 |
| ECE, 10 bins | 0.1057 |
| Brier score | 0.0936 |

## Frozen test results

| Metric | Point estimate | Bootstrap 95% CI |
| --- | ---: | ---: |
| AUROC | 0.8953 | 0.8643-0.9226 |
| AUPRC | 0.7226 | 0.6551-0.7863 |
| Sensitivity | 0.7305 | 0.6575-0.8014 |
| Specificity | 0.8830 | 0.8565-0.9060 |
| Balanced accuracy | 0.8067 | 0.7672-0.8460 |
| F1 | 0.6417 | 0.5804-0.7034 |
| ECE, 10 bins | 0.1187 | 0.1018-0.1457 |
| Brier score | 0.1285 | 0.1115-0.1477 |

The frozen confusion matrix is TP=103, TN=581, FP=77, FN=38. Test prevalence
is 17.65%, while precision at the validation-selected threshold is 57.22%.

The first evaluation implementation recomputed Youden's threshold from a second
validation inference pass and obtained 0.666992 because of half-precision and
batch-size rounding. It produced the same confusion matrix and metrics, but was
retained as an audit artifact. Evaluation revision `98b87cc` instead loads the
checkpoint-frozen 0.668457 threshold. The `*_test_corrected` directory is the
canonical result.

## Subgroup observations

Only subgroups with meaningful sample counts should be interpreted.

| Test subgroup | n | Positives | AUROC | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hand | 244 | 73 | 0.8365 | 0.7808 | 0.7485 |
| Leg | 418 | 41 | 0.9230 | 0.7317 | 0.9523 |
| Frontal | 444 | 73 | 0.8759 | 0.6712 | 0.8949 |
| Lateral | 235 | 46 | 0.9350 | 0.8696 | 0.8624 |
| Oblique | 66 | 7 | 0.8015 | 0.5714 | 0.8814 |

Hip and shoulder contain only two positive test examples each, so their apparent
zero sensitivity cannot support a stable comparative claim. All 20 hardware
test images are positive, preventing specificity or AUROC estimation for that
subgroup.

## Error analysis

There are 115 thresholded errors: 38 false negatives and 77 false positives.
The largest absolute error counts occur in hand and frontal images, which are
also large subgroups. The most confident false negative received probability
0.000003 and the most confident false positive 0.9976. These extreme errors and
the reliability diagram show that the raw score is not yet a calibrated clinical
confidence.

Qualitative contact sheets suggest possible shortcut sensitivity to multi-image
panels, large borders, unusual positioning, anatomy mixture, and hardware. This
is a hypothesis for controlled validation, not proof of causality. Ground-truth
boxes are drawn on false-negative examples to support review; no diagnostic
reinterpretation of the dataset labels is made.

Generated local artifacts:

- `artifacts/runs/track_b_mobilenet_v3_small_15ep/`
- `artifacts/evaluations/track_b_mobilenet_v3_small_15ep_test_corrected/`
- `error_analysis/worst_false_negatives.jpg`
- `error_analysis/worst_false_positives.jpg`
- `error_analysis/training_curves.png`
- `error_analysis/score_distribution.png`
- `error_analysis/reliability_diagram.png`
- `error_analysis/ranked_errors.csv`

## Baseline decision and completed comparator

The mobile baseline is retained as the lightweight comparison. Its ranking
performance justified the end-to-end path, while its calibration and subgroup
gaps ruled out presenting the raw probability as clinical certainty.

Before training the heavier DenseNet121 comparator, the next validation-only
experiment should test a shortcut-resistant MobileNetV3-Small variant:

1. use letterbox/padded resizing instead of center cropping, preserving the full radiograph;
2. add controlled border and multi-panel augmentation without altering anatomy;
3. select epoch by validation AUPRC rather than AUROC because positives are sparse;
4. fit temperature scaling on validation only and report calibrated ECE/Brier;
5. compare against this baseline without reopening the frozen test during tuning.

These planned steps were completed. The final DenseNet121 run used 224×224
letterbox preprocessing, ImageNet initialization, weighted BCE, AdamW at 3e-4,
batch size 8, and 15 epochs. Epoch 13 was selected by validation AUPRC. It was
then temperature-scaled on validation only with T=1.1858, producing a calibrated
threshold of 0.3595.

## Final DenseNet121 frozen test

| Metric | Point estimate | Bootstrap 95% CI |
| --- | ---: | ---: |
| AUROC | 0.9120 | 0.8829–0.9386 |
| AUPRC | 0.7811 | 0.7177–0.8378 |
| Sensitivity | 0.7730 | 0.6974–0.8429 |
| Specificity | 0.8815 | 0.8567–0.9044 |
| Balanced accuracy | 0.8273 | 0.7872–0.8645 |
| F1 | 0.6646 | 0.6000–0.7210 |
| ECE, 10 bins | 0.0573 | 0.0426–0.0788 |
| Brier score | 0.0841 | 0.0707–0.0991 |

The confusion matrix is TP=109, TN=580, FP=78, and FN=32. The canonical
evaluation artifact is
`artifacts/evaluations/track_b_densenet121_letterbox_auprc_15ep_test_calibrated_b8/`.
The negative-aware detector and fused consistency analysis were also completed;
their results are recorded in `docs/track_b_detection_log.md`.
