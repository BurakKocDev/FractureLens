# FracAtlas v7 Data Gate Report

## Decision

**Pass with documented limitations.** FracAtlas v7 is technically usable after canonical path
selection, EXIF-aware decoding, exact-pixel deduplication, exclusion of
label-conflicting duplicate components, and conservative near-duplicate
grouping.

Model training must use the FractureLens clean manifest, not the raw folder
names or the dataset landing-page count.

## Source verification

| Field | Verified value |
|---|---|
| DOI | `10.6084/m9.figshare.22363012.v7` |
| File id | `65518038` |
| Size | `338436460` bytes |
| MD5 | `fe9da2c7c285915ebee69dfdab8fd396` |
| License | CC BY 4.0 |

The archive passed both size and MD5 verification before extraction.

## Archive structure

| Record type | Count |
|---|---:|
| CSV rows / unique image ids | 4,083 / 4,083 |
| Physical JPG files | 4,085 |
| Canonical positive / negative | 719 / 3,364 |
| COCO images / annotations | 719 / 924 |
| YOLO label files | 4,083 |
| Pascal VOC XML files | 4,083 |
| Official positive split | 574 train / 82 validation / 63 test |

`IMG0003375.jpg` and `IMG0003376.jpg` occur in both `Fractured` and
`Non_fractured`. The two pairs are byte-identical; CSV, COCO, and official split
metadata classify both as fractured. The CSV-driven `Fractured` path is the
canonical copy.

## Decode and orientation

- All 4,083 canonical images decode.
- 59 decode only when truncated-JPEG recovery is enabled; this is stored as a
  per-image quality flag.
- 25 carry non-default EXIF orientation.
- COCO and VOC dimensions agree with the image after EXIF orientation is
  applied.
- Without EXIF normalization, 22 VOC records and one COCO record appear to have
  swapped dimensions.

This establishes a mandatory loader rule: **decode with recovery when needed,
then apply EXIF transpose before augmentation or annotation use.**

## Annotation integrity

- All 719 positive canonical images occur in COCO and have at least one
  annotation.
- No COCO box or polygon is empty, invalid, or outside oriented image bounds.
- YOLO and VOC object counts match CSV `fracture_count` for all images.
- COCO–YOLO box conversion has maximum coordinate error below 0.003 pixel.
- COCO–VOC conversion has maximum coordinate error of 0.5 pixel.
- No positive image is missing from the official positive split files.

The annotation formats are therefore internally consistent once EXIF
orientation is handled correctly.

## Duplicate findings

### Exact file bytes

- 12 groups
- 15 redundant images
- No fracture-label conflicts

These counts reproduce the duplicate finding reported in the 2026 FAD-MIL
study.

### Oriented decoded pixels

- 79 groups
- 84 redundant images
- Three groups contain opposing fracture labels
- Seven images belong to those three conflicting components

The label-conflicting groups show the same radiograph with a fracture box/mask
on one record and a negative label on another. All seven members are excluded;
no attempt is made to choose the clinically correct label.

### Near duplicates

- 97 candidate pairs generated within the same anatomical label using pHash,
  dHash, and global structural similarity
- 78 connected components remain after exact-pixel cleaning
- 164 images belong to those components
- Eight pairwise conflicts connect mixed-label components
- 18 records in mixed-label near-duplicate components are excluded

Ten contact sheets covering all candidate pairs were visually reviewed. Many
pairs are differently encoded, cropped, windowed, or bordered versions of the
same or near-identical acquisition. For same-label candidates, the conservative
decision is to retain the records but force the entire component into one split.

## Cleaning result

| Step | Records remaining |
|---|---:|
| Canonical CSV records | 4,083 |
| Remove 7 exact-pixel conflict members | 4,076 |
| Remove 80 consistent exact-pixel redundants | 3,996 |
| Remove 18 near-duplicate conflict-component members | 3,978 |

The exclusion manifest contains 105 rows. Raw files remain untouched.

## Frozen split

Seed: `20260912`

| Split | Total | Positive | Negative |
|---|---:|---:|---:|
| Train | 2,784 | 485 | 2,299 |
| Validation | 395 | 68 | 327 |
| Test | 799 | 141 | 658 |
| **Total** | **3,978** | **694** | **3,284** |

Integrity checks:

- Cross-split duplicate-aware group violations: 0
- Cross-split exact-pixel duplicate violations: 0
- Manifest SHA-256:
  `416b599c6026c9fcaaa8edbe94e65cf8e657b230cca9a6b8ff6684b02c3c40c1`

The split is stratified by fracture label and anatomy while preserving
duplicate-aware groups. It is not patient-level because the source does not
provide patient identifiers.

## Outputs

- `artifacts/data_audit/dataset_summary.json`
- `artifacts/data_audit/image_inventory.csv`
- `artifacts/data_audit/exact_file_duplicates.csv`
- `artifacts/data_audit/exact_pixel_duplicates.csv`
- `artifacts/data_audit/near_duplicate_candidates.csv`
- `artifacts/data_audit/qa_contact_sheet.jpg`
- `artifacts/data_audit/conflicting_label_duplicates.jpg`
- `artifacts/data_audit/mask_overlay_qa.jpg`
- `artifacts/data_audit/near_duplicate_review_01.jpg` through `_10.jpg`
- `manifests/fracatlas_v7_clean_split_seed20260912.csv`
- `manifests/fracatlas_v7_exclusions.csv`
- `manifests/fracatlas_v7_clean_split_summary.json`

Audit artifacts containing local paths or source-image previews remain outside
Git. The portable manifests and this report are versionable.

## Phase 0 closure

- The manifest-backed decoder enforces EXIF transpose and truncated-JPEG
  recovery and returns classification, box, and rasterized mask targets.
- One batch from each frozen split passes structural validation without needing
  the eventual training framework.
- A 20-image anatomy-stratified mask-overlay sheet passed visual review.
- Local compute is an NVIDIA RTX 3050 Ti Laptop GPU with 4 GB VRAM. Initial
  experiments therefore use 640-pixel inputs, automatic mixed precision, and a
  small batch with gradient accumulation where needed.
- Track A configuration is frozen separately under `configs/experiment/`.
