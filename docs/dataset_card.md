# FracAtlas v7 Dataset Card

## Identity

- Dataset: FracAtlas
- Pinned DOI: https://doi.org/10.6084/m9.figshare.22363012.v7
- Figshare article id: `22363012`
- Figshare file id: `65518038`
- Published: 2026-06-13
- Archive: `FracAtlas.zip`
- Archive size: `338436460` bytes
- Archive MD5: `fe9da2c7c285915ebee69dfdab8fd396`
- License: CC BY 4.0

## Attribution

Abedeen I, Rahman MA, Prottyasha FZ, et al. FracAtlas: A Dataset for
Fracture Classification, Localization and Segmentation of Musculoskeletal
Radiographs. Scientific Data. 2023;10:521.
https://doi.org/10.1038/s41597-023-02432-4

## Intended project use

FractureLens uses FracAtlas for retrospective research into fracture
classification, localization, instance segmentation, probability calibration,
and selective prediction. It is not used to provide medical diagnosis or
treatment guidance.

## Version-specific observations

The v7 web record says 4,073 images and warns that the archive differs from the
original paper. Direct audit of the pinned archive found:

- 4,083 rows and 4,083 unique `image_id` values in `dataset.csv`
- 4,085 physical JPG files because two positive images are also duplicated in
  the `Non_fractured` directory
- 719 positive and 3,364 negative canonical CSV records
- 924 fracture instances
- 719 COCO image entries and 924 COCO annotations
- 4,083 YOLO label files plus `classes.txt`
- 4,083 Pascal VOC XML files

Counts in FractureLens must therefore be sourced from the pinned archive audit,
not copied from either the v7 landing-page description or the 2023 paper.

## Known quality risks

- No patient identifiers are provided, so patient-level separation cannot be
  guaranteed.
- There are 12 exact-byte duplicate groups with 15 redundant records.
- EXIF-normalized pixel hashing finds 79 duplicate groups with 84 redundant
  records, including three components with conflicting fracture labels.
- Conservative perceptual-hash review finds additional near-duplicate groups,
  including label-conflicting components.
- 59 JPEGs require Pillow's truncated-image recovery but decode successfully.
- 25 images require EXIF orientation normalization; annotations match the
  oriented coordinate system.
- Some images contain text, logos, fixation hardware, or multi-view montages
  that may act as shortcuts.
- Demographic metadata are unavailable at image level.
- Images are JPG rather than original DICOM, so acquisition metadata and native
  pixel depth are absent.

## Cleaning policy

- Preserve the raw archive unchanged.
- Select the canonical file path from the CSV fracture label.
- Remove redundant exact-pixel records, retaining one deterministic
  representative from label-consistent groups.
- Exclude every member of exact or near-duplicate connected components that
  contain conflicting fracture labels.
- Keep label-consistent near-duplicate candidates but force them into the same
  split group.
- Apply EXIF orientation before image decoding and annotation transforms.
- Record truncated-image recovery as a quality flag rather than silently
  ignoring it.

The resulting frozen split manifest contains 3,978 records. This is still not a
patient-level split.

