# FractureLens Architecture

## Inference path

```mermaid
flowchart LR
    A["JPEG / PNG / WebP"] --> B["Decode + RGB normalization"]
    B --> C["DenseNet121<br/>224×224 letterbox"]
    B --> D["YOLOv8s<br/>640 px"]
    C --> E["Temperature scaling<br/>threshold 0.3595"]
    D --> F["confidence ≥ 0.2435<br/>NMS IoU 0.50"]
    F --> G["Low / medium / high<br/>display tier"]
    E --> H["Fusion and disagreement"]
    G --> H
    H --> I["JSON API response"]
    I --> J["Browser UI"]
    J --> K["Annotated PNG report"]
```

The classifier answers whether the complete radiograph contains a global
fracture signal. The detector independently proposes local regions. OR fusion
is retained as the sensitive research view; AND is exposed as the strict view.
Disagreement becomes an explicit review state rather than being silently
resolved.

## Data and experiment path

```mermaid
flowchart TD
    A["Pinned FracAtlas v7 archive"] --> B["Integrity and schema audit"]
    B --> C["EXIF-normalized pixel hashing"]
    C --> D["Exact + near-duplicate grouping"]
    D --> E["Label-conflict exclusions"]
    E --> F["Frozen duplicate-aware manifest"]
    F --> G["Track A<br/>positive-only reproduction"]
    F --> H["Track B<br/>full classification"]
    F --> I["Track B<br/>negative-aware detection"]
    G --> I
    H --> J["Frozen fused bundle"]
    I --> J
    J --> K["Consistency + error analysis"]
```

## Runtime components

| Component | Responsibility |
| --- | --- |
| `inference/pipeline.py` | Loads verified weights and runs both models |
| `inference/decision.py` | Produces OR, AND, agreement, and review states |
| `inference/confidence.py` | Applies validation-derived display tiers |
| `service/app.py` | Validates uploads and exposes FastAPI endpoints |
| `service/static/index.html` | Local upload, visualization, and PNG export UI |
| `configs/inference/track_b_fused.json` | Freezes weights, hashes, thresholds, and NMS |

## Trust boundaries

1. Upload validation limits format and size but does not assess radiographic
   quality or whether an image is in distribution.
2. SHA-256 checks prevent silently loading different model binaries.
3. Thresholds and post-processing choices are validation-derived and frozen.
4. The API returns evidence and disagreement; it does not make a clinical
   recommendation.
5. Raw data and weights remain local and are not committed to the repository.
