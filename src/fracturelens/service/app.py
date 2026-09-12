from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError

from fracturelens.inference.pipeline import FractureLensPipeline

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DISCLAIMER = "Research prototype only; not for diagnosis or treatment decisions."

app = FastAPI(
    title="FractureLens API",
    version="0.1.0",
    description="Fused fracture classification and localization research API.",
)


@lru_cache(maxsize=1)
def get_pipeline() -> FractureLensPipeline:
    return FractureLensPipeline(
        PROJECT_ROOT,
        PROJECT_ROOT / "configs/inference/track_b_fused.json",
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "fracturelens", "disclaimer": DISCLAIMER}


@app.post("/v1/predict")
async def predict(
    request: Request,
    file: Annotated[
        UploadFile | None,
        File(description="JPEG, PNG, or WebP X-ray image (maximum 25 MiB)"),
    ] = None,
) -> dict:
    if file is None:
        content_type = request.headers.get("content-type", "").split(";", maxsplit=1)[0]
        body = await request.body()
    else:
        content_type = (file.content_type or "").split(";", maxsplit=1)[0]
        body = await file.read(25 * 1024 * 1024 + 1)

    if content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image")
    if not body or len(body) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be between 1 byte and 25 MiB")
    try:
        with Image.open(BytesIO(body)) as source:
            source.load()
            image = source.copy()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=422, detail="Image could not be decoded") from exc
    return get_pipeline().predict(image)
