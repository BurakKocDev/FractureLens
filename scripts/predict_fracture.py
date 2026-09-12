from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import ImageDraw

from fracturelens.data.audit import load_display_image
from fracturelens.inference.pipeline import FractureLensPipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fused FractureLens inference")
    parser.add_argument("image", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--overlay", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = args.image.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    image = load_display_image(image_path).convert("RGB")
    pipeline = FractureLensPipeline(
        PROJECT_ROOT,
        PROJECT_ROOT / "configs/inference/track_b_fused.json",
        device=args.device,
    )
    result = pipeline.predict(image, image_path.name)
    output = json.dumps(result, indent=2)
    print(output)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(output + "\n", encoding="utf-8")
    if args.overlay:
        draw = ImageDraw.Draw(image)
        width = max(2, round(min(image.size) / 250))
        for box in result["localization"]["boxes"]:
            draw.rectangle(tuple(box["xyxy"]), outline="#ff3b30", width=width)
            x, y = box["xyxy"][:2]
            draw.text((x + 3, max(0, y - 14)), f"{box['confidence']:.2f}", fill="#ff3b30")
        args.overlay.parent.mkdir(parents=True, exist_ok=True)
        image.save(args.overlay, quality=92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
