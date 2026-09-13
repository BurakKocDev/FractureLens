from __future__ import annotations

import json
from pathlib import Path

from fracturelens.inference.pipeline import file_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    config_path = PROJECT_ROOT / "configs/inference/track_b_fused.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    report: dict[str, object] = {"bundle": config["name"], "ready": True, "models": {}}
    for component in ("classifier", "detector"):
        details = config[component]
        path = PROJECT_ROOT / details["weights"]
        exists = path.is_file()
        actual_hash = file_sha256(path) if exists else None
        matches = exists and actual_hash == details["weights_sha256"]
        report["models"][component] = {
            "path": str(path),
            "exists": exists,
            "sha256_matches": matches,
        }
        report["ready"] = bool(report["ready"]) and matches
    print(json.dumps(report, indent=2))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
