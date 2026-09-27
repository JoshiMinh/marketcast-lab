from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import platform
import subprocess

import pandas as pd


def create_run_directory(root: Path, config: dict[str, object]) -> Path:
    encoded = json.dumps(config, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:8]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = root / f"{timestamp}-{digest}"
    path.mkdir(parents=True, exist_ok=False)
    (path / "model").mkdir()
    (path / "figures").mkdir()
    return path


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def environment_manifest(seed: int) -> dict[str, object]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {"python": platform.python_version(), "platform": platform.platform(), "seed": seed, "git_commit": revision}


def write_predictions(path: Path, predictions: pd.DataFrame) -> tuple[str, str | None]:
    try:
        predictions.to_parquet(path / "fold_predictions.parquet", index=False)
        return "fold_predictions.parquet", None
    except ImportError:
        predictions.to_csv(path / "fold_predictions.csv", index=False)
        return "fold_predictions.csv", "Parquet engine unavailable; predictions stored as CSV"
