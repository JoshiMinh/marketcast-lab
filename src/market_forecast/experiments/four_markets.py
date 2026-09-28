"""Run the configured real market asset catalog."""
from __future__ import annotations

from pathlib import Path

from .study import run_asset_study


def run_four_markets(config_path: str | Path, output: str | Path = "assets/phase3",
                     assets_path: str | Path = "configs/assets_four.json") -> Path:
    return run_asset_study(config_path, output, assets_path)
