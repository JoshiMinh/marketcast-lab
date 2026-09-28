"""Deterministic synthetic inputs for network-free software checks."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from market_forecast.config import AssetSpec
from .study import AssetSource, run_asset_study


def build_offline_fixture(config_path: str | Path = "configs/four_asset_full.json",
                          output: str | Path = "assets/offline-fixture",
                          assets_path: str | Path = "configs/assets_four.json") -> Path:
    fixture_dir = Path("data/fixtures/generated")
    fixture_dir.mkdir(parents=True, exist_ok=True)
    cutoff = pd.Timestamp("2025-10-14", tz="UTC")
    business = pd.bdate_range(end=cutoff, periods=210, tz="UTC").difference(
        pd.DatetimeIndex(["2025-07-04", "2025-09-01"], tz="UTC"))[-180:]
    centers = {"crypto": 100, "equity": 300, "forex": 1.1, "oil": 70,
               "gas": 3.0, "yield": 4.0}

    def source_for(asset: AssetSpec, index: int) -> AssetSource:
        asset_class, symbol = asset.asset_id.split(":", 1)
        symbol = symbol.replace("-", "/") if asset_class in {"crypto", "forex"} else symbol
        dates = pd.date_range(end=cutoff, periods=180, tz="UTC") if asset_class == "crypto" else business
        center = centers.get(asset_class, 50)
        rng = np.random.default_rng(410 + index)
        scale = center * .004
        values = center + np.cumsum(rng.normal(.03 * scale, scale, len(dates)))
        frame = pd.DataFrame({"asset_id": asset.asset_id, "symbol": symbol, "asset_class": asset_class,
                              "timestamp": dates, "frequency": "daily", "open": values,
                              "high": values * 1.002, "low": values * .998, "close": values,
                              "adjusted_close": values * (.995 if asset_class == "equity" else 1),
                              "volume": np.nan, "currency": asset.provider_options.get("currency", "PERCENT" if asset_class == "yield" else "USD"),
                              "provider": "synthetic_fixture", "retrieved_at": cutoff})
        path = fixture_dir / f"{asset.asset_id.replace(':', '_').replace('/', '_')}.csv"
        frame.to_csv(path, index=False)
        unit = asset.provider_options.get("unit", asset.provider_options.get(
            "currency", "percent" if asset_class == "yield" else "USD"))
        return "canonical_fixture_csv", path, {"unit": unit}

    destination = run_asset_study(config_path, output, assets_path, source_for=source_for)
    (destination / "SYNTHETIC_FIXTURE.txt").write_text(
        "Generated data for offline software checks only. Do not interpret as market evidence.\n",
        encoding="utf-8",
    )
    return destination
