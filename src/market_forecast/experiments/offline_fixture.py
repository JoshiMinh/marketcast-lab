"""Small deterministic four-asset fixture for network-free reproduction checks."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
from time import perf_counter

import numpy as np
import pandas as pd

from market_forecast.config import ExperimentConfig
from market_forecast.reports import write_cross_market_report
from .engine import run_experiment


def build_offline_fixture(config_path: str | Path = "configs/four_asset_full.json",
                          output: str | Path = "artifacts/offline-fixture") -> Path:
    base = ExperimentConfig.from_json(config_path)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    fixture_dir = Path("data/fixtures/generated")
    fixture_dir.mkdir(parents=True, exist_ok=True)
    cutoff = pd.Timestamp("2025-10-14", tz="UTC")
    business = pd.bdate_range(end=cutoff, periods=210, tz="UTC").difference(
        pd.DatetimeIndex(["2025-07-04", "2025-09-01"], tz="UTC"))[-180:]
    definitions = (
        ("crypto:BTC-USD", "BTC/USD", "crypto", pd.date_range(end=cutoff, periods=180, tz="UTC"), 7, 100),
        ("equity:SPY", "SPY", "equity", business, 5, 300),
        ("forex:EUR-USD", "EUR/USD", "forex", business, 5, 1.1),
        ("oil:WTI-CUSHING-SPOT", "WTI-CUSHING-SPOT", "oil", business, 5, 70),
    )
    runs = []
    for index, (asset_id, symbol, asset_class, dates, period, center) in enumerate(definitions):
        rng = np.random.default_rng(410 + index)
        scale = center * .004
        values = center + np.cumsum(rng.normal(.03 * scale, scale, len(dates)))
        frame = pd.DataFrame({"asset_id": asset_id, "symbol": symbol, "asset_class": asset_class,
                              "timestamp": dates, "frequency": "daily", "open": values,
                              "high": values * 1.002, "low": values * .998, "close": values,
                              "adjusted_close": values * (.995 if asset_class == "equity" else 1),
                              "volume": np.nan, "currency": "USD", "provider": "synthetic_fixture",
                              "retrieved_at": cutoff})
        path = fixture_dir / f"{asset_class}.csv"
        frame.to_csv(path, index=False)
        parameters = {key: dict(value) for key, value in (base.model_params or {}).items()}
        parameters["holt_winters"] = {"seasonal_period": period}
        parameters["sarima"] = {"order": [1, 1, 0], "seasonal_order": [1, 0, 0, period]}
        config = replace(base, provider="canonical_fixture_csv", asset_id=asset_id, data_path=path,
                         seasonal_period=period, model_params=parameters)
        started = perf_counter()
        run = run_experiment(config)
        runs.append({"asset_id": asset_id, "asset_class": asset_class, "run_id": run.name,
                     "run_dir": str(run), "wall_seconds": perf_counter() - started})
    (destination / "run_index.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    write_cross_market_report(destination)
    (destination / "SYNTHETIC_FIXTURE.txt").write_text(
        "Generated data for offline software checks only. Do not interpret as market evidence.\n", encoding="utf-8")
    return destination
