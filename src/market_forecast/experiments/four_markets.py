from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
from time import perf_counter

from market_forecast.config import ExperimentConfig
from market_forecast.reports import write_cross_market_report
from .engine import run_experiment


MARKETS = (
    ("crypto_csv", "crypto:BTC-USD", "data/crypto_statistics_data.csv", 7),
    ("yahoo_spy", "equity:SPY", "data/raw/spy_chart.json", 5),
    ("ecb_eurusd", "forex:EUR-USD", "data/raw/ecb_eurusd.csv", 5),
    ("eia_wti", "oil:WTI-CUSHING-SPOT", "data/raw/eia_wti.xls", 5),
)


def run_four_markets(config_path: str | Path, output: str | Path = "artifacts/phase3") -> Path:
    base = ExperimentConfig.from_json(config_path)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    runs = []
    for provider, asset_id, path, period in MARKETS:
        parameters = {key: dict(value) for key, value in (base.model_params or {}).items()}
        parameters["holt_winters"] = {"seasonal_period": period}
        parameters["sarima"] = {"order": [1, 1, 0], "seasonal_order": [1, 0, 0, period]}
        config = replace(base, provider=provider, asset_id=asset_id, data_path=Path(path),
                         seasonal_period=period, model_params=parameters)
        started = perf_counter()
        run = run_experiment(config)
        runs.append({"asset_id": asset_id, "asset_class": asset_id.split(":")[0],
                     "run_id": run.name, "run_dir": str(run), "wall_seconds": perf_counter() - started})
        print(f"{asset_id}: {run} ({runs[-1]['wall_seconds']:.1f}s)", flush=True)
    (destination / "run_index.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    write_cross_market_report(destination)
    return destination
