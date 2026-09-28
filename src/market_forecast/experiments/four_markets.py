from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
from time import perf_counter

from market_forecast.config import ExperimentConfig, load_assets
from market_forecast.reports import write_cross_market_report
from .engine import run_experiment


def run_four_markets(config_path: str | Path, output: str | Path = "artifacts/phase3",
                     assets_path: str | Path = "configs/assets_four.json") -> Path:
    base = ExperimentConfig.from_json(config_path)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    runs = []
    for asset in load_assets(assets_path):
        provider, asset_id, path, period = asset.provider, asset.asset_id, asset.data_path, asset.seasonal_period
        parameters = {key: dict(value) for key, value in (base.model_params or {}).items()}
        parameters["holt_winters"] = {"seasonal_period": period}
        parameters["sarima"] = {"order": [1, 1, 0], "seasonal_order": [1, 0, 0, period]}
        config = replace(base, provider=provider, asset_id=asset_id, data_path=Path(path),
                         seasonal_period=period, model_params=parameters,
                         provider_options=asset.provider_options)
        started = perf_counter()
        run = run_experiment(config)
        runs.append({"asset_id": asset_id, "asset_class": asset_id.split(":")[0],
                     "run_id": run.name, "run_dir": str(run), "wall_seconds": perf_counter() - started})
        print(f"{asset_id}: {run} ({runs[-1]['wall_seconds']:.1f}s)", flush=True)
    (destination / "run_index.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    (destination / "study_manifest.json").write_text(json.dumps({
        "config_path": str(config_path), "assets_path": str(assets_path), "synthetic": False,
    }, indent=2), encoding="utf-8")
    write_cross_market_report(destination)
    return destination
