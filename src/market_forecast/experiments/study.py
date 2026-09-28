"""Shared execution and indexing for real and synthetic asset studies."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from collections.abc import Callable

from market_forecast.config import AssetSpec, ExperimentConfig, load_assets
from market_forecast.reports import write_cross_market_report

from .engine import run_experiment


AssetSource = tuple[str, Path, dict[str, str]]


def run_asset_study(
    config_path: str | Path, output: str | Path, assets_path: str | Path,
    *, source_for: Callable[[AssetSpec, int], AssetSource] | None = None,
) -> Path:
    base = ExperimentConfig.from_json(config_path)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    runs = []
    for index, asset in enumerate(load_assets(assets_path)):
        provider, path, options = (
            source_for(asset, index) if source_for else
            (asset.provider, asset.data_path, asset.provider_options)
        )
        period = asset.seasonal_period
        parameters = {key: dict(value) for key, value in (base.model_params or {}).items()}
        parameters["holt_winters"] = {"seasonal_period": period}
        parameters["sarima"] = {"order": [1, 1, 0], "seasonal_order": [1, 0, 0, period]}
        config = replace(
            base, provider=provider, asset_id=asset.asset_id, data_path=path,
            seasonal_period=period, model_params=parameters, provider_options=options,
        )
        started = perf_counter()
        run = run_experiment(config)
        runs.append({
            "asset_id": asset.asset_id, "asset_class": asset.asset_id.split(":")[0],
            "run_id": run.name, "run_dir": str(run), "wall_seconds": perf_counter() - started,
        })
        print(f"{asset.asset_id}: {run} ({runs[-1]['wall_seconds']:.1f}s)", flush=True)
    (destination / "run_index.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    (destination / "study_manifest.json").write_text(json.dumps({
        "config_path": str(config_path), "assets_path": str(assets_path),
        "synthetic": source_for is not None,
    }, indent=2), encoding="utf-8")
    write_cross_market_report(destination)
    return destination
