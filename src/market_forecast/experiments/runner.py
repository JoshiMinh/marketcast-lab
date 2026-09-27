from __future__ import annotations

import json
import platform
import random
from pathlib import Path

import numpy as np
import pandas as pd

from market_forecast.config import BaselineExperimentConfig
from market_forecast.data import load_bundled_crypto, quality_report, select_asset
from market_forecast.evaluation import chronological_partitions, regression_metrics
from market_forecast.models import DriftBaseline, LastValueBaseline, SeasonalNaiveBaseline


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_btc_baseline_experiment(config: BaselineExperimentConfig) -> Path:
    random.seed(config.seed)
    np.random.seed(config.seed)
    canonical = load_bundled_crypto(config.data_path)
    btc = select_asset(canonical, config.asset_id)
    partitions = chronological_partitions(
        btc,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
    )

    train_values = partitions.train["close"].to_numpy()
    validation_values = partitions.validation["close"].to_numpy()
    models: dict[str, object] = {
        "last_value": LastValueBaseline(),
        "drift": DriftBaseline(),
    }
    if config.seasonal_period is not None:
        models["seasonal_naive"] = SeasonalNaiveBaseline(config.seasonal_period)

    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    for name, model in models.items():
        predictions = model.fit(train_values).predict(len(validation_values))
        scores = regression_metrics(validation_values, predictions)
        metric_rows.append({"model": name, "partition": "validation", **scores})
        prediction_rows.append(
            pd.DataFrame(
                {
                    "asset_id": config.asset_id,
                    "partition": "validation",
                    "model": name,
                    "timestamp": partitions.validation["timestamp"],
                    "actual": validation_values,
                    "prediction": predictions,
                }
            )
        )

    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "config.json", config.to_dict())
    _write_json(
        output / "data_manifest.json",
        {
            "source": str(config.data_path),
            "asset_id": config.asset_id,
            "quality": quality_report(btc),
            "boundaries": partitions.boundaries(),
            "final_test_locked": True,
        },
    )
    _write_json(
        output / "environment.json",
        {"python": platform.python_version(), "platform": platform.platform(), "seed": config.seed},
    )
    pd.DataFrame(metric_rows).to_csv(output / "metrics.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(output / "fold_predictions.csv", index=False)
    return output

