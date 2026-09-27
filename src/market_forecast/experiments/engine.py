from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

from market_forecast.analysis import series_diagnostics
from market_forecast.config import ExperimentConfig
from market_forecast.data import load_bundled_crypto, quality_report, select_asset
from market_forecast.evaluation import evaluate_final_holdout, evaluate_models, expanding_window_folds
from market_forecast.models import default_registry
from market_forecast.reports import write_report

from .artifacts import create_run_directory, environment_manifest, write_json, write_predictions
from .tuning import tune_on_validation_folds


def run_experiment(config: ExperimentConfig) -> Path:
    random.seed(config.seed)
    np.random.seed(config.seed)
    data = select_asset(load_bundled_crypto(config.data_path), config.asset_id)
    if config.max_train_rows is not None:
        required = config.final_test_size + config.folds * max(config.horizons) + config.min_train_size
        data = data.iloc[-max(config.max_train_rows, required):].reset_index(drop=True)
    folds = expanding_window_folds(
        len(data), folds=config.folds, test_size=max(config.horizons),
        final_test_size=config.final_test_size, min_train_size=config.min_train_size,
    )
    parameters = config.model_params or {}
    parameters.setdefault("seasonal_naive", {"period": config.seasonal_period})
    registry = default_registry()
    tuning_results = pd.DataFrame()
    if config.tuning_grid:
        parameters, tuning_results = tune_on_validation_folds(
            data, folds=folds, grids=config.tuning_grid, horizons=config.horizons,
            registry=registry, seed=config.seed, base_parameters=parameters,
        )
    validation = evaluate_models(
        data, folds=folds, models=config.models, horizons=config.horizons,
        registry=registry, parameters=parameters, strategy=config.strategy, seed=config.seed,
    )
    successful = tuple(sorted(validation.metrics["model"].unique())) if not validation.metrics.empty else ()
    final = evaluate_final_holdout(
        data, final_test_size=config.final_test_size, models=successful, horizons=config.horizons,
        registry=registry, parameters=parameters, strategy=config.strategy, seed=config.seed,
    )
    run = create_run_directory(config.artifacts_dir, config.to_dict())
    predictions = pd.concat([validation.predictions, final.predictions], ignore_index=True)
    metrics = pd.concat([validation.metrics, final.metrics], ignore_index=True)
    prediction_file, warning = write_predictions(run, predictions)
    metrics.to_csv(run / "metrics.csv", index=False)
    if not tuning_results.empty:
        tuning_results.to_csv(run / "tuning_results.csv", index=False)
    write_json(run / "config.json", config.to_dict())
    write_json(run / "environment.json", environment_manifest(config.seed))
    write_json(run / "data_manifest.json", {
        "asset_id": config.asset_id, "source": str(config.data_path), "quality": quality_report(data),
        "folds": [{"fold": fold.fold, "train_start": fold.train_indices[0], "train_end": fold.train_indices[-1],
                   "test_start": fold.test_indices[0], "test_end": fold.test_indices[-1]} for fold in folds],
        "final_test_start": len(data) - config.final_test_size, "prediction_file": prediction_file,
        "selected_model_parameters": parameters,
    })
    write_json(run / "diagnostics.json", {
        "series": series_diagnostics(data.iloc[:len(data) - config.final_test_size]["close"].to_numpy()),
        "models": validation.diagnostics + final.diagnostics,
    })
    write_json(run / "failures.json", validation.failures + final.failures)
    write_json(run / "warnings.json", [warning] if warning else [])
    selection_values = data.iloc[:len(data) - config.final_test_size]["close"].to_numpy()
    for model_name in successful:
        params = dict(parameters.get(model_name, {}))
        if model_name in {"ridge", "random_forest", "xgboost", "rnn", "lstm", "gru"}:
            params.setdefault("seed", config.seed)
        model = registry.create(model_name, params).fit(selection_values)
        suffix = ".pt" if model_name in {"rnn", "lstm", "gru"} else ".bin"
        model.save(run / "model" / f"{model_name}{suffix}")
    write_report(run)
    return run
