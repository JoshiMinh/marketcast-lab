from __future__ import annotations

import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from market_forecast.analysis import series_diagnostics
from market_forecast.config import ExperimentConfig
from market_forecast.data import quality_report, select_asset
from market_forecast.data.providers import load_provider
from market_forecast.evaluation import BacktestFold, evaluate_final_holdout, evaluate_models, expanding_window_folds
from market_forecast.models import ModelRegistry, default_registry
from market_forecast.reports import write_report, write_asset_eda

from .artifacts import create_run_directory, environment_manifest, write_json, write_predictions


def tune_on_validation_folds(
    frame: pd.DataFrame,
    *,
    folds: tuple[BacktestFold, ...],
    grids: dict[str, list[dict[str, object]]],
    horizons: tuple[int, ...],
    registry: ModelRegistry,
    seed: int,
    base_parameters: dict[str, dict[str, object]],
    target_column: str = "close",
) -> tuple[dict[str, dict[str, object]], pd.DataFrame]:
    selected = {name: dict(params) for name, params in base_parameters.items()}
    rows = []
    for model_name, candidates in grids.items():
        for candidate_index, candidate in enumerate(candidates):
            params = {**selected.get(model_name, {}), **candidate}
            result = evaluate_models(
                frame, folds=folds, models=(model_name,), horizons=horizons,
                registry=registry, parameters={model_name: params}, seed=seed,
                target_column=target_column,
            )
            score = float(result.metrics["rmse"].mean()) if not result.metrics.empty else float("inf")
            rows.append({
                "model": model_name, "candidate": candidate_index, "mean_validation_rmse": score,
                "parameters": params, "failures": len(result.failures),
            })
        model_rows = [row for row in rows if row["model"] == model_name]
        best = min(model_rows, key=lambda row: row["mean_validation_rmse"])
        selected[model_name] = dict(best["parameters"])
    return selected, pd.DataFrame(rows)


def run_experiment(config: ExperimentConfig) -> Path:
    random.seed(config.seed)
    np.random.seed(config.seed)
    provider_result = load_provider(config.provider, config.data_path,
                                    asset_id=config.asset_id, **(config.provider_options or {}))
    data = select_asset(provider_result.frame, config.asset_id)
    if config.end_date is not None:
        data = data.loc[data["timestamp"] <= pd.Timestamp(config.end_date, tz="UTC")].reset_index(drop=True)
    target_column = str(provider_result.manifest["target_column"])
    quality = {**quality_report(data), "provenance": provider_result.manifest}
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
            target_column=target_column,
        )
    with warnings.catch_warnings(record=True) as observed_warnings:
        warnings.simplefilter("always")
        validation = evaluate_models(
            data, folds=folds, models=config.models, horizons=config.horizons,
            registry=registry, parameters=parameters, strategy=config.strategy, seed=config.seed,
            target_column=target_column,
        )
        successful = tuple(sorted(validation.metrics["model"].unique())) if not validation.metrics.empty else ()
        final = evaluate_final_holdout(
            data, final_test_size=config.final_test_size, models=successful, horizons=config.horizons,
            registry=registry, parameters=parameters, strategy=config.strategy, seed=config.seed,
            target_column=target_column,
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
        **provider_result.manifest, "asset_id": config.asset_id, "source": str(config.data_path), "quality": quality,
        "folds": [{"fold": fold.fold, "train_start": fold.train_indices[0], "train_end": fold.train_indices[-1],
                   "test_start": fold.test_indices[0], "test_end": fold.test_indices[-1],
                   "train_start_date": data["timestamp"].iloc[fold.train_indices[0]].isoformat(),
                   "train_end_date": data["timestamp"].iloc[fold.train_indices[-1]].isoformat(),
                   "test_start_date": data["timestamp"].iloc[fold.test_indices[0]].isoformat(),
                   "test_end_date": data["timestamp"].iloc[fold.test_indices[-1]].isoformat()} for fold in folds],
        "final_test_start": len(data) - config.final_test_size, "prediction_file": prediction_file,
        "final_test_start_date": data["timestamp"].iloc[-config.final_test_size].isoformat(),
        "final_test_end_date": data["timestamp"].iloc[-1].isoformat(),
        "selected_model_parameters": parameters,
    })
    write_json(run / "diagnostics.json", {
        "series": series_diagnostics(data.iloc[:len(data) - config.final_test_size][target_column].to_numpy(),
                                     seasonal_period=config.seasonal_period),
        "models": validation.diagnostics + final.diagnostics,
    })
    write_json(run / "failures.json", validation.failures + final.failures)
    write_json(run / "warnings.json", list(dict.fromkeys(
        ([warning] if warning else []) + [f"{type(item.message).__name__}: {item.message}" for item in observed_warnings]
    )))
    write_json(run / "quality_report.json", quality)
    write_asset_eda(run, data.iloc[:-config.final_test_size], target_column, config.seasonal_period)
    selection_values = data.iloc[:len(data) - config.final_test_size][target_column].to_numpy()
    for model_name in successful:
        params = dict(parameters.get(model_name, {}))
        if model_name in {"ridge", "lasso", "random_forest", "xgboost", "rnn", "lstm", "gru"}:
            params.setdefault("seed", config.seed)
        model = registry.create(model_name, params).fit(selection_values)
        suffix = ".pt" if model_name in {"rnn", "lstm", "gru"} else ".bin"
        model.save(run / "model" / f"{model_name}{suffix}")
    write_report(run)
    return run
