from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from market_forecast.analysis import residual_diagnostics
from market_forecast.models import ModelRegistry

from .metrics import regression_metrics
from .splits import BacktestFold


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    diagnostics: list[dict[str, object]]
    failures: list[dict[str, object]]


def _params_for(model_name: str, parameters: dict[str, dict[str, object]], seed: int) -> dict[str, object]:
    params = dict(parameters.get(model_name, {}))
    if model_name in {"ridge", "random_forest", "xgboost", "rnn", "lstm", "gru"}:
        params.setdefault("seed", seed)
    return params


def _forecast(
    *, model_name: str, train_values: np.ndarray, test_values: np.ndarray, horizon: int,
    registry: ModelRegistry, params: dict[str, object], strategy: str,
) -> tuple[np.ndarray, float, float, object]:
    if strategy == "recursive":
        model = registry.create(model_name, params)
        started = perf_counter()
        model.fit(train_values)
        fit_seconds = perf_counter() - started
        started = perf_counter()
        forecast = model.predict(horizon)
        return forecast, fit_seconds, perf_counter() - started, model
    predictions, fit_seconds, inference_seconds = [], 0.0, 0.0
    model = None
    for step in range(horizon):
        history = np.concatenate([train_values, test_values[:step]])
        model = registry.create(model_name, params)
        started = perf_counter()
        model.fit(history)
        fit_seconds += perf_counter() - started
        started = perf_counter()
        predictions.append(float(model.predict(1)[0]))
        inference_seconds += perf_counter() - started
    return np.asarray(predictions), fit_seconds, inference_seconds, model


def evaluate_models(
    frame: pd.DataFrame,
    *,
    folds: tuple[BacktestFold, ...],
    models: tuple[str, ...],
    horizons: tuple[int, ...],
    registry: ModelRegistry,
    parameters: dict[str, dict[str, object]] | None = None,
    strategy: str = "recursive",
    seed: int = 42,
) -> BacktestResult:
    if strategy not in {"recursive", "one_step_observed"}:
        raise ValueError("strategy must be recursive or one_step_observed")
    parameters = parameters or {}
    prediction_rows, metric_rows, diagnostics, failures = [], [], [], []
    for fold in folds:
        train = frame.iloc[list(fold.train_indices)]
        test = frame.iloc[list(fold.test_indices)]
        for model_name in models:
            try:
                predicted, fit_seconds, inference_seconds, model = _forecast(
                    model_name=model_name, train_values=train["close"].to_numpy(),
                    test_values=test["close"].to_numpy(), horizon=max(horizons),
                    registry=registry, params=_params_for(model_name, parameters, seed), strategy=strategy,
                )
                for horizon in horizons:
                    actual = test["close"].to_numpy()[:horizon]
                    forecast = predicted[:horizon]
                    scores = regression_metrics(
                        actual, forecast, training=train["close"].to_numpy(), seasonal_period=1,
                    )
                    interval_coverage = np.nan
                    if strategy == "recursive" and hasattr(model, "predict_interval"):
                        interval = model.predict_interval(horizon)
                        if interval is not None:
                            lower, upper = interval
                            interval_coverage = float(np.mean((actual >= lower) & (actual <= upper)))
                    metric_rows.append({
                        "model": model_name, "fold": fold.fold, "partition": "validation",
                        "horizon": horizon, "strategy": strategy, "fit_seconds": fit_seconds,
                        "inference_seconds": inference_seconds, "diagnostics_available": True,
                        "interval_coverage": interval_coverage, **scores,
                    })
                    for step, (timestamp, truth, estimate) in enumerate(
                        zip(test["timestamp"].iloc[:horizon], actual, forecast), start=1
                    ):
                        prediction_rows.append({
                            "asset_id": train["asset_id"].iloc[0], "model": model_name,
                            "fold": fold.fold, "partition": "validation", "horizon": horizon,
                            "strategy": strategy, "step": step, "timestamp": timestamp,
                            "actual": truth, "prediction": estimate,
                        })
                residuals = test["close"].to_numpy()[:max(horizons)] - predicted
                diagnostics.append({"model": model_name, "fold": fold.fold, **model.diagnostics(), **residual_diagnostics(residuals)})
            except Exception as exc:
                failures.append({"model": model_name, "fold": fold.fold, "error_type": type(exc).__name__, "message": str(exc)})
    return BacktestResult(pd.DataFrame(prediction_rows), pd.DataFrame(metric_rows), diagnostics, failures)


def evaluate_final_holdout(
    frame: pd.DataFrame,
    *,
    final_test_size: int,
    models: tuple[str, ...],
    horizons: tuple[int, ...],
    registry: ModelRegistry,
    parameters: dict[str, dict[str, object]] | None = None,
    strategy: str = "recursive",
    seed: int = 42,
) -> BacktestResult:
    split = len(frame) - final_test_size
    fold = BacktestFold(0, tuple(range(split)), tuple(range(split, len(frame))))
    result = evaluate_models(
        frame, folds=(fold,), models=models, horizons=horizons, registry=registry,
        parameters=parameters, strategy=strategy, seed=seed,
    )
    if not result.predictions.empty:
        result.predictions["partition"] = "final_test"
    if not result.metrics.empty:
        result.metrics["partition"] = "final_test"
    return result
