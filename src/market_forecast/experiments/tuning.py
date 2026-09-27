from __future__ import annotations

import pandas as pd

from market_forecast.evaluation import BacktestFold, evaluate_models
from market_forecast.models import ModelRegistry


def tune_on_validation_folds(
    frame: pd.DataFrame,
    *,
    folds: tuple[BacktestFold, ...],
    grids: dict[str, list[dict[str, object]]],
    horizons: tuple[int, ...],
    registry: ModelRegistry,
    seed: int,
    base_parameters: dict[str, dict[str, object]],
) -> tuple[dict[str, dict[str, object]], pd.DataFrame]:
    selected = {name: dict(params) for name, params in base_parameters.items()}
    rows = []
    for model_name, candidates in grids.items():
        for candidate_index, candidate in enumerate(candidates):
            params = {**selected.get(model_name, {}), **candidate}
            result = evaluate_models(
                frame, folds=folds, models=(model_name,), horizons=horizons,
                registry=registry, parameters={model_name: params}, seed=seed,
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
