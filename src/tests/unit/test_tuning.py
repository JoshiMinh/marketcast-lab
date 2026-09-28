from __future__ import annotations

import pandas as pd
import pytest

from market_forecast.evaluation import BacktestFold
from market_forecast.evaluation.backtest import BacktestResult
from market_forecast.experiments import engine
from market_forecast.models import default_registry


def test_tuning_rejects_candidate_with_missing_fold(monkeypatch: pytest.MonkeyPatch) -> None:
    def evaluate(*args, **kwargs):
        candidate = kwargs["parameters"]["ridge"]["lookback"]
        rows = [1.0] if candidate == 5 else [2.0, 2.0]
        return BacktestResult(pd.DataFrame(), pd.DataFrame({"rmse": rows}), [], [])

    monkeypatch.setattr(engine, "evaluate_models", evaluate)
    folds = (
        BacktestFold(1, (0, 1), (2,)),
        BacktestFold(2, (0, 1, 2), (3,)),
    )
    selected, results = engine.tune_on_validation_folds(
        pd.DataFrame(), folds=folds,
        grids={"ridge": [{"lookback": 5}, {"lookback": 10}]},
        horizons=(1,), registry=default_registry(), seed=42, base_parameters={},
    )
    assert selected["ridge"]["lookback"] == 10
    assert results["eligible"].tolist() == [False, True]
