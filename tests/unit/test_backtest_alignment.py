import pandas as pd

from market_forecast.evaluation import evaluate_models, expanding_window_folds
from market_forecast.models import default_registry


def test_models_use_identical_fold_targets_and_horizon_timestamps() -> None:
    frame = pd.DataFrame({
        "asset_id": ["crypto:BTC-USD"] * 80,
        "timestamp": pd.date_range("2024-01-01", periods=80, tz="UTC"),
        "close": range(80),
    })
    folds = expanding_window_folds(80, folds=3, test_size=5, final_test_size=5, min_train_size=40)
    result = evaluate_models(
        frame, folds=folds, models=("last_value", "ridge"), horizons=(1, 5),
        registry=default_registry(), parameters={"ridge": {"lookback": 5}},
    )
    for fold in range(1, 4):
        subset = result.predictions[(result.predictions["fold"] == fold) & (result.predictions["horizon"] == 5)]
        targets = subset.groupby("model")["timestamp"].apply(list)
        assert targets.iloc[0] == targets.iloc[1]
    assert not result.failures


def test_one_step_observed_strategy_is_distinct_and_recorded() -> None:
    frame = pd.DataFrame({
        "asset_id": ["a"] * 30,
        "timestamp": pd.date_range("2024-01-01", periods=30, tz="UTC"),
        "close": range(30),
    })
    folds = expanding_window_folds(30, folds=1, test_size=3, final_test_size=3, min_train_size=20)
    result = evaluate_models(
        frame, folds=folds, models=("last_value",), horizons=(3,), registry=default_registry(),
        strategy="one_step_observed",
    )
    assert result.predictions["prediction"].tolist() == [23.0, 24.0, 25.0]
    assert set(result.predictions["strategy"]) == {"one_step_observed"}

