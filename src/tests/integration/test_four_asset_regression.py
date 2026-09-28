from __future__ import annotations

import numpy as np
import pandas as pd

from market_forecast.data import validate_canonical
from market_forecast.evaluation import evaluate_models, expanding_window_folds
from market_forecast.models import default_registry


def test_four_asset_calendar_and_metric_regression() -> None:
    calendars = {
        "crypto:BTC-USD": pd.date_range("2025-01-01", periods=100, freq="D", tz="UTC"),
        "equity:SPY": pd.bdate_range("2025-01-01", periods=100, tz="UTC").difference(pd.DatetimeIndex(["2025-01-20"], tz="UTC"))[:80],
        "forex:EUR-USD": pd.bdate_range("2025-01-01", periods=80, tz="UTC"),
        "oil:WTI-CUSHING-SPOT": pd.bdate_range("2025-01-01", periods=81, tz="UTC").difference(pd.DatetimeIndex(["2025-01-20"], tz="UTC")),
    }
    expected = {"crypto:BTC-USD": "2025-04-05", "equity:SPY": "2025-04-16",
                "forex:EUR-USD": "2025-04-15", "oil:WTI-CUSHING-SPOT": "2025-04-16"}
    for asset_id, dates in calendars.items():
        n = len(dates)
        prices = 100.0 + np.arange(n, dtype=float)
        frame = pd.DataFrame({"asset_id": asset_id, "symbol": asset_id.split(":")[1],
                              "asset_class": asset_id.split(":")[0], "timestamp": dates,
                              "frequency": "daily", "open": prices, "high": prices,
                              "low": prices, "close": prices, "adjusted_close": prices,
                              "volume": np.nan, "currency": "USD", "provider": "fixture",
                              "retrieved_at": pd.Timestamp("2025-05-01", tz="UTC")})
        frame = validate_canonical(frame)
        folds = expanding_window_folds(n, folds=3, test_size=5, final_test_size=5, min_train_size=50)
        assert frame.timestamp.iloc[folds[-1].test_indices[-1]].strftime("%Y-%m-%d") == expected[asset_id]
        result = evaluate_models(frame, folds=folds, models=("last_value",), horizons=(1, 5),
                                 registry=default_registry())
        assert not result.failures
        assert len(result.metrics) == 6
        np.testing.assert_allclose(result.metrics.loc[result.metrics.horizon == 1, "rmse"], 1.0, atol=1e-10)
        np.testing.assert_allclose(result.metrics.loc[result.metrics.horizon == 5, "rmse"], np.sqrt(11), atol=1e-10)
        assert result.predictions.asset_id.unique().tolist() == [asset_id]
