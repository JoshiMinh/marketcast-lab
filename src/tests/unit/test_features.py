import pandas as pd

from market_forecast.features import add_past_features


def test_lags_and_rolls_never_cross_asset_boundaries_or_use_current_value() -> None:
    frame = pd.DataFrame(
        {
            "asset_id": ["b", "a", "b", "a", "b", "a"],
            "timestamp": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
                utc=True,
            ),
            "close": [100.0, 1.0, 200.0, 2.0, 300.0, 3.0],
        }
    )
    featured = add_past_features(frame, lags=(1,), rolling_windows=(2,))
    first_rows = featured.groupby("asset_id").head(1)
    assert first_rows["close_lag_1"].isna().all()
    assert first_rows["close_rolling_mean_2"].isna().all()
    a = featured[featured["asset_id"] == "a"].reset_index(drop=True)
    b = featured[featured["asset_id"] == "b"].reset_index(drop=True)
    assert a["close_lag_1"].tolist()[1:] == [1.0, 2.0]
    assert b["close_lag_1"].tolist()[1:] == [100.0, 200.0]
    assert a.loc[2, "close_rolling_mean_2"] == 1.5
    assert b.loc[2, "close_rolling_mean_2"] == 150.0


def test_changing_current_close_does_not_change_same_rows_past_features() -> None:
    frame = pd.DataFrame(
        {
            "asset_id": ["a"] * 3,
            "timestamp": pd.date_range("2024-01-01", periods=3, tz="UTC"),
            "close": [1.0, 2.0, 3.0],
        }
    )
    original = add_past_features(frame, lags=(1,), rolling_windows=(2,))
    frame.loc[2, "close"] = 999.0
    changed = add_past_features(frame, lags=(1,), rolling_windows=(2,))
    columns = ["close_lag_1", "close_rolling_mean_2", "log_return_lag_1", "volatility_7"]
    pd.testing.assert_series_equal(original.loc[2, columns], changed.loc[2, columns])


def test_calendar_trend_and_volatility_features_are_available() -> None:
    frame = pd.DataFrame({
        "asset_id": ["a"] * 10,
        "timestamp": pd.date_range("2024-01-01", periods=10, tz="UTC"),
        "close": range(1, 11),
    })
    result = add_past_features(frame)
    assert {"trend_index", "day_of_week", "month", "volatility_7"} <= set(result.columns)
    assert result["trend_index"].tolist() == list(range(10))

