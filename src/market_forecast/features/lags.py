from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def add_past_features(
    frame: pd.DataFrame,
    *,
    lags: Iterable[int] = (1, 7),
    rolling_windows: Iterable[int] = (7, 14),
) -> pd.DataFrame:
    """Add features based strictly on earlier observations within each asset."""
    required = {"asset_id", "timestamp", "close"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Cannot create features without columns: {missing}")
    result = frame.sort_values(["asset_id", "timestamp"], kind="stable").copy()
    grouped = result.groupby("asset_id", sort=False, group_keys=False)["close"]
    for lag in lags:
        if lag < 1:
            raise ValueError("lags must be positive")
        result[f"close_lag_{lag}"] = grouped.shift(lag)
    prior_close = grouped.shift(1)
    prior_grouped = prior_close.groupby(result["asset_id"], sort=False)
    for window in rolling_windows:
        if window < 1:
            raise ValueError("rolling windows must be positive")
        result[f"close_rolling_mean_{window}"] = prior_grouped.transform(
            lambda values: values.rolling(window, min_periods=window).mean()
        )
        result[f"close_rolling_std_{window}"] = prior_grouped.transform(
            lambda values: values.rolling(window, min_periods=window).std()
        )
    result["log_return_lag_1"] = grouped.transform(
        lambda values: np.log(values.shift(1) / values.shift(2))
    )
    result["volatility_7"] = grouped.transform(
        lambda values: np.log(values.shift(1) / values.shift(2)).rolling(7, min_periods=7).std()
    )
    result["trend_index"] = result.groupby("asset_id", sort=False).cumcount()
    timestamps = pd.to_datetime(result["timestamp"], utc=True)
    result["day_of_week"] = timestamps.dt.dayofweek
    result["month"] = timestamps.dt.month
    return result.reset_index(drop=True)

