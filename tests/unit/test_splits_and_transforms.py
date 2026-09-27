import numpy as np
import pandas as pd
import pytest

from market_forecast.evaluation import chronological_partitions
from market_forecast.features import TrainOnlyStandardScaler


def single_asset_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asset_id": ["crypto:BTC-USD"] * 10,
            "timestamp": pd.date_range("2024-01-01", periods=10, tz="UTC"),
            "close": np.arange(10, dtype=float),
        }
    )


def test_partitions_are_non_overlapping_and_chronological() -> None:
    parts = chronological_partitions(single_asset_frame())
    assert len(parts.train) == 6
    assert len(parts.validation) == 2
    assert len(parts.final_test) == 2
    assert parts.train["timestamp"].max() < parts.validation["timestamp"].min()
    assert parts.validation["timestamp"].max() < parts.final_test["timestamp"].min()


def test_split_rejects_mixed_assets() -> None:
    frame = single_asset_frame()
    frame.loc[9, "asset_id"] = "crypto:ETH-USD"
    with pytest.raises(ValueError, match="one preselected asset"):
        chronological_partitions(frame)


def test_scaler_fit_metadata_proves_only_training_rows_were_seen() -> None:
    parts = chronological_partitions(single_asset_frame())
    scaler = TrainOnlyStandardScaler().fit(
        parts.train[["close"]].to_numpy(), timestamps=parts.train["timestamp"]
    )
    scaler.transform(parts.validation[["close"]].to_numpy())
    scaler.transform(parts.final_test[["close"]].to_numpy())
    assert scaler.fit_row_count_ == len(parts.train)
    assert scaler.fit_end_timestamp_ == parts.train["timestamp"].max()
    assert scaler.mean_[0] == parts.train["close"].mean()

