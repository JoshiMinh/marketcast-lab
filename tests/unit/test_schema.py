import pandas as pd
import pytest

from market_forecast.data import DataValidationError, normalize_legacy_crypto, select_asset


def legacy_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["ETH", "BTC", "BTC"],
            "crypto": ["ETH/USD", "BTC/USD", "BTC/USD"],
            "date": ["2024-01-01", "2024-01-02", "2024-01-01"],
            "open": [2.0, 11.0, 10.0],
            "high": [2.2, 12.0, 11.0],
            "low": [1.8, 10.0, 9.0],
            "close": [2.1, 11.5, 10.5],
        }
    )


def test_normalization_creates_stable_ids_and_sorts_each_asset() -> None:
    canonical = normalize_legacy_crypto(
        legacy_rows(), retrieved_at=pd.Timestamp("2024-02-01", tz="UTC")
    )
    assert canonical["asset_id"].tolist() == ["crypto:BTC-USD", "crypto:BTC-USD", "crypto:ETH-USD"]
    btc = select_asset(canonical, "crypto:BTC-USD")
    assert btc["timestamp"].is_monotonic_increasing
    assert btc["symbol"].unique().tolist() == ["BTC/USD"]


def test_duplicate_asset_timestamp_is_rejected() -> None:
    rows = pd.concat([legacy_rows(), legacy_rows().iloc[[1]]], ignore_index=True)
    with pytest.raises(DataValidationError, match="Duplicate"):
        normalize_legacy_crypto(rows)


def test_missing_canonical_source_column_is_rejected() -> None:
    with pytest.raises(DataValidationError, match="missing columns"):
        normalize_legacy_crypto(legacy_rows().drop(columns="close"))

