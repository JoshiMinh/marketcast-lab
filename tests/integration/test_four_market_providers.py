from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from market_forecast.data.providers import load_provider


@pytest.mark.parametrize("name,expected,raw", [
    ("ecb_eurusd", "forex:EUR-USD", b"TIME_PERIOD,OBS_VALUE\n2025-01-03,1.04\n2025-01-06,1.05\n"),
    ("yahoo_spy", "equity:SPY", json.dumps({"chart": {"result": [{
        "timestamp": [1735862400, 1736121600],
        "indicators": {"quote": [{"open": [100, 101], "high": [101, 102],
                                  "low": [99, 100], "close": [100, 101], "volume": [10, 11]}],
                       "adjclose": [{"adjclose": [90, 91]}]}}]}}).encode()),
])
def test_provider_normalization_offline(monkeypatch, tmp_path: Path, name: str, expected: str, raw: bytes) -> None:
    monkeypatch.setattr("market_forecast.data.providers._cached_response",
                        lambda *_args, **_kwargs: (raw, "2025-02-01T00:00:00+00:00"))
    result = load_provider(name, tmp_path / "source")
    assert result.frame.asset_id.unique().tolist() == [expected]
    assert len(result.frame) == 2
    assert result.frame.timestamp.dt.dayofweek.tolist() == [4, 0]
    assert result.frame.timestamp.diff().dt.days.iloc[1] == 3
    assert result.manifest["symbol_mapping"]
    if name == "yahoo_spy":
        assert result.frame.adjusted_close.tolist() == [90, 91]
        assert result.frame.close.tolist() == [100, 101]
    else:
        assert result.frame.open.isna().all()


def test_eia_point_series_keeps_holidays_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("market_forecast.data.providers._cached_response",
                        lambda *_args, **_kwargs: (b"xls", "2025-02-01T00:00:00+00:00"))
    monkeypatch.setattr("market_forecast.data.providers.pd.read_excel",
                        lambda *_args, **_kwargs: pd.DataFrame({"date": ["2025-01-03", "2025-01-06"],
                                                                 "price": [73.0, 74.0]}))
    result = load_provider("eia_wti", tmp_path / "source")
    assert result.frame.asset_id.unique().tolist() == ["oil:WTI-CUSHING-SPOT"]
    assert len(result.frame) == 2
    assert result.frame.open.isna().all()
    assert result.manifest["adjustment_method"].endswith("no futures rolls")


def test_canonical_offline_fixture_is_labeled_synthetic(tmp_path: Path) -> None:
    source = tmp_path / "fixture.csv"
    pd.DataFrame({"asset_id": ["equity:SPY"], "symbol": ["SPY"], "asset_class": ["equity"],
                  "timestamp": ["2025-01-02"], "frequency": ["daily"], "open": [100],
                  "high": [101], "low": [99], "close": [100], "adjusted_close": [90],
                  "volume": [10], "currency": ["USD"], "provider": ["synthetic_fixture"],
                  "retrieved_at": ["2025-01-03"]}).to_csv(source, index=False)
    result = load_provider("canonical_fixture_csv", source)
    assert result.manifest["target_column"] == "adjusted_close"
    assert "synthetic" in result.manifest["source_name"].lower()
    assert result.frame.adjusted_close.iloc[0] == 90
