from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from market_forecast.config import load_assets
from market_forecast.data import quality_report, validate_canonical
from market_forecast.data.providers import load_provider


def test_extended_catalog_has_distinct_assets_and_new_classes() -> None:
    assets = load_assets("configs/assets_extended.json")
    ids = {asset.asset_id for asset in assets}
    assert len(ids) == len(assets)
    assert {"crypto:ETH-USD", "equity:QQQ", "forex:EUR-JPY",
            "gas:HENRY-HUB-SPOT", "yield:US-TREASURY-10Y"} <= ids


def test_bundled_crypto_selects_requested_asset() -> None:
    result = load_provider("crypto_csv", "data/crypto_statistics_data.csv", asset_id="crypto:ETH-USD")
    assert result.frame.asset_id.unique().tolist() == ["crypto:ETH-USD"]
    assert result.manifest["symbol_mapping"] == {"ETH/USD": "crypto:ETH-USD"}


@pytest.mark.parametrize("name,asset_id,options,raw,currency", [
    ("ecb_reference", "forex:EUR-JPY", {"series": "D.JPY.EUR.SP00.A", "currency": "JPY"},
     b"TIME_PERIOD,OBS_VALUE\n2025-01-03,163.4\n2025-01-06,162.9\n", "JPY"),
    ("fred_csv", "yield:US-TREASURY-10Y", {"series": "DGS10"},
     b"DATE,DGS10\n2025-01-03,0\n2025-01-06,-0.1\n", "PERCENT"),
])
def test_new_point_providers_preserve_units_and_dates(monkeypatch, tmp_path: Path, name: str,
                                                       asset_id: str, options: dict[str, str],
                                                       raw: bytes, currency: str) -> None:
    monkeypatch.setattr("market_forecast.data.providers._cached_response",
                        lambda *_args, **_kwargs: (raw, "2025-01-07T00:00:00+00:00"))
    result = load_provider(name, tmp_path / "source", asset_id=asset_id, **options)
    assert result.frame.asset_id.unique().tolist() == [asset_id]
    assert result.frame.currency.unique().tolist() == [currency]
    assert result.frame.timestamp.diff().dt.days.iloc[1] == 3
    assert result.frame.open.isna().all()
    if name == "fred_csv":
        assert result.frame.close.tolist() == [0, -0.1]
        assert quality_report(result.frame)["non_positive_ohlc_count"] == 0


def test_eia_gas_uses_separate_asset_and_units(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("market_forecast.data.providers._cached_response",
                        lambda *_args, **_kwargs: (b"xls", "2025-01-07T00:00:00+00:00"))
    monkeypatch.setattr("market_forecast.data.providers.pd.read_excel",
                        lambda *_args, **_kwargs: pd.DataFrame({"date": ["2025-01-03", "2025-01-06"],
                                                                 "price": [3.2, 3.1]}))
    result = load_provider("eia_spot_xls", tmp_path / "source", asset_id="gas:HENRY-HUB-SPOT",
                           source_url="https://www.eia.gov/dnav/ng/hist_xls/RNGWHHDd.xls", unit="USD/MMBtu")
    assert result.frame.asset_class.unique().tolist() == ["gas"]
    assert result.manifest["unit"] == "USD/MMBtu"
    assert len(validate_canonical(result.frame)) == 2
