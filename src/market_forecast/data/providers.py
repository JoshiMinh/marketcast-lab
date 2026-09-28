"""Four single-series providers. Raw responses are cached outside version control."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO, StringIO
from pathlib import Path
import hashlib
import json

import pandas as pd
import requests

from .schema import CANONICAL_COLUMNS, load_bundled_crypto, select_asset, validate_canonical


@dataclass
class ProviderResult:
    frame: pd.DataFrame
    manifest: dict[str, object]


def _cached_response(url: str, cache: Path, *, refresh: bool) -> tuple[bytes, str]:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if refresh or not cache.exists():
        response = requests.get(url, headers={"User-Agent": "MarketCastLab/0.1 research"}, timeout=60)
        response.raise_for_status()
        cache.write_bytes(response.content)
    return cache.read_bytes(), datetime.fromtimestamp(cache.stat().st_mtime, timezone.utc).isoformat()


def _point_frame(dates: pd.Series, values: pd.Series, *, asset_id: str, symbol: str,
                 asset_class: str, provider: str, retrieved_at: str) -> pd.DataFrame:
    result = pd.DataFrame({"timestamp": pd.to_datetime(dates, utc=True, errors="coerce"),
                           "close": pd.to_numeric(values, errors="coerce")})
    result = result.dropna(subset=["timestamp", "close"]).reset_index(drop=True)
    result["asset_id"], result["symbol"], result["asset_class"] = asset_id, symbol, asset_class
    result["frequency"], result["currency"], result["provider"] = "daily", "USD", provider
    result["retrieved_at"] = retrieved_at
    for column in ("open", "high", "low", "volume"):
        result[column] = float("nan")
    result["adjusted_close"] = result["close"]
    return validate_canonical(result.loc[:, CANONICAL_COLUMNS])


def load_provider(name: str, data_path: str | Path, *, refresh: bool = False) -> ProviderResult:
    path = Path(data_path)
    if name == "canonical_fixture_csv":
        raw = path.read_bytes()
        retrieved_at = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
        frame = validate_canonical(pd.read_csv(path))
        asset_id = str(frame.asset_id.iloc[0])
        source_url, source_name = None, "Deterministic synthetic offline fixture"
        license_status, adjustment, timezone_name = "Generated fixture; redistribution permitted", "synthetic adjusted value", "UTC synthetic observation date"
        symbol_mapping, target = {str(frame.symbol.iloc[0]): asset_id}, "adjusted_close" if asset_id.startswith("equity:") else "close"
    elif name == "crypto_csv":
        frame = select_asset(load_bundled_crypto(path), "crypto:BTC-USD")
        source_url, source_name = None, "Bundled cryptocurrency statistics CSV; upstream provenance unknown"
        license_status, adjustment, timezone_name = "Unknown; do not redistribute beyond existing repository", "none", "UTC date"
        symbol_mapping, target = {"BTC/USD": "crypto:BTC-USD"}, "close"
        retrieved_at = pd.Timestamp(frame["retrieved_at"].iloc[0]).isoformat()
        raw = path.read_bytes()
    elif name == "yahoo_spy":
        # Fixed historical request keeps the sample end reproducible across runs.
        source_url = ("https://query1.finance.yahoo.com/v8/finance/chart/SPY"
                      "?period1=1672531200&period2=1798761600&interval=1d&events=history&includeAdjustedClose=true")
        raw, retrieved_at = _cached_response(source_url, path, refresh=refresh)
        result = json.loads(raw)["chart"]["result"][0]
        quote = result["indicators"]["quote"][0]
        adjusted = result["indicators"]["adjclose"][0]["adjclose"]
        frame = pd.DataFrame({"timestamp": pd.to_datetime(result["timestamp"], unit="s", utc=True).normalize(),
                              **{key: quote[key] for key in ("open", "high", "low", "close", "volume")},
                              "adjusted_close": adjusted})
        frame = frame.dropna(subset=["close", "adjusted_close"]).reset_index(drop=True)
        frame["asset_id"], frame["symbol"], frame["asset_class"] = "equity:SPY", "SPY", "equity"
        frame["frequency"], frame["currency"], frame["provider"] = "daily", "USD", "yahoo_finance_chart"
        frame["retrieved_at"] = retrieved_at
        frame = validate_canonical(frame.loc[:, CANONICAL_COLUMNS])
        source_name, license_status = "Yahoo Finance chart, SPDR S&P 500 ETF Trust", "Personal research only; raw redistribution unverified"
        adjustment, timezone_name = "Yahoo adjusted close (corporate action adjustment); raw OHLC retained", "America/New_York session date stored as UTC midnight"
        symbol_mapping, target = {"SPY": "equity:SPY"}, "adjusted_close"
    elif name == "ecb_eurusd":
        source_url = ("https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A"
                      "?startPeriod=2023-01-01&endPeriod=2026-12-31&format=csvdata")
        raw, retrieved_at = _cached_response(source_url, path, refresh=refresh)
        table = pd.read_csv(StringIO(raw.decode("utf-8")), low_memory=False)
        frame = _point_frame(table["TIME_PERIOD"], table["OBS_VALUE"], asset_id="forex:EUR-USD",
                             symbol="EUR/USD", asset_class="forex", provider="ecb_reference_rate",
                             retrieved_at=retrieved_at)
        source_name, license_status = "ECB euro foreign exchange reference rate", "Free reuse with ECB citation and transformation disclosure"
        adjustment, timezone_name = "none; USD per EUR reference fixing", "Europe/Frankfurt reference date stored as UTC midnight"
        symbol_mapping, target = {"D.USD.EUR.SP00.A": "forex:EUR-USD"}, "close"
    elif name == "eia_wti":
        source_url = "https://www.eia.gov/dnav/pet/hist_xls/RWTCd.xls"
        raw, retrieved_at = _cached_response(source_url, path, refresh=refresh)
        table = pd.read_excel(BytesIO(raw), sheet_name="Data 1", skiprows=3, header=None, names=["date", "price"])
        table = table.loc[pd.to_datetime(table["date"], errors="coerce") >= pd.Timestamp("2023-01-01")]
        frame = _point_frame(table["date"], table["price"], asset_id="oil:WTI-CUSHING-SPOT",
                             symbol="WTI-CUSHING-SPOT", asset_class="oil", provider="eia_wti_spot",
                             retrieved_at=retrieved_at)
        source_name, license_status = "EIA Cushing, OK WTI spot price FOB", "EIA government data is public domain; redistribution permitted with attribution"
        adjustment, timezone_name = "none; spot benchmark, no futures rolls", "America/New_York assessment date stored as UTC midnight"
        symbol_mapping, target = {"RWTC": "oil:WTI-CUSHING-SPOT"}, "close"
    else:
        raise ValueError(f"Unknown provider: {name}")
    manifest = {"provider": name, "source_name": source_name, "source_url": source_url,
                "local_path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                "retrieved_at": retrieved_at, "license_redistribution_status": license_status,
                "retrieval_time_basis": ("source_file_mtime_only; original retrieval unknown" if name == "crypto_csv"
                                         else "generated_fixture_file_mtime" if name == "canonical_fixture_csv"
                                         else "local_download_cache_mtime"),
                "symbol_mapping": symbol_mapping, "timezone": timezone_name, "currency": "USD",
                "adjustment_method": adjustment, "target_column": target,
                "target_semantics": "adjusted ETF closing price" if target == "adjusted_close" else "daily observed spot/reference/close price",
                "calendar": "seven_day" if name == "crypto_csv" else "observed_dates_only",
                "transformations": ["parse source dates", "drop missing source prices", "sort timestamps", "no calendar filling"]}
    return ProviderResult(frame, manifest)
