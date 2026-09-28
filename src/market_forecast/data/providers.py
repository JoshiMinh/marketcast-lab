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
                 asset_class: str, provider: str, retrieved_at: str, currency: str = "USD") -> pd.DataFrame:
    result = pd.DataFrame({"timestamp": pd.to_datetime(dates, utc=True, errors="coerce"),
                           "close": pd.to_numeric(values, errors="coerce")})
    result = result.dropna(subset=["timestamp", "close"]).reset_index(drop=True)
    result["asset_id"], result["symbol"], result["asset_class"] = asset_id, symbol, asset_class
    result["frequency"], result["currency"], result["provider"] = "daily", currency, provider
    result["retrieved_at"] = retrieved_at
    for column in ("open", "high", "low", "volume"):
        result[column] = float("nan")
    result["adjusted_close"] = result["close"]
    return validate_canonical(result.loc[:, CANONICAL_COLUMNS])


def load_provider(name: str, data_path: str | Path, *, refresh: bool = False,
                  asset_id: str | None = None, symbol: str | None = None,
                  series: str | None = None, currency: str = "USD",
                  source_url: str | None = None, unit: str | None = None) -> ProviderResult:
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
        frame = select_asset(load_bundled_crypto(path), asset_id or "crypto:BTC-USD")
        selected_id = str(frame.asset_id.iloc[0])
        selected_symbol = str(frame.symbol.iloc[0])
        source_url, source_name = None, "Bundled cryptocurrency statistics CSV; upstream provenance unknown"
        license_status, adjustment, timezone_name = "Unknown; do not redistribute beyond existing repository", "none", "UTC date"
        symbol_mapping, target = {selected_symbol: selected_id}, "close"
        retrieved_at = pd.Timestamp(frame["retrieved_at"].iloc[0]).isoformat()
        raw = path.read_bytes()
    elif name in {"yahoo_spy", "yahoo_chart"}:
        symbol = symbol or "SPY"
        selected_id = asset_id or f"equity:{symbol}"
        # Fixed historical request keeps the sample end reproducible across runs.
        source_url = source_url or (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
                                    "?period1=1672531200&period2=1798761600&interval=1d&events=history&includeAdjustedClose=true")
        raw, retrieved_at = _cached_response(source_url, path, refresh=refresh)
        result = json.loads(raw)["chart"]["result"][0]
        quote = result["indicators"]["quote"][0]
        adjusted = result["indicators"]["adjclose"][0]["adjclose"]
        frame = pd.DataFrame({"timestamp": pd.to_datetime(result["timestamp"], unit="s", utc=True).normalize(),
                              **{key: quote[key] for key in ("open", "high", "low", "close", "volume")},
                              "adjusted_close": adjusted})
        frame = frame.dropna(subset=["close", "adjusted_close"]).reset_index(drop=True)
        frame["asset_id"], frame["symbol"], frame["asset_class"] = selected_id, symbol, "equity"
        frame["frequency"], frame["currency"], frame["provider"] = "daily", "USD", "yahoo_finance_chart"
        frame["retrieved_at"] = retrieved_at
        frame = validate_canonical(frame.loc[:, CANONICAL_COLUMNS])
        source_name, license_status = f"Yahoo Finance chart, {symbol}", "Personal research only; raw redistribution unverified"
        adjustment, timezone_name = "Yahoo adjusted close (corporate action adjustment); raw OHLC retained", "America/New_York session date stored as UTC midnight"
        symbol_mapping, target = {symbol: selected_id}, "adjusted_close"
    elif name in {"ecb_eurusd", "ecb_reference"}:
        series = series or "D.USD.EUR.SP00.A"
        selected_id = asset_id or "forex:EUR-USD"
        quote = currency
        selected_symbol = selected_id.split(":", 1)[1].replace("-", "/")
        source_url = source_url or (f"https://data-api.ecb.europa.eu/service/data/EXR/{series}"
                                    "?startPeriod=2023-01-01&endPeriod=2026-12-31&format=csvdata")
        raw, retrieved_at = _cached_response(source_url, path, refresh=refresh)
        table = pd.read_csv(StringIO(raw.decode("utf-8")), low_memory=False)
        frame = _point_frame(table["TIME_PERIOD"], table["OBS_VALUE"], asset_id=selected_id,
                             symbol=selected_symbol, asset_class="forex", provider="ecb_reference_rate",
                             retrieved_at=retrieved_at, currency=quote)
        source_name, license_status = "ECB euro foreign exchange reference rate", "Free reuse with ECB citation and transformation disclosure"
        adjustment, timezone_name = f"none; {quote} per EUR reference fixing", "Europe/Frankfurt reference date stored as UTC midnight"
        symbol_mapping, target = {series: selected_id}, "close"
    elif name in {"eia_wti", "eia_spot_xls"}:
        selected_id = asset_id or "oil:WTI-CUSHING-SPOT"
        symbol = symbol or selected_id.split(":", 1)[1]
        source_url = source_url or "https://www.eia.gov/dnav/pet/hist_xls/RWTCd.xls"
        raw, retrieved_at = _cached_response(source_url, path, refresh=refresh)
        table = pd.read_excel(BytesIO(raw), sheet_name="Data 1", skiprows=3, header=None, names=["date", "price"])
        table = table.loc[pd.to_datetime(table["date"], errors="coerce") >= pd.Timestamp("2023-01-01")]
        frame = _point_frame(table["date"], table["price"], asset_id=selected_id,
                             symbol=symbol, asset_class=selected_id.split(":", 1)[0], provider="eia_spot",
                             retrieved_at=retrieved_at)
        source_name, license_status = f"EIA {symbol} spot price", "Check source data reuse terms before redistribution"
        adjustment, timezone_name = "none; spot benchmark, no futures rolls", "America/New_York assessment date stored as UTC midnight"
        symbol_mapping, target = {symbol: selected_id}, "close"
    elif name == "fred_csv":
        series = series or "DGS10"
        selected_id = asset_id or f"yield:{series}"
        source_url = source_url or f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
        raw, retrieved_at = _cached_response(source_url, path, refresh=refresh)
        table = pd.read_csv(StringIO(raw.decode("utf-8")))
        frame = _point_frame(table.iloc[:, 0], table[series], asset_id=selected_id,
                             symbol=series, asset_class="yield", provider="fred",
                             retrieved_at=retrieved_at, currency="PERCENT")
        source_name, license_status = f"FRED {series}; original source Federal Reserve", "Check series reuse terms"
        adjustment, timezone_name = "none; percentage yield", "US observation date stored as UTC midnight"
        symbol_mapping, target = {series: selected_id}, "close"
    else:
        raise ValueError(f"Unknown provider: {name}")
    manifest = {"provider": name, "source_name": source_name, "source_url": source_url,
                "local_path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                "retrieved_at": retrieved_at, "license_redistribution_status": license_status,
                "retrieval_time_basis": ("source_file_mtime_only; original retrieval unknown" if name == "crypto_csv"
                                         else "generated_fixture_file_mtime" if name == "canonical_fixture_csv"
                                         else "local_download_cache_mtime"),
                "symbol_mapping": symbol_mapping, "timezone": timezone_name,
                "currency": str(frame.currency.iloc[0]), "unit": unit or ("percent" if name == "fred_csv" else str(frame.currency.iloc[0])),
                "adjustment_method": adjustment, "target_column": target,
                "target_semantics": ("percentage yield" if str(frame.asset_class.iloc[0]) == "yield" else
                                     "adjusted ETF closing price" if target == "adjusted_close" else
                                     "daily observed spot/reference/close price"),
                "target_kind": "yield" if str(frame.asset_class.iloc[0]) == "yield" else "price",
                "calendar": "seven_day" if name == "crypto_csv" else "observed_dates_only",
                "transformations": ["parse source dates", "drop missing source prices", "sort timestamps", "no calendar filling"]}
    return ProviderResult(frame, manifest)
