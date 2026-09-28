from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

CANONICAL_COLUMNS = (
    "asset_id",
    "symbol",
    "asset_class",
    "timestamp",
    "frequency",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
    "currency",
    "provider",
    "retrieved_at",
)
KEY_COLUMNS = ("asset_id", "timestamp")


class DataValidationError(ValueError):
    """Raised when data violates the canonical modeling contract."""


def _retrieved_at(path: Path) -> pd.Timestamp:
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return pd.Timestamp(modified)


def normalize_legacy_crypto(
    frame: pd.DataFrame,
    *,
    provider: str = "bundled_crypto_csv",
    retrieved_at: pd.Timestamp | None = None,
) -> pd.DataFrame:
    required = {"symbol", "crypto", "date", "open", "high", "low", "close"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataValidationError(f"Legacy crypto data is missing columns: {missing}")

    normalized = pd.DataFrame(index=frame.index)
    normalized["symbol"] = frame["crypto"].astype("string").str.strip()
    normalized["asset_id"] = "crypto:" + normalized["symbol"].str.replace("/", "-", regex=False)
    normalized["asset_class"] = "crypto"
    normalized["timestamp"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    normalized["frequency"] = "daily"
    for column in ("open", "high", "low", "close"):
        normalized[column] = pd.to_numeric(frame[column], errors="coerce")
    normalized["adjusted_close"] = normalized["close"]
    normalized["volume"] = pd.Series(pd.NA, index=frame.index, dtype="Float64")
    normalized["currency"] = normalized["symbol"].str.rsplit("/", n=1).str[-1]
    normalized["provider"] = provider
    normalized["retrieved_at"] = (
        retrieved_at if retrieved_at is not None else pd.Timestamp.now(tz="UTC")
    )
    normalized = normalized.loc[:, CANONICAL_COLUMNS]
    return validate_canonical(normalized)


def validate_canonical(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(CANONICAL_COLUMNS).difference(frame.columns))
    if missing:
        raise DataValidationError(f"Canonical data is missing columns: {missing}")

    validated = frame.loc[:, CANONICAL_COLUMNS].copy()
    validated["timestamp"] = pd.to_datetime(validated["timestamp"], utc=True, errors="coerce")
    validated["retrieved_at"] = pd.to_datetime(validated["retrieved_at"], utc=True, errors="coerce")
    if validated[list(KEY_COLUMNS)].isna().any().any():
        raise DataValidationError("asset_id and timestamp must not contain missing values")
    duplicate_mask = validated.duplicated(list(KEY_COLUMNS), keep=False)
    if duplicate_mask.any():
        examples = validated.loc[duplicate_mask, list(KEY_COLUMNS)].head(5).to_dict("records")
        raise DataValidationError(f"Duplicate asset/timestamp keys found: {examples}")
    if validated["close"].isna().any():
        raise DataValidationError("close must be numeric and non-missing")
    if (validated[["open", "high", "low", "close"]] <= 0).any().any():
        raise DataValidationError("Observed OHLC values must be positive")
    partial = validated[["open", "high", "low"]].isna().sum(axis=1).isin([1, 2])
    if partial.any():
        raise DataValidationError("OHLC must be complete or entirely absent for point series")
    return validated.sort_values(list(KEY_COLUMNS), kind="stable").reset_index(drop=True)


def load_bundled_crypto(path: str | Path = "data/crypto_statistics_data.csv") -> pd.DataFrame:
    source = Path(path)
    frame = pd.read_csv(source)
    return normalize_legacy_crypto(frame, retrieved_at=_retrieved_at(source))


def select_asset(frame: pd.DataFrame, asset_id: str) -> pd.DataFrame:
    selected = frame.loc[frame["asset_id"] == asset_id].copy()
    if selected.empty:
        available = sorted(frame["asset_id"].dropna().unique().tolist())
        raise DataValidationError(f"Unknown asset_id {asset_id!r}; available sample: {available[:10]}")
    if selected["asset_id"].nunique() != 1:
        raise DataValidationError("Asset selection must contain exactly one asset")
    return selected.sort_values("timestamp", kind="stable").reset_index(drop=True)


def quality_report(frame: pd.DataFrame) -> dict[str, Any]:
    ordered = frame.sort_values(["asset_id", "timestamp"])
    gaps: dict[str, int] = {}
    weekday_gaps: dict[str, int] = {}
    intervals: dict[str, dict[str, int]] = {}
    for asset_id, group in ordered.groupby("asset_id", sort=True):
        dates = pd.DatetimeIndex(group["timestamp"])
        expected = pd.date_range(dates.min(), dates.max(), freq="D", tz="UTC")
        gaps[str(asset_id)] = int(len(expected.difference(dates)))
        weekday_gaps[str(asset_id)] = int(len(pd.bdate_range(dates.min(), dates.max(), tz="UTC").difference(dates)))
        intervals[str(asset_id)] = {str(k): int(v) for k, v in dates.to_series().diff().dt.days.value_counts().items()}
    invalid_high = ordered["high"] < ordered[["open", "low", "close"]].max(axis=1)
    invalid_low = ordered["low"] > ordered[["open", "high", "close"]].min(axis=1)
    return {
        "row_count": int(len(ordered)),
        "asset_count": int(ordered["asset_id"].nunique()),
        "start": ordered["timestamp"].min().isoformat(),
        "end": ordered["timestamp"].max().isoformat(),
        "duplicate_key_count": int(ordered.duplicated(list(KEY_COLUMNS)).sum()),
        "missing_values": {key: int(value) for key, value in ordered.isna().sum().items()},
        "missing_daily_timestamps_by_asset": gaps,
        "missing_weekday_timestamps_by_asset_including_holidays": weekday_gaps,
        "observation_gap_days_distribution": intervals,
        "non_positive_ohlc_count": int((ordered[["open", "high", "low", "close"]] <= 0).sum().sum()),
        "inconsistent_ohlc_row_count": int((invalid_high | invalid_low).sum()),
        "provider": sorted(ordered["provider"].dropna().unique().tolist()),
        "close_return_outlier_dates": ordered.loc[ordered.groupby("asset_id")["close"].pct_change().abs() > 0.1,
                                                 "timestamp"].dt.strftime("%Y-%m-%d").tolist(),
    }

