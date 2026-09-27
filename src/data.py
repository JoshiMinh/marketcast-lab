import numpy as np
import pandas as pd

from market_forecast.features import add_past_features

def load_data(path="data/crypto_statistics_data.csv"):
    """Load the main statistics dataset."""
    return pd.read_csv(path, parse_dates=["date"])

def load_raw_data(hourly_path="data/Crypto_Hourly_Refined.csv", stats_path="data/Crypto_Stats_Refined.csv"):
    """Load and merge optional raw hourly/stats datasets."""
    try:
        df_hourly = pd.read_csv(hourly_path)
        df_stats = pd.read_csv(stats_path)

        df_hourly["date"] = pd.to_datetime(df_hourly["date"])
        df_stats["date"] = pd.to_datetime(df_stats["date"])

        df_hourly = df_hourly.sort_values("date").reset_index(drop=True)
        df_stats = df_stats.sort_values("date").reset_index(drop=True)

        df_hourly["date_day"] = df_hourly["date"].dt.floor("D")
        df_stats["date_day"] = df_stats["date"]

        df = df_hourly.merge(
            df_stats[["date_day", "average", "range", "pct_change"]],
            on="date_day",
            how="left",
            suffixes=("", "_daily"),
        )
        return df.drop(columns=["date_day"])
    except FileNotFoundError as exc:
        print(f"File not found: {exc}")
        return pd.DataFrame()

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def ma(series, window):
    return series.rolling(window=window).mean()


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain_rolling = pd.Series(gain, index=series.index).rolling(window=period).mean()
    loss_rolling = pd.Series(loss, index=series.index).rolling(window=period).mean()
    loss_rolling = loss_rolling.replace(0, 1e-9)
    rs = gain_rolling / loss_rolling
    return pd.Series(100 - (100 / (1 + rs)), index=series.index)


def engineer_features(df):
    if "close" not in df.columns:
        return df
    prepared = df.copy()
    if "asset_id" not in prepared.columns:
        if "crypto" not in prepared.columns:
            raise ValueError("Feature engineering requires asset_id or crypto")
        prepared["asset_id"] = "crypto:" + prepared["crypto"].astype(str).str.strip()
    if "timestamp" not in prepared.columns:
        if "date" not in prepared.columns:
            raise ValueError("Feature engineering requires timestamp or date")
        prepared["timestamp"] = pd.to_datetime(prepared["date"], utc=True)
    return add_past_features(prepared).dropna(subset=["close_lag_1"]).reset_index(drop=True)
