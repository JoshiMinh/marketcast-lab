from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
import sys
from typing import Any
import warnings

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
    sys.path.append(str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from streamlit.runtime import exists as streamlit_runtime_exists
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from src.data import engineer_features, load_data

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="TensorFlow GPU support is not available.*")

RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_LOOKBACK = 30
DEFAULT_MAX_POINTS = 5000
TEST_RATIO = 0.2
DEFAULT_FORECAST_DAYS = 7
MAX_FORECAST_DAYS = 30
MODEL_OPTIONS = ["lstm", "gru", "arima", "prophet", "ensemble"]
OPTIMIZER_OPTIONS = ["adam", "rmsprop"]
MODEL_LABELS = {
    "lstm": "LSTM",
    "gru": "GRU",
    "arima": "ARIMA",
    "prophet": "Prophet",
    "ensemble": "Ensemble",
}
COLOR_MAP = {
    "lstm": "#22d3ee",
    "gru": "#60a5fa",
    "arima": "#fb923c",
    "prophet": "#34d399",
    "ensemble": "#f8fafc",
}
CHART_TEXT_COLOR = "#f8fafc"
CHART_GRID_COLOR = "rgba(148,163,184,0.18)"
PRICE_LINE_COLOR = "#2dd4bf"
PRICE_FILL_COLOR = "rgba(45,212,191,0.10)"
MA_COLOR_MAP = {
    20: "#f59e0b",
    50: "#38bdf8",
    200: "#f97316",
}


def apply_app_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');
        html, body, [class*="css"]  {
            font-family: 'Space Grotesk', sans-serif;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(245,158,11,0.15), transparent 35%),
                radial-gradient(circle at top right, rgba(234,179,8,0.10), transparent 30%),
                linear-gradient(180deg, #000000 0%, #050505 55%, #0a0a0a 100%);
            color: #f5f5f5;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1280px;
        }
        .hero-shell {
            background: linear-gradient(135deg, rgba(8,8,8,0.95) 0%, rgba(20,20,20,0.93) 55%, rgba(34,24,10,0.9) 100%);
            border: 1px solid rgba(245, 158, 11, 0.35);
            border-radius: 28px;
            padding: 2rem 2.2rem;
            box-shadow: 0 20px 45px rgba(2, 6, 23, 0.45);
            margin-bottom: 1.2rem;
            animation: fadeUp 360ms ease-out;
        }
        .hero-kicker {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: rgba(245, 158, 11, 0.18);
            color: #fcd34d;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .hero-title {
            margin: 0.85rem 0 0.35rem 0;
            font-size: 2.4rem;
            line-height: 1.05;
            color: #f8fafc;
            letter-spacing: -0.02em;
        }
        .btc-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 1.9rem;
            height: 1.9rem;
            margin-right: 0.55rem;
            border-radius: 999px;
            border: 1px solid rgba(245, 158, 11, 0.55);
            background: radial-gradient(circle at 30% 30%, #fbbf24, #f59e0b 62%, #b45309 100%);
            color: #111827;
            font-size: 1.2rem;
            font-weight: 700;
            vertical-align: text-bottom;
        }
        .hero-copy {
            margin: 0;
            max-width: 56rem;
            color: #fde68a;
            font-size: 1rem;
        }
        .metric-card {
            background: rgba(10, 10, 10, 0.85);
            border: 1px solid rgba(245, 158, 11, 0.26);
            border-radius: 22px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 30px rgba(2, 6, 23, 0.3);
        }
        .hero-chip-row {
            margin-top: 1rem;
            display: flex;
            gap: 0.55rem;
            flex-wrap: wrap;
        }
        .hero-chip {
            border: 1px solid rgba(245, 158, 11, 0.45);
            background: rgba(30, 20, 5, 0.75);
            color: #fef3c7;
            border-radius: 999px;
            padding: 0.28rem 0.72rem;
            font-size: 0.8rem;
        }
        .sidebar-shell {
            background: rgba(12, 12, 12, 0.78);
            border: 1px solid rgba(148,163,184,0.22);
            border-radius: 16px;
            padding: 0.7rem 0.75rem 0.72rem 0.75rem;
            margin-bottom: 0.65rem;
        }
        .sidebar-eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.07em;
            font-size: 0.78rem;
            color: #fbbf24;
            margin-bottom: 0.2rem;
        }
        .sidebar-copy {
            color: #fde68a;
            font-size: 0.9rem;
            margin: 0;
        }
        .metric-label {
            color: #fbbf24;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.35rem;
        }
        .metric-value {
            color: #f8fafc;
            font-size: 1.65rem;
            font-weight: 700;
            line-height: 1.05;
        }
        .metric-note {
            color: #fde68a;
            font-size: 0.9rem;
            margin-top: 0.35rem;
        }
        .section-note {
            color: #fde68a;
            margin: 0.2rem 0 1rem 0;
        }
        .spotlight-card {
            background: linear-gradient(180deg, rgba(20,20,20,0.92), rgba(8,8,8,0.92));
            border: 1px solid rgba(245, 158, 11, 0.28);
            border-radius: 24px;
            padding: 1.1rem 1.2rem;
            min-height: 150px;
        }
        .spotlight-model {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #fbbf24;
        }
        .spotlight-value {
            font-size: 1.9rem;
            color: #f8fafc;
            font-weight: 700;
            margin: 0.4rem 0 0.35rem 0;
        }
        .spotlight-copy {
            color: #fde68a;
            font-size: 0.92rem;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(0,0,0,0.98), rgba(12,12,12,0.98));
            border-right: 1px solid rgba(148,163,184,0.20);
        }
        [data-testid="stSidebar"] * {
            color: #e5e7eb;
        }
        [data-baseweb="select"] > div,
        [data-baseweb="tag"] {
            background-color: rgba(20,20,20,0.95) !important;
            border-color: rgba(245,158,11,0.45) !important;
            color: #f8fafc !important;
        }
        [data-baseweb="slider"] [role="slider"] {
            background: #f59e0b;
        }
        .stTabs {
            margin-top: 0.75rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.45rem;
            margin-bottom: 0.9rem;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            padding: 0.65rem 1rem;
            background: rgba(18,18,18,0.9);
            border: 1px solid rgba(245, 158, 11, 0.25);
        }
        .stTabs [aria-selected="true"] {
            background: #f59e0b !important;
            color: #1a1204 !important;
            font-weight: 700;
        }
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }
        code, kbd, pre {
            font-family: 'JetBrains Mono', monospace;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_chart_theme(figure: go.Figure) -> go.Figure:
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,10,0.92)",
        font=dict(color=CHART_TEXT_COLOR),
        title_font=dict(color=CHART_TEXT_COLOR),
        legend=dict(
            orientation="h",
            font=dict(color=CHART_TEXT_COLOR),
            bgcolor="rgba(10,10,10,0.75)",
            bordercolor="rgba(148,163,184,0.24)",
            borderwidth=1,
        ),
        hoverlabel=dict(
            bgcolor="rgba(10,10,10,0.95)",
            bordercolor="rgba(148,163,184,0.35)",
            font=dict(color=CHART_TEXT_COLOR),
        ),
    )
    figure.update_xaxes(
        tickfont=dict(color=CHART_TEXT_COLOR),
        title_font=dict(color=CHART_TEXT_COLOR),
        gridcolor=CHART_GRID_COLOR,
        zerolinecolor=CHART_GRID_COLOR,
    )
    figure.update_yaxes(
        tickfont=dict(color=CHART_TEXT_COLOR),
        title_font=dict(color=CHART_TEXT_COLOR),
        gridcolor=CHART_GRID_COLOR,
        zerolinecolor=CHART_GRID_COLOR,
    )
    figure.update_annotations(font=dict(color=CHART_TEXT_COLOR))
    return figure


def create_sequences(series: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for index in range(len(series) - lookback):
        X.append(series[index : index + lookback])
        y.append(series[index + lookback])
    return np.asarray(X), np.asarray(y)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(np.ravel(y_true), np.ravel(y_pred))))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.ravel(np.asarray(y_true, dtype=float))
    y_pred = np.ravel(np.asarray(y_pred, dtype=float))
    mask = y_true != 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


@st.cache_data(show_spinner=False)
def load_prepared_frame(max_points: int = DEFAULT_MAX_POINTS) -> pd.DataFrame:
    df = load_data()
    if df.empty:
        return df

    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df


def crypto_selector_options(df: pd.DataFrame) -> list[tuple[str, str]]:
    if "crypto" not in df.columns:
        return []

    option_frame = df.copy()
    option_frame["crypto"] = option_frame["crypto"].astype(str).str.strip()
    option_frame = option_frame[option_frame["crypto"] != ""]

    if "symbol" in option_frame.columns:
        option_frame["symbol"] = option_frame["symbol"].fillna("").astype(str).str.strip()
    else:
        option_frame["symbol"] = ""

    if "name" in option_frame.columns:
        option_frame["name"] = option_frame["name"].fillna("").astype(str).str.strip()
    else:
        option_frame["name"] = ""

    option_frame = option_frame[["crypto", "symbol", "name"]].drop_duplicates(subset=["crypto"]).sort_values("crypto")
    options: list[tuple[str, str]] = []
    for row in option_frame.itertuples(index=False):
        crypto_code = row.crypto
        label_parts = [crypto_code]
        if row.symbol:
            label_parts.append(f"({row.symbol})")
        if row.name:
            label_parts.append(f"- {row.name}")
        options.append((crypto_code, " ".join(label_parts)))
    return options


def filter_frame_by_crypto(df: pd.DataFrame, selected_crypto: str) -> pd.DataFrame:
    if "crypto" not in df.columns:
        return df
    selected = df[df["crypto"] == selected_crypto].sort_values("date").reset_index(drop=True)
    selected = engineer_features(selected)
    if len(selected) > DEFAULT_MAX_POINTS:
        selected = selected.iloc[-DEFAULT_MAX_POINTS:].reset_index(drop=True)
    return selected


def prepare_split(df: pd.DataFrame, lookback: int = DEFAULT_LOOKBACK, test_ratio: float = TEST_RATIO) -> dict[str, Any] | None:
    if df.empty or "close" not in df.columns:
        return None

    dates = pd.to_datetime(df["date"]).to_numpy() if "date" in df.columns else None
    close_series = df["close"].astype(float).to_numpy()

    split_idx = int(len(close_series) * (1 - test_ratio))
    split_idx = max(split_idx, lookback + 1)
    if split_idx >= len(close_series):
        return None

    train_series = close_series[:split_idx]
    test_series = close_series[split_idx:]
    train_dates = dates[:split_idx] if dates is not None else None
    test_dates = dates[split_idx:] if dates is not None else None

    if len(train_series) <= lookback or len(test_series) == 0:
        return None

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_series.reshape(-1, 1))
    X_train, y_train = create_sequences(train_scaled, lookback=lookback)
    if len(X_train) == 0:
        return None

    return {
        "train_series": train_series,
        "test_series": test_series,
        "train_dates": train_dates,
        "test_dates": test_dates,
        "scaler": scaler,
        "X_train": X_train,
        "y_train": y_train,
        "history": train_series.tolist(),
        "lookback": lookback,
        "full_dates": dates,
        "full_series": close_series,
    }


def prepare_forward_context(df: pd.DataFrame, lookback: int = DEFAULT_LOOKBACK) -> dict[str, Any] | None:
    if df.empty or "close" not in df.columns:
        return None

    close_series = df["close"].astype(float).to_numpy()
    if len(close_series) <= lookback:
        return None

    dates = pd.to_datetime(df["date"]).to_numpy() if "date" in df.columns else None
    scaler = MinMaxScaler()
    scaler.fit(close_series.reshape(-1, 1))
    return {
        "series": close_series,
        "dates": dates,
        "history": close_series.tolist(),
        "lookback": lookback,
        "scaler": scaler,
        "last_close": float(close_series[-1]),
    }


def resolve_sequence_artifact(model_name: str, optimizer: str) -> Path:
    model_root = RESULTS_DIR / optimizer
    candidates = [
        model_root / f"{model_name}.keras",
        model_root / f"{model_name}_model.keras",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing model file for {MODEL_LABELS[model_name]} in {model_root}")


@st.cache_resource(show_spinner=False)
def load_sequence_model(model_name: str, optimizer: str):
    from tensorflow.keras.models import load_model

    return load_model(resolve_sequence_artifact(model_name, optimizer))


def forecast_sequence_model(model: Any, context: dict[str, Any], test_series: np.ndarray) -> np.ndarray:
    predictions = []
    history = list(context["history"])
    scaler = context["scaler"]
    lookback = context["lookback"]

    for actual_value in test_series:
        window = np.asarray(history[-lookback:], dtype=float).reshape(-1, 1)
        window_scaled = scaler.transform(window).reshape(1, lookback, 1)
        pred_scaled = model.predict(window_scaled, verbose=0)
        pred_value = scaler.inverse_transform(pred_scaled)[0, 0]
        predictions.append(float(pred_value))
        history.append(float(actual_value))

    return np.asarray(predictions, dtype=float)


def forecast_sequence_future(model: Any, context: dict[str, Any], forecast_days: int) -> np.ndarray:
    predictions = []
    history = list(context["history"])
    scaler = context["scaler"]
    lookback = context["lookback"]

    for _ in range(forecast_days):
        window = np.asarray(history[-lookback:], dtype=float).reshape(-1, 1)
        window_scaled = scaler.transform(window).reshape(1, lookback, 1)
        pred_scaled = model.predict(window_scaled, verbose=0)
        pred_value = float(scaler.inverse_transform(pred_scaled)[0, 0])
        predictions.append(pred_value)
        history.append(pred_value)

    return np.asarray(predictions, dtype=float)


def forecast_arima(train_series: np.ndarray, test_series: np.ndarray) -> np.ndarray:
    from statsmodels.tsa.arima.model import ARIMA

    fitted = ARIMA(train_series, order=(5, 1, 0)).fit()
    forecast = fitted.forecast(steps=len(test_series))
    return np.asarray(forecast, dtype=float)


def forecast_arima_future(series: np.ndarray, forecast_days: int) -> np.ndarray:
    from statsmodels.tsa.arima.model import ARIMA

    fitted = ARIMA(series, order=(5, 1, 0)).fit()
    forecast = fitted.forecast(steps=forecast_days)
    return np.asarray(forecast, dtype=float)


def _resolve_future_step(dates: np.ndarray | None) -> pd.Timedelta:
    if dates is None or len(dates) < 2:
        return pd.Timedelta(days=1)

    date_index = pd.DatetimeIndex(pd.to_datetime(dates))
    diffs = pd.Series(date_index).diff().dropna()
    positive_diffs = diffs[diffs > pd.Timedelta(0)]
    if positive_diffs.empty:
        return pd.Timedelta(days=1)
    return positive_diffs.median()


def build_future_index(dates: np.ndarray | None, forecast_days: int):
    if dates is None or len(dates) == 0:
        return np.arange(1, forecast_days + 1)

    date_index = pd.DatetimeIndex(pd.to_datetime(dates))
    step = _resolve_future_step(dates)
    return pd.date_range(start=date_index[-1] + step, periods=forecast_days, freq=step)


def forecast_prophet(train_series: np.ndarray, train_dates: np.ndarray | None, test_dates: np.ndarray | None) -> np.ndarray | None:
    if train_dates is None or test_dates is None:
        return None

    try:
        from prophet import Prophet
    except ImportError:
        return None

    train_df = pd.DataFrame({"ds": pd.to_datetime(train_dates), "y": train_series.astype(float)})
    forecast_frame = pd.DataFrame({"ds": pd.to_datetime(test_dates)})

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        interval_width=0.95,
        changepoint_prior_scale=0.05,
        uncertainty_samples=0,
    )
    model.fit(train_df)
    forecast = model.predict(forecast_frame)
    return forecast["yhat"].to_numpy()


def forecast_prophet_future(series: np.ndarray, dates: np.ndarray | None, forecast_days: int) -> tuple[np.ndarray, pd.Index] | tuple[None, None]:
    if dates is None:
        return None, None

    try:
        from prophet import Prophet
    except ImportError:
        return None, None

    train_df = pd.DataFrame({"ds": pd.to_datetime(dates), "y": np.asarray(series, dtype=float)})
    future_index = build_future_index(dates, forecast_days)
    forecast_frame = pd.DataFrame({"ds": pd.to_datetime(future_index)})

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        interval_width=0.95,
        changepoint_prior_scale=0.05,
        uncertainty_samples=0,
    )
    model.fit(train_df)
    forecast = model.predict(forecast_frame)
    return forecast["yhat"].to_numpy(), pd.Index(future_index)


def forecast_ensemble(results: dict[str, dict[str, Any]]) -> np.ndarray | None:
    available_results = [result for result in results.values() if result.get("predictions") is not None]
    if len(available_results) < 2:
        return None

    min_length = min(len(result["predictions"]) for result in available_results)
    stacked_predictions = np.column_stack([result["predictions"][-min_length:] for result in available_results])
    return stacked_predictions.mean(axis=1)


def build_result(model_name: str, actual: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    return {
        "model_key": model_name,
        "model": MODEL_LABELS[model_name],
        "predictions": predictions,
        "actual": actual,
        "rmse": rmse(actual, predictions),
        "mae": float(mean_absolute_error(actual, predictions)),
        "mape": mape(actual, predictions),
        "status": "ready",
    }


def build_future_result(model_name: str, predictions: np.ndarray, future_index, last_close: float) -> dict[str, Any]:
    final_value = float(predictions[-1])
    change_pct = 0.0 if last_close == 0 else float(((final_value - last_close) / last_close) * 100)
    return {
        "model_key": model_name,
        "model": MODEL_LABELS[model_name],
        "predictions": np.asarray(predictions, dtype=float),
        "future_index": future_index,
        "final_value": final_value,
        "change_pct": change_pct,
        "status": "ready",
    }


def run_predictions(selected_models: list[str], context: dict[str, Any], optimizer: str) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    test_series = context["test_series"]

    for model_name in selected_models:
        if model_name in {"lstm", "gru"}:
            try:
                model = load_sequence_model(model_name, optimizer)
                predictions = forecast_sequence_model(model, context, test_series)
                results[model_name] = build_result(model_name, np.asarray(test_series, dtype=float), predictions)
            except Exception as exc:
                results[model_name] = {"model_key": model_name, "model": MODEL_LABELS[model_name], "status": "skipped", "message": str(exc)}
        elif model_name == "arima":
            try:
                predictions = forecast_arima(context["train_series"], test_series)
                results[model_name] = build_result(model_name, np.asarray(test_series, dtype=float), predictions)
            except Exception as exc:
                results[model_name] = {"model_key": model_name, "model": MODEL_LABELS[model_name], "status": "skipped", "message": str(exc)}
        elif model_name == "prophet":
            try:
                predictions = forecast_prophet(context["train_series"], context["train_dates"], context["test_dates"])
                if predictions is None:
                    raise RuntimeError("Prophet requires the package to be installed and date values to be available.")
                results[model_name] = build_result(model_name, np.asarray(test_series, dtype=float), predictions)
            except Exception as exc:
                results[model_name] = {"model_key": model_name, "model": MODEL_LABELS[model_name], "status": "skipped", "message": str(exc)}

    if "ensemble" in selected_models:
        ensemble_predictions = forecast_ensemble(results)
        if ensemble_predictions is None:
            results["ensemble"] = {
                "model_key": "ensemble",
                "model": MODEL_LABELS["ensemble"],
                "status": "skipped",
                "message": "Ensemble requires at least two successful component predictions.",
            }
        else:
            actual = np.asarray(test_series, dtype=float)[-len(ensemble_predictions) :]
            results["ensemble"] = build_result("ensemble", actual, ensemble_predictions)

    return results


def run_future_forecasts(
    selected_models: list[str],
    context: dict[str, Any],
    forecast_days: int,
    optimizer: str,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    future_index = build_future_index(context["dates"], forecast_days)

    for model_name in selected_models:
        if model_name in {"lstm", "gru"}:
            try:
                model = load_sequence_model(model_name, optimizer)
                predictions = forecast_sequence_future(model, context, forecast_days)
                results[model_name] = build_future_result(model_name, predictions, future_index, context["last_close"])
            except Exception as exc:
                results[model_name] = {"model_key": model_name, "model": MODEL_LABELS[model_name], "status": "skipped", "message": str(exc)}
        elif model_name == "arima":
            try:
                predictions = forecast_arima_future(context["series"], forecast_days)
                results[model_name] = build_future_result(model_name, predictions, future_index, context["last_close"])
            except Exception as exc:
                results[model_name] = {"model_key": model_name, "model": MODEL_LABELS[model_name], "status": "skipped", "message": str(exc)}
        elif model_name == "prophet":
            try:
                predictions, prophet_index = forecast_prophet_future(context["series"], context["dates"], forecast_days)
                if predictions is None or prophet_index is None:
                    raise RuntimeError("Prophet requires the package to be installed and date values to be available.")
                results[model_name] = build_future_result(model_name, predictions, prophet_index, context["last_close"])
            except Exception as exc:
                results[model_name] = {"model_key": model_name, "model": MODEL_LABELS[model_name], "status": "skipped", "message": str(exc)}

    if "ensemble" in selected_models:
        ensemble_predictions = forecast_ensemble(results)
        if ensemble_predictions is None:
            results["ensemble"] = {
                "model_key": "ensemble",
                "model": MODEL_LABELS["ensemble"],
                "status": "skipped",
                "message": "Ensemble requires at least two successful future forecasts.",
            }
        else:
            results["ensemble"] = build_future_result("ensemble", ensemble_predictions, future_index, context["last_close"])

    return results


def build_price_history_figure(
    df: pd.DataFrame,
    selected_crypto: str,
    moving_averages: list[int],
    show_volume: bool,
) -> go.Figure:
    has_volume = show_volume and "volume" in df.columns
    figure = make_subplots(specs=[[{"secondary_y": has_volume}]]) if has_volume else go.Figure()
    x_values = pd.to_datetime(df["date"]) if "date" in df.columns else np.arange(len(df))
    close_series = df["close"].astype(float)
    close_trace = go.Scatter(
        x=x_values,
        y=close_series,
        name="Close",
        mode="lines",
        line=dict(color=PRICE_LINE_COLOR, width=2.3),
        fill="tozeroy",
        fillcolor=PRICE_FILL_COLOR,
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Close: $%{y:,.2f}<extra></extra>",
    )
    if has_volume:
        figure.add_trace(close_trace, secondary_y=False)
    else:
        figure.add_trace(close_trace)

    for window in moving_averages:
        if len(df) >= window:
            ma_series = close_series.rolling(window=window).mean()
            ma_trace = go.Scatter(
                x=x_values,
                y=ma_series,
                name=f"MA{window}",
                mode="lines",
                line=dict(color=MA_COLOR_MAP.get(window, "#94a3b8"), width=1.7, dash="dash"),
                hovertemplate=f"Date: %{{x|%Y-%m-%d}}<br>MA{window}: $%{{y:,.2f}}<extra></extra>",
            )
            if has_volume:
                figure.add_trace(ma_trace, secondary_y=False)
            else:
                figure.add_trace(ma_trace)

    if has_volume:
        figure.add_trace(
            go.Bar(
                x=x_values,
                y=df["volume"].astype(float),
                name="Volume",
                marker_color="rgba(96,165,250,0.22)",
                opacity=0.35,
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Volume: %{y:,.0f}<extra></extra>",
            ),
            secondary_y=True,
        )

    figure.update_layout(
        title=f"{selected_crypto} Price History",
        height=340,
        margin=dict(l=10, r=10, t=56, b=10),
        hovermode="x unified",
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all", label="All"),
                ],
                bgcolor="rgba(15,23,42,0.95)",
                activecolor="rgba(45,212,191,0.25)",
                bordercolor="rgba(148,163,184,0.25)",
                font=dict(color=CHART_TEXT_COLOR),
            ),
        ),
    )
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(
        showgrid=True,
        gridcolor=CHART_GRID_COLOR,
        tickprefix="$",
        tickformat=",.0f",
    )
    if has_volume:
        figure.update_yaxes(showgrid=False, secondary_y=True, tickformat=",.0f", title_text="Volume")
    return apply_chart_theme(figure)


def build_comparison_figure(results: dict[str, dict[str, Any]], test_dates: np.ndarray | None) -> go.Figure:
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.08,
        subplot_titles=("Predictions vs Actual", "Residuals"),
    )

    successful_results = {name: result for name, result in results.items() if result.get("status") == "ready"}
    if not successful_results:
        figure.add_annotation(text="No predictions available", x=0.5, y=0.5, showarrow=False, row=1, col=1)
        return figure

    first_result = next(iter(successful_results.values()))
    actual = np.asarray(first_result["actual"], dtype=float)
    x_values = pd.to_datetime(test_dates) if test_dates is not None else np.arange(len(actual))

    figure.add_trace(
        go.Scatter(x=x_values, y=actual, name="Actual", line=dict(color="#111827", width=2.4)),
        row=1,
        col=1,
    )

    for model_name, result in successful_results.items():
        figure.add_trace(
            go.Scatter(
                x=x_values[-len(result["predictions"]):],
                y=result["predictions"],
                name=result["model"],
                mode="lines",
                line=dict(color=COLOR_MAP.get(model_name, "#666666"), width=2.2),
            ),
            row=1,
            col=1,
        )

    best_name = min(successful_results, key=lambda name: successful_results[name]["rmse"])
    best_result = successful_results[best_name]
    residuals = np.asarray(best_result["actual"], dtype=float) - np.asarray(best_result["predictions"], dtype=float)
    figure.add_trace(
        go.Histogram(x=residuals, name=f"Residuals ({best_result['model']})", marker_color="rgba(15,23,42,0.55)"),
        row=2,
        col=1,
    )

    figure.update_layout(
        height=760,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    figure.update_xaxes(title_text="Date", row=1, col=1, showgrid=False)
    figure.update_yaxes(title_text="Price", row=1, col=1, gridcolor=CHART_GRID_COLOR)
    figure.update_xaxes(title_text="Residual", row=2, col=1, showgrid=False)
    figure.update_yaxes(title_text="Count", row=2, col=1, gridcolor=CHART_GRID_COLOR)
    return apply_chart_theme(figure)


def build_future_figure(results: dict[str, dict[str, Any]], forward_context: dict[str, Any]) -> go.Figure:
    figure = go.Figure()
    successful_results = {name: result for name, result in results.items() if result.get("status") == "ready"}
    if not successful_results:
        figure.add_annotation(text="No forward forecasts available", x=0.5, y=0.5, showarrow=False)
        return figure

    last_close = forward_context["last_close"]
    dates = forward_context["dates"]
    if dates is not None and len(dates) > 0:
        last_x = pd.to_datetime(dates[-1])
    else:
        last_x = 0

    for model_name, result in successful_results.items():
        future_index = result["future_index"]
        if isinstance(future_index, pd.Index):
            x_values = pd.to_datetime(future_index)
            anchor_x = [last_x]
        else:
            x_values = future_index
            anchor_x = [0]
        anchor_and_future_x = list(anchor_x) + list(x_values)
        anchor_and_future_y = [last_close] + list(result["predictions"])
        figure.add_trace(
            go.Scatter(
                x=anchor_and_future_x,
                y=anchor_and_future_y,
                name=result["model"],
                mode="lines+markers",
                line=dict(color=COLOR_MAP.get(model_name, "#666666"), width=2.8, shape="spline", smoothing=0.5),
                marker=dict(size=6, symbol="circle"),
                hovertemplate=f"{result['model']}<br>%{{x}}<br>%{{y:.4f}}<extra></extra>",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[anchor_and_future_x[-1]],
                y=[anchor_and_future_y[-1]],
                mode="markers+text",
                text=[f"{anchor_and_future_y[-1]:.2f}"],
                textposition="top center",
                showlegend=False,
                marker=dict(
                    color=COLOR_MAP.get(model_name, "#666666"),
                    size=10,
                    line=dict(color="#111827", width=1),
                ),
                hovertemplate=f"{result['model']} final<br>%{{x}}<br>%{{y:.4f}}<extra></extra>",
            )
        )

    figure.add_hline(y=last_close, line_dash="dot", line_color="rgba(203,213,225,0.55)")
    figure.update_layout(
        title="Forward Forecast Comparison",
        height=560,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    figure.update_xaxes(title_text="Forecast Horizon", showgrid=False)
    figure.update_yaxes(title_text="Projected Close", gridcolor=CHART_GRID_COLOR)
    return apply_chart_theme(figure)


def render_metric_cards(df: pd.DataFrame, evaluation_context: dict[str, Any] | None, forecast_days: int) -> None:
    cards = st.columns(4)
    last_close = float(df["close"].iloc[-1]) if not df.empty and "close" in df.columns else 0.0
    train_points = len(evaluation_context["train_series"]) if evaluation_context else 0
    test_points = len(evaluation_context["test_series"]) if evaluation_context else 0
    latest_date = pd.to_datetime(df["date"].iloc[-1]).date().isoformat() if "date" in df.columns and not df.empty else "n/a"

    card_specs = [
        ("Latest Close", f"{last_close:,.4f}", f"Most recent row: {latest_date}"),
        ("Rows Loaded", f"{len(df):,}", "Prepared dataset after feature engineering"),
        ("Backtest Split", f"{train_points:,}/{test_points:,}", "Train points / test points"),
        ("Forecast Horizon", f"{forecast_days} day(s)", "Forward projection control"),
    ]

    for column, (label, value, note) in zip(cards, card_specs):
        column.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_spotlight_cards(future_results: dict[str, dict[str, Any]], backtest_results: dict[str, dict[str, Any]]) -> None:
    successful_future = [result for result in future_results.values() if result.get("status") == "ready"]
    successful_backtest = [result for result in backtest_results.values() if result.get("status") == "ready"]
    if not successful_future and not successful_backtest:
        return

    best_future = max(successful_future, key=lambda result: result["change_pct"]) if successful_future else None
    most_defensive = min(successful_future, key=lambda result: result["change_pct"]) if successful_future else None
    best_backtest = min(successful_backtest, key=lambda result: result["rmse"]) if successful_backtest else None

    columns = st.columns(3)
    spotlight_specs = [
        (
            "Strongest Upside",
            best_future["model"] if best_future else "Unavailable",
            f"{best_future['change_pct']:+.2f}%" if best_future else "n/a",
            f"Projected final close: {best_future['final_value']:.4f}" if best_future else "No successful forecast available.",
        ),
        (
            "Best Backtest",
            best_backtest["model"] if best_backtest else "Unavailable",
            f"RMSE {best_backtest['rmse']:.4f}" if best_backtest else "n/a",
            f"MAE {best_backtest['mae']:.4f} | MAPE {best_backtest['mape']:.2f}%" if best_backtest else "No successful backtest available.",
        ),
        (
            "Most Defensive",
            most_defensive["model"] if most_defensive else "Unavailable",
            f"{most_defensive['change_pct']:+.2f}%" if most_defensive else "n/a",
            f"Projected final close: {most_defensive['final_value']:.4f}" if most_defensive else "No successful forecast available.",
        ),
    ]

    for column, (label, model_name, value, note) in zip(columns, spotlight_specs):
        column.markdown(
            f"""
            <div class="spotlight-card">
                <div class="spotlight-model">{label}</div>
                <div class="spotlight-value">{model_name}</div>
                <div class="spotlight-copy"><strong>{value}</strong></div>
                <div class="spotlight-copy">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_hero(selected_crypto: str, selected_models: list[str], forecast_days: int) -> None:
    model_tags = "".join(f'<span class="hero-chip">{MODEL_LABELS[model_name]}</span>' for model_name in selected_models[:5])
    if len(selected_models) > 5:
        model_tags += f'<span class="hero-chip">+{len(selected_models) - 5} more</span>'
    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-kicker">Forecast Studio</div>
            <h1 class="hero-title"><span class="btc-icon">M</span>MarketCast Lab</h1>
            <p class="hero-copy">
                Compare saved model performance on the latest backtest split, then project the next {forecast_days} days
                for <strong>{selected_crypto}</strong>.
            </p>
            <div class="hero-chip-row">{model_tags}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics_table(results: dict[str, dict[str, Any]]) -> None:
    rows = []
    for model_name in MODEL_OPTIONS:
        result = results.get(model_name)
        if not result:
            continue
        if result.get("status") != "ready":
            rows.append({"Model": result["model"], "Status": result.get("status", "skipped"), "Message": result.get("message", "")})
            continue
        rows.append(
            {
                "Model": result["model"],
                "Status": "ready",
                "RMSE": round(result["rmse"], 4),
                "MAE": round(result["mae"], 4),
                "MAPE %": round(result["mape"], 2),
            }
        )

    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_forward_table(results: dict[str, dict[str, Any]]) -> None:
    rows = []
    for model_name in MODEL_OPTIONS:
        result = results.get(model_name)
        if not result:
            continue
        if result.get("status") != "ready":
            rows.append({"Model": result["model"], "Status": result.get("status", "skipped"), "Message": result.get("message", "")})
            continue
        rows.append(
            {
                "Model": result["model"],
                "Status": "ready",
                "Projected Close": round(result["final_value"], 4),
                "Change %": round(result["change_pct"], 2),
                "Horizon Points": len(result["predictions"]),
            }
        )

    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def main() -> None:
    st.set_page_config(page_title="MarketCast Lab", layout="wide")
    apply_app_styles()

    df = load_prepared_frame()

    if df.empty:
        st.error("No data loaded.")
        return

    crypto_options = crypto_selector_options(df)
    if not crypto_options:
        st.error("No crypto options found in the dataset.")
        return

    default_crypto = "BTC/USD" if any(code == "BTC/USD" for code, _ in crypto_options) else crypto_options[0][0]

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-shell">
                <div class="sidebar-eyebrow">Control Room</div>
                <p class="sidebar-copy">Pick asset, models, and horizon, then run the full comparison view.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        selected_crypto = st.selectbox(
            "Crypto",
            options=[code for code, _ in crypto_options],
            index=next((i for i, (code, _) in enumerate(crypto_options) if code == default_crypto), 0),
            format_func=lambda crypto_code: next(label for code, label in crypto_options if code == crypto_code),
        )
        selected_models = st.multiselect(
            "Models",
            options=MODEL_OPTIONS,
            default=["lstm", "gru", "arima", "prophet", "ensemble"],
            format_func=lambda model_name: MODEL_LABELS[model_name],
        )
        selected_optimizer = st.selectbox(
            "Optimizer",
            options=OPTIMIZER_OPTIONS,
            index=0,
            help="Sequence models are loaded from results/<optimizer>/",
        )
        forecast_days = st.slider("Days ahead", min_value=1, max_value=MAX_FORECAST_DAYS, value=DEFAULT_FORECAST_DAYS)
        moving_average_windows = st.multiselect(
            "Moving averages",
            options=[20, 50, 200],
            default=[20, 50],
            help="Overlay moving average lines on the price history chart.",
        )
        show_volume_overlay = st.checkbox("Show volume overlay", value=False)
        st.caption(f"Lookback window: {DEFAULT_LOOKBACK} points")
        st.caption(f"Backtest split: {int(TEST_RATIO * 100)}%")
        submit = st.button("Run Forecast Studio", type="primary")

    df = filter_frame_by_crypto(df, selected_crypto)
    evaluation_context = prepare_split(df)
    forward_context = prepare_forward_context(df)
    render_hero(selected_crypto, selected_models, forecast_days)

    if evaluation_context is None or forward_context is None:
        st.error("Not enough data to build the prediction views.")
        return

    if not submit:
        st.info("Choose one or more models, set the forecast horizon, then click Run Forecast Studio.")
        st.plotly_chart(
            build_price_history_figure(df, selected_crypto, moving_average_windows, show_volume_overlay),
            width="stretch",
        )
        return

    if not selected_models:
        st.warning("Select at least one model.")
        return

    with st.spinner("Loading saved models, scoring the backtest split, and building the forward forecast..."):
        backtest_results = run_predictions(selected_models, evaluation_context, selected_optimizer)
        future_results = run_future_forecasts(selected_models, forward_context, forecast_days, selected_optimizer)

    skipped_messages = [
        f"{result['model']}: {result.get('message', 'Skipped')}"
        for result in list(backtest_results.values()) + list(future_results.values())
        if result.get("status") == "skipped"
    ]
    if skipped_messages:
        with st.expander("Skipped models", expanded=False):
            for message in skipped_messages:
                st.warning(message)

    render_spotlight_cards(future_results, backtest_results)
    tabs = st.tabs(["Forward Outlook", "Backtest", "Metrics"])

    with tabs[0]:
        st.subheader(f"{forecast_days}-Day Forward Outlook")
        st.plotly_chart(build_future_figure(future_results, forward_context), width="stretch")
        render_forward_table(future_results)

    with tabs[1]:
        st.subheader("Historical Backtest Comparison")
        st.plotly_chart(build_comparison_figure(backtest_results, evaluation_context["test_dates"]), width="stretch")

    with tabs[2]:
        st.subheader("Model Metrics")
        render_metrics_table(backtest_results)


if __name__ == "__main__":
    if not streamlit_runtime_exists():
        raise SystemExit("Unsupported launcher. Start the UI with: streamlit run src/streamlit.py")
    main()
