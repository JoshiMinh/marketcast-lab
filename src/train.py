import contextlib
import io
import json
import logging
import os
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA

from .data import engineer_features, load_data
from .models import (
    BASE_MODELS,
    MODEL_LABELS,
    SEQUENCE_MODELS,
    available_models,
    build_gru_model,
    build_lstm_model,
    build_prophet_model,
    normalize_model_selection,
    normalize_optimizer,
)
from .models import HAS_PROPHET
from market_forecast.data import load_bundled_crypto, select_asset

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

RESULTS_DIR = Path("results")


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(np.ravel(y_true), np.ravel(y_pred)))


def mape(y_true, y_pred):
    y_true = np.ravel(np.asarray(y_true, dtype=float))
    y_pred = np.ravel(np.asarray(y_pred, dtype=float))
    mask = y_true != 0
    if mask.sum() == 0:
        return 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def create_sequences(series, lookback):
    X, y = [], []
    for index in range(len(series) - lookback):
        X.append(series[index : index + lookback])
        y.append(series[index + lookback])
    return np.array(X), np.array(y)


def progress_line(current, total, label, width=20):
    filled = width if total == 0 else int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total} {label}"


@contextlib.contextmanager
def _silence_external_output():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield


def _prepare_time_split(close_series, dates, lookback, test_ratio=0.2):
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

    return {
        "train_series": train_series,
        "test_series": test_series,
        "train_dates": train_dates,
        "test_dates": test_dates,
    }


def _prepare_sequence_context(train_series, lookback):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_series.reshape(-1, 1))
    X_train, y_train = create_sequences(train_scaled, lookback=lookback)
    if len(X_train) == 0:
        return None

    return {
        "scaler": scaler,
        "X_train": X_train,
        "y_train": y_train,
        "history": train_series.tolist(),
    }


def _forecast_sequence_model(model, context, test_series):
    predictions = []
    history = list(context["history"])
    scaler = context["scaler"]
    lookback = context["X_train"].shape[1]

    for actual_value in test_series:
        window = np.asarray(history[-lookback:], dtype=float).reshape(-1, 1)
        window_scaled = scaler.transform(window).reshape(1, lookback, 1)
        with _silence_external_output():
            pred_scaled = model.predict(window_scaled, verbose=0)
        pred_value = scaler.inverse_transform(pred_scaled)[0, 0]
        predictions.append(pred_value)
        history.append(float(actual_value))

    return np.asarray(predictions)


def _artifact_path(results_dir, model_name):
    suffix = ".keras" if model_name in SEQUENCE_MODELS else ".json"
    return results_dir / f"{model_name}{suffix}"


def _legacy_sequence_artifact_path(results_dir, model_name):
    if model_name not in SEQUENCE_MODELS:
        return None
    return results_dir / f"{model_name}_model.keras"


def _serialize_result(result):
    payload = {
        "model_key": result["model_key"],
        "model": result["model"],
        "status": result.get("status", "ready"),
        "optimizer": result.get("optimizer"),
    }
    for key in ("rmse", "mae", "mape"):
        if key in result:
            payload[key] = float(result[key])
    for key in ("predictions", "actual"):
        if key in result and result[key] is not None:
            payload[key] = np.asarray(result[key], dtype=float).tolist()
    if result.get("message"):
        payload["message"] = result["message"]
    return payload


def _write_result_artifact(results_dir, result):
    if result["model_key"] in SEQUENCE_MODELS:
        legacy_path = _legacy_sequence_artifact_path(results_dir, result["model_key"])
        if legacy_path is not None and legacy_path.exists():
            legacy_path.unlink()
        return

    artifact_path = _artifact_path(results_dir, result["model_key"])
    artifact_path.write_text(json.dumps(_serialize_result(result), indent=2), encoding="utf-8")


def _train_sequence_model(model_name, builder, context, test_series, results_dir, optimizer):
    model = builder(context["X_train"].shape[1], optimizer=optimizer)
    with _silence_external_output():
        model.fit(
            context["X_train"],
            context["y_train"],
            epochs=5,
            batch_size=128,
            verbose=0,
            validation_split=0.1,
        )

    predictions = _forecast_sequence_model(model, context, test_series)
    metrics = {
        "model_key": model_name,
        "model": MODEL_LABELS[model_name],
        "optimizer": optimizer,
        "predictions": predictions,
        "actual": np.asarray(test_series, dtype=float),
        "rmse": rmse(test_series, predictions),
        "mae": mean_absolute_error(test_series, predictions),
        "mape": mape(test_series, predictions),
    }
    artifact_path = _artifact_path(results_dir, model_name)
    model.save(artifact_path)
    legacy_path = _legacy_sequence_artifact_path(results_dir, model_name)
    if legacy_path is not None and legacy_path.exists():
        legacy_path.unlink()
    return metrics


def _train_arima(train_series, test_series):
    with _silence_external_output():
        fitted = ARIMA(train_series, order=(5, 1, 0)).fit()
        forecast = fitted.forecast(steps=len(test_series))

    predictions = np.asarray(forecast)
    return {
        "model_key": "arima",
        "model": "ARIMA",
        "optimizer": None,
        "predictions": predictions,
        "actual": np.asarray(test_series, dtype=float),
        "rmse": rmse(test_series, predictions),
        "mae": mean_absolute_error(test_series, predictions),
        "mape": mape(test_series, predictions),
    }


def _train_prophet(train_series, test_series, train_dates, test_dates):
    if not HAS_PROPHET:
        return None

    train_df = pd.DataFrame({"ds": pd.to_datetime(train_dates), "y": train_series.astype(float)})
    forecast_frame = pd.DataFrame({"ds": pd.to_datetime(test_dates)})

    with _silence_external_output():
        model = build_prophet_model()
        model.fit(train_df)
        forecast = model.predict(forecast_frame)

    predictions = forecast["yhat"].to_numpy()
    return {
        "model_key": "prophet",
        "model": "Prophet",
        "optimizer": None,
        "predictions": predictions,
        "actual": np.asarray(test_series, dtype=float),
        "rmse": rmse(test_series, predictions),
        "mae": mean_absolute_error(test_series, predictions),
        "mape": mape(test_series, predictions),
    }


def _train_ensemble(component_results):
    available_results = [result for result in component_results if result and result.get("predictions") is not None]
    if len(available_results) < 2:
        return None

    min_length = min(len(result["predictions"]) for result in available_results)
    stacked_predictions = np.column_stack([result["predictions"][-min_length:] for result in available_results])
    actual = np.asarray(available_results[0]["actual"][-min_length:], dtype=float)
    predictions = stacked_predictions.mean(axis=1)

    return {
        "model_key": "ensemble",
        "model": "Ensemble",
        "optimizer": None,
        "predictions": predictions,
        "actual": actual,
        "rmse": rmse(actual, predictions),
        "mae": mean_absolute_error(actual, predictions),
        "mape": mape(actual, predictions),
    }


def load_and_prepare_data(asset_id="crypto:BTC-USD"):
    print("Loading data...")
    canonical = load_bundled_crypto()
    df = select_asset(canonical, asset_id).rename(columns={"timestamp": "date"})
    df["crypto"] = df["symbol"]
    if df.empty:
        print("No data loaded.")
        return None

    print("Engineering features...")
    return engineer_features(df)


def run_pipeline(selected_models=None, optimizer="adam"):
    try:
        df = load_and_prepare_data()
        if df is None:
            return []

        results = train_models(df, selected_models=selected_models, optimizer=optimizer)
        print("Done.")
        return results
    except KeyboardInterrupt:
        print("\nTraining interrupted. Terminating...")
        raise SystemExit(0)


def train_models(df, selected_models=None, max_points=5000, lookback=30, optimizer="adam"):
    if df.empty:
        print("No data loaded.")
        return []

    if "close" not in df.columns:
        print("Missing close column.")
        return []

    asset_column = "asset_id" if "asset_id" in df.columns else "crypto" if "crypto" in df.columns else None
    if asset_column is not None and df[asset_column].nunique() != 1:
        raise ValueError("Legacy training requires one asset selected before feature engineering and splitting")

    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
        dates = pd.to_datetime(df["date"]).to_numpy()
    else:
        df = df.reset_index(drop=True)
        dates = None

    df_subset = df.iloc[-max_points:] if len(df) > max_points else df
    close_series = df_subset["close"].astype(float).to_numpy()
    dates = dates[-len(df_subset) :] if dates is not None else None

    selected_models = normalize_model_selection(selected_models) or available_models(include_ensemble=True)
    results = []
    result_map = {}

    split = _prepare_time_split(close_series, dates, lookback)
    if split is None:
        print("Not enough data to train the selected models.")
        return []

    sequence_context = None
    if any(model_name in {"lstm", "gru"} for model_name in selected_models):
        sequence_context = _prepare_sequence_context(split["train_series"], lookback)
        if sequence_context is None:
            print("Not enough data for LSTM/GRU.")

    optimizer_name = normalize_optimizer(optimizer)
    print(f"Training {len(selected_models)} model(s) with optimizer: {optimizer_name}")
    print(progress_line(0, len(selected_models), "starting"))

    # Store each training run under its optimizer namespace.
    results_dir = RESULTS_DIR / optimizer_name
    results_dir.mkdir(parents=True, exist_ok=True)

    for index, model_name in enumerate(selected_models, start=1):
        result = None

        if model_name == "lstm" and sequence_context is not None:
            result = _train_sequence_model(
                "lstm",
                build_lstm_model,
                sequence_context,
                split["test_series"],
                results_dir,
                optimizer_name,
            )
        elif model_name == "gru" and sequence_context is not None:
            result = _train_sequence_model(
                "gru",
                build_gru_model,
                sequence_context,
                split["test_series"],
                results_dir,
                optimizer_name,
            )
        elif model_name == "arima":
            result = _train_arima(split["train_series"], split["test_series"])
        elif model_name == "prophet":
            result = _train_prophet(split["train_series"], split["test_series"], split["train_dates"], split["test_dates"])
            if result is None and not HAS_PROPHET:
                print("Prophet is not installed; skipped.")
        elif model_name == "ensemble":
            component_results = [result_map.get(base_model) for base_model in BASE_MODELS]
            result = _train_ensemble(component_results)
        else:
            result = None

        if result is None:
            skipped = {
                "model_key": model_name,
                "model": MODEL_LABELS[model_name],
                "optimizer": optimizer_name if model_name in SEQUENCE_MODELS else None,
                "status": "skipped",
                "message": "Model artifact could not be generated.",
            }
            results.append(skipped)
            _write_result_artifact(results_dir, skipped)
            print(progress_line(index, len(selected_models), f"{MODEL_LABELS[model_name]} skipped"))
            continue

        result_map[model_name] = result
        results.append(result)
        _write_result_artifact(results_dir, result)
        print(
            progress_line(
                index,
                len(selected_models),
                f"{result['model']} | RMSE {result['rmse']:.4f} MAE {result['mae']:.4f} MAPE {result['mape']:.2f}%",
            )
        )

    return results
