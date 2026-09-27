from __future__ import annotations

import numpy as np


def regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    training: np.ndarray | None = None,
    seasonal_period: int = 1,
) -> dict[str, float]:
    truth = np.asarray(actual, dtype=float).reshape(-1)
    forecast = np.asarray(predicted, dtype=float).reshape(-1)
    if len(truth) != len(forecast) or len(truth) == 0:
        raise ValueError("Actual and predicted values must have the same non-zero length")
    errors = truth - forecast
    absolute = np.abs(errors)
    denominator = np.abs(truth) + np.abs(forecast)
    smape_terms = np.divide(2 * absolute, denominator, out=np.zeros_like(absolute), where=denominator > 0)
    nonzero = np.abs(truth) > np.finfo(float).eps
    mape = np.mean(absolute[nonzero] / np.abs(truth[nonzero])) * 100 if nonzero.any() else float("nan")
    mase = float("nan")
    if training is not None:
        history = np.asarray(training, dtype=float).reshape(-1)
        if len(history) > seasonal_period:
            scale = np.mean(np.abs(history[seasonal_period:] - history[:-seasonal_period]))
            if scale > 0:
                mase = float(np.mean(absolute) / scale)
    return {
        "mae": float(np.mean(np.abs(errors))),
        "mse": float(np.mean(np.square(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mape": float(mape),
        "smape": float(np.mean(smape_terms) * 100),
        "mase": mase,
        "mean_error": float(np.mean(errors)),
    }

