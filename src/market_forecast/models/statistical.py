from __future__ import annotations

from pathlib import Path

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .base import ForecastModel


class StatisticalModel(ForecastModel):
    def __init__(
        self,
        *,
        kind: str,
        order: tuple[int, int, int] = (1, 1, 0),
        seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
        seasonal_period: int = 7,
    ) -> None:
        self.kind = kind
        self.order = order
        self.seasonal_order = seasonal_order
        self.seasonal_period = seasonal_period

    def fit(self, values: np.ndarray) -> "StatisticalModel":
        train = np.asarray(values, dtype=float).reshape(-1)
        if self.kind == "exponential_smoothing":
            self.result_ = ExponentialSmoothing(train, trend="add", initialization_method="estimated").fit()
        elif self.kind == "holt_winters":
            self.result_ = ExponentialSmoothing(
                train, trend="add", seasonal="add", seasonal_periods=self.seasonal_period,
                initialization_method="estimated",
            ).fit()
        elif self.kind in {"arima", "sarima"}:
            seasonal = self.seasonal_order if self.kind == "sarima" else (0, 0, 0, 0)
            self.result_ = SARIMAX(
                train, order=self.order, seasonal_order=seasonal,
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False)
        else:
            raise ValueError(f"Unknown statistical model: {self.kind}")
        return self

    def predict(self, horizon: int, context: np.ndarray | None = None) -> np.ndarray:
        return np.asarray(self.result_.forecast(horizon), dtype=float)

    def get_params(self) -> dict[str, object]:
        return {
            "kind": self.kind, "order": self.order,
            "seasonal_order": self.seasonal_order, "seasonal_period": self.seasonal_period,
        }

    def diagnostics(self) -> dict[str, object]:
        return {"aic": float(getattr(self.result_, "aic", np.nan)), "bic": float(getattr(self.result_, "bic", np.nan))}

    def predict_interval(self, horizon: int, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray] | None:
        if self.kind not in {"arima", "sarima"}:
            return None
        intervals = np.asarray(self.result_.get_forecast(horizon).conf_int(alpha=alpha), dtype=float)
        return intervals[:, 0], intervals[:, 1]

    def save(self, path: Path) -> None:
        self.result_.save(path)
