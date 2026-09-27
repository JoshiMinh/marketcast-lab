from __future__ import annotations

import numpy as np
import json
from pathlib import Path

from .base import ForecastModel


def _training_values(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("Training values must be non-empty and finite")
    return array


class _Baseline(ForecastModel):
    def get_params(self) -> dict[str, object]:
        return {}

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.get_params(), indent=2), encoding="utf-8")


class LastValueBaseline(_Baseline):
    def fit(self, values: np.ndarray) -> "LastValueBaseline":
        self.last_ = float(_training_values(values)[-1])
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(_valid_horizon(horizon), self.last_)


class DriftBaseline(_Baseline):
    def fit(self, values: np.ndarray) -> "DriftBaseline":
        train = _training_values(values)
        if len(train) < 2:
            raise ValueError("Drift requires at least two training observations")
        self.last_ = float(train[-1])
        self.slope_ = float((train[-1] - train[0]) / (len(train) - 1))
        return self

    def predict(self, horizon: int) -> np.ndarray:
        steps = np.arange(1, _valid_horizon(horizon) + 1)
        return self.last_ + self.slope_ * steps


class SeasonalNaiveBaseline(_Baseline):
    def __init__(self, period: int) -> None:
        if period < 2:
            raise ValueError("A defensible seasonal period must be at least 2")
        self.period = period

    def fit(self, values: np.ndarray) -> "SeasonalNaiveBaseline":
        train = _training_values(values)
        if len(train) < self.period:
            raise ValueError("Training data is shorter than the seasonal period")
        self.season_ = train[-self.period :].copy()
        return self

    def predict(self, horizon: int) -> np.ndarray:
        length = _valid_horizon(horizon)
        return np.resize(self.season_, length)

    def get_params(self) -> dict[str, object]:
        return {"period": self.period}


def _valid_horizon(horizon: int) -> int:
    if horizon < 1:
        raise ValueError("horizon must be positive")
    return int(horizon)

