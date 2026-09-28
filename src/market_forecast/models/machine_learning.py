from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from .base import ForecastModel


def lag_matrix(values: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=float).reshape(-1)
    if len(array) <= lookback:
        raise ValueError("Training history must be longer than lookback")
    features = np.asarray([array[index - lookback:index] for index in range(lookback, len(array))])
    return features, array[lookback:]


class LagRegressor(ForecastModel):
    def __init__(self, *, kind: str, lookback: int = 20, seed: int = 42, **params: object) -> None:
        self.kind, self.lookback, self.seed, self.params = kind, lookback, seed, params

    def _estimator(self):
        columns = list(range(self.lookback))
        if self.kind in {"ridge", "lasso"}:
            preprocessing = ColumnTransformer([("lags", StandardScaler(), columns)])
            estimator = (Ridge(alpha=float(self.params.get("alpha", 1.0))) if self.kind == "ridge"
                         else Lasso(alpha=float(self.params.get("alpha", 0.01)), max_iter=10000, random_state=self.seed))
            return Pipeline([("preprocess", preprocessing), ("model", estimator)])
        if self.kind == "random_forest":
            estimator = RandomForestRegressor(
                n_estimators=int(self.params.get("n_estimators", 50)),
                max_depth=self.params.get("max_depth", 6), random_state=self.seed, n_jobs=1,
            )
            preprocessing = ColumnTransformer([("lags", "passthrough", columns)])
            return Pipeline([("preprocess", preprocessing), ("model", estimator)])
        if self.kind == "xgboost":
            try:
                from xgboost import XGBRegressor
            except ImportError as exc:
                raise ImportError("Install MarketCast Lab with the 'boosting' extra") from exc
            estimator = XGBRegressor(
                n_estimators=int(self.params.get("n_estimators", 50)), max_depth=int(self.params.get("max_depth", 3)),
                learning_rate=float(self.params.get("learning_rate", 0.05)), random_state=self.seed, n_jobs=1,
            )
            preprocessing = ColumnTransformer([("lags", "passthrough", columns)])
            return Pipeline([("preprocess", preprocessing), ("model", estimator)])
        raise ValueError(f"Unknown lag regressor: {self.kind}")

    def fit(self, values: np.ndarray) -> "LagRegressor":
        features, target = lag_matrix(values, self.lookback)
        self.history_ = np.asarray(values, dtype=float).reshape(-1).tolist()
        self.estimator_ = self._estimator()
        self.estimator_.fit(features, target)
        return self

    def predict(self, horizon: int, context: np.ndarray | None = None) -> np.ndarray:
        history = list(np.asarray(context, dtype=float).reshape(-1)) if context is not None else list(self.history_)
        predictions = []
        for _ in range(horizon):
            value = float(self.estimator_.predict(np.asarray(history[-self.lookback:]).reshape(1, -1))[0])
            predictions.append(value)
            history.append(value)
        return np.asarray(predictions)

    def get_params(self) -> dict[str, object]:
        return {"kind": self.kind, "lookback": self.lookback, "seed": self.seed, **self.params}

    def save(self, path: Path) -> None:
        path.write_bytes(pickle.dumps(self.estimator_))
