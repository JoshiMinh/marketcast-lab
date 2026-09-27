from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class TrainOnlyStandardScaler:
    """A scaler with auditable training-boundary metadata."""

    def __init__(self) -> None:
        self._scaler = StandardScaler()
        self.fit_row_count_: int | None = None
        self.fit_end_timestamp_: pd.Timestamp | None = None

    def fit(self, values: np.ndarray, *, timestamps: pd.Series | None = None) -> "TrainOnlyStandardScaler":
        array = np.asarray(values, dtype=float)
        self._scaler.fit(array)
        self.fit_row_count_ = len(array)
        if timestamps is not None:
            parsed = pd.to_datetime(timestamps, utc=True)
            self.fit_end_timestamp_ = parsed.max()
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.fit_row_count_ is None:
            raise RuntimeError("Scaler must be fit on training data before transform")
        return self._scaler.transform(np.asarray(values, dtype=float))

    def fit_transform(self, values: np.ndarray, *, timestamps: pd.Series | None = None) -> np.ndarray:
        return self.fit(values, timestamps=timestamps).transform(values)

    @property
    def mean_(self) -> np.ndarray:
        return self._scaler.mean_

