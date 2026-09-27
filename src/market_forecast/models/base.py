from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class ForecastModel(ABC):
    supports_intervals = False

    @abstractmethod
    def fit(self, values: np.ndarray) -> "ForecastModel": ...

    @abstractmethod
    def predict(self, horizon: int, context: np.ndarray | None = None) -> np.ndarray: ...

    @abstractmethod
    def get_params(self) -> dict[str, object]: ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    def diagnostics(self) -> dict[str, object]:
        return {}
