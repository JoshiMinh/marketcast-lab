from __future__ import annotations

from collections.abc import Callable

from .base import ForecastModel
from .baselines import DriftBaseline, LastValueBaseline, SeasonalNaiveBaseline
from .deep_learning import TorchSequenceModel
from .machine_learning import LagRegressor
from .statistical import StatisticalModel

ModelFactory = Callable[[dict[str, object]], ForecastModel]


class ModelRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, ModelFactory] = {}

    def register(self, name: str, factory: ModelFactory) -> None:
        if name in self._factories:
            raise ValueError(f"Model already registered: {name}")
        self._factories[name] = factory

    def create(self, name: str, params: dict[str, object] | None = None) -> ForecastModel:
        if name not in self._factories:
            raise KeyError(f"Unknown model: {name}")
        return self._factories[name](params or {})

    def names(self) -> tuple[str, ...]:
        return tuple(self._factories)


def default_registry() -> ModelRegistry:
    registry = ModelRegistry()
    registry.register("last_value", lambda _: LastValueBaseline())
    registry.register("drift", lambda _: DriftBaseline())
    registry.register("seasonal_naive", lambda p: SeasonalNaiveBaseline(int(p.get("period", 7))))
    for name in ("exponential_smoothing", "holt_winters", "arima", "sarima"):
        registry.register(name, lambda p, kind=name: StatisticalModel(kind=kind, **p))
    for name in ("ridge", "random_forest", "xgboost"):
        registry.register(name, lambda p, kind=name: LagRegressor(kind=kind, **p))
    for name in ("rnn", "lstm", "gru"):
        registry.register(name, lambda p, kind=name: TorchSequenceModel(kind=kind, **p))
    return registry
