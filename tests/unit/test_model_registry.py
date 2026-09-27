import numpy as np
import pytest

from market_forecast.models import ForecastModel, default_registry


def test_registry_creates_every_required_model_adapter() -> None:
    registry = default_registry()
    expected = {
        "last_value", "drift", "seasonal_naive", "exponential_smoothing", "holt_winters",
        "arima", "sarima", "ridge", "random_forest", "xgboost", "rnn", "lstm", "gru",
    }
    assert expected == set(registry.names())
    assert isinstance(registry.create("ridge", {"lookback": 2}), ForecastModel)
    with pytest.raises(KeyError):
        registry.create("unknown")


def test_recursive_ridge_forecast_has_requested_length() -> None:
    model = default_registry().create("ridge", {"lookback": 3})
    predictions = model.fit(np.arange(20, dtype=float)).predict(5)
    assert predictions.shape == (5,)
    assert np.isfinite(predictions).all()

