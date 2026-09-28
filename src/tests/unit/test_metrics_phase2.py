import numpy as np

from market_forecast.evaluation import regression_metrics


def test_all_metrics_have_known_values_and_mape_handles_zero() -> None:
    result = regression_metrics(
        np.array([0.0, 2.0, 4.0]), np.array([1.0, 2.0, 2.0]),
        training=np.array([1.0, 2.0, 3.0]), seasonal_period=1,
    )
    assert result["mae"] == 1.0
    assert result["mse"] == 5 / 3
    assert np.isclose(result["rmse"], np.sqrt(5 / 3))
    assert result["mape"] == 25.0
    assert result["mase"] == 1.0
    assert result["mean_error"] == 1 / 3

