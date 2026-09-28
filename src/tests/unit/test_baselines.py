import numpy as np

from market_forecast.models import DriftBaseline, LastValueBaseline, SeasonalNaiveBaseline


def test_baseline_predictions() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(LastValueBaseline().fit(values).predict(2), [4.0, 4.0])
    np.testing.assert_array_equal(DriftBaseline().fit(values).predict(2), [5.0, 6.0])
    np.testing.assert_array_equal(
        SeasonalNaiveBaseline(2).fit(values).predict(3), [3.0, 4.0, 3.0]
    )

