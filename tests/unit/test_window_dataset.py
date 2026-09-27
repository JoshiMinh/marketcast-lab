import numpy as np

from market_forecast.models import lag_matrix


def test_lag_matrix_alignment_uses_only_prior_values() -> None:
    features, target = lag_matrix(np.arange(6, dtype=float), lookback=3)
    np.testing.assert_array_equal(features, [[0, 1, 2], [1, 2, 3], [2, 3, 4]])
    np.testing.assert_array_equal(target, [3, 4, 5])

