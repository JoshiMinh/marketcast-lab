from pathlib import Path

import numpy as np

from market_forecast.models import default_registry


def test_statistical_ml_and_cpu_torch_smoke(tmp_path: Path) -> None:
    values = 100 + np.sin(np.arange(80) / 4) + np.arange(80) * 0.1
    cases = {
        "exponential_smoothing": {},
        "ridge": {"lookback": 8},
        "rnn": {"lookback": 8, "hidden_size": 4, "epochs": 1, "batch_size": 16},
        "lstm": {"lookback": 8, "hidden_size": 4, "epochs": 1, "batch_size": 16},
        "gru": {"lookback": 8, "hidden_size": 4, "epochs": 1, "batch_size": 16},
    }
    registry = default_registry()
    for name, params in cases.items():
        model = registry.create(name, params).fit(values)
        predictions = model.predict(5)
        assert predictions.shape == (5,)
        assert np.isfinite(predictions).all()
        suffix = ".pt" if name in {"rnn", "lstm", "gru"} else ".bin"
        model.save(tmp_path / f"{name}{suffix}")
        assert (tmp_path / f"{name}{suffix}").stat().st_size > 0

