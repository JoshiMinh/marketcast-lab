from pathlib import Path

import pandas as pd

import pytest

from market_forecast.experiments.artifacts import create_run_directory, staged_run_directory, write_predictions


def test_artifact_directories_are_unique_and_predictions_round_trip(tmp_path: Path) -> None:
    first = create_run_directory(tmp_path, {"seed": 42})
    second = create_run_directory(tmp_path, {"seed": 42})
    assert first != second
    source = pd.DataFrame({"timestamp": ["2024-01-01"], "prediction": [1.5]})
    filename, _ = write_predictions(first, source)
    loaded = pd.read_parquet(first / filename) if filename.endswith("parquet") else pd.read_csv(first / filename)
    assert loaded["prediction"].tolist() == [1.5]


def test_failed_run_is_never_published(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="save failed"):
        with staged_run_directory(tmp_path, {"seed": 42}) as (staging, published):
            (staging / "metrics.csv").write_text("partial", encoding="utf-8")
            assert not published.exists()
            raise RuntimeError("save failed")
    assert list(tmp_path.iterdir()) == []
