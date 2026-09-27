from pathlib import Path

import pandas as pd

from market_forecast.config import ExperimentConfig
from market_forecast.experiments import run_experiment


def test_repeated_baseline_runs_have_equivalent_splits_and_scores(tmp_path: Path) -> None:
    config = ExperimentConfig(
        artifacts_dir=tmp_path, models=("last_value", "drift"), horizons=(1, 5),
        folds=3, final_test_size=5, min_train_size=60, max_train_rows=100,
    )
    first, second = run_experiment(config), run_experiment(config)
    first_metrics = pd.read_csv(first / "metrics.csv").drop(columns=["fit_seconds", "inference_seconds"])
    second_metrics = pd.read_csv(second / "metrics.csv").drop(columns=["fit_seconds", "inference_seconds"])
    pd.testing.assert_frame_equal(first_metrics, second_metrics, check_exact=False, rtol=1e-12)
    assert (first / "data_manifest.json").read_text() == (second / "data_manifest.json").read_text()
