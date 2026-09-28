import json
from pathlib import Path

import pandas as pd

from market_forecast.config import BaselineExperimentConfig
from market_forecast.experiments import run_btc_baseline_experiment


def test_bundled_btc_baseline_writes_traceable_artifacts(tmp_path: Path) -> None:
    output = run_btc_baseline_experiment(
        BaselineExperimentConfig(output_dir=tmp_path / "run")
    )
    expected = {
        "config.json",
        "data_manifest.json",
        "environment.json",
        "metrics.csv",
        "fold_predictions.csv",
    }
    assert expected == {path.name for path in output.iterdir()}
    manifest = json.loads((output / "data_manifest.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(output / "fold_predictions.csv")
    metrics = pd.read_csv(output / "metrics.csv")
    assert manifest["asset_id"] == "crypto:BTC-USD"
    assert manifest["final_test_locked"] is True
    assert set(predictions["asset_id"]) == {"crypto:BTC-USD"}
    assert set(metrics["model"]) == {"last_value", "drift", "seasonal_naive"}

