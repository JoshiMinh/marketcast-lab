from __future__ import annotations

from pathlib import Path
import importlib
import sys

import pytest


def test_dashboard_pages_render_from_saved_artifacts() -> None:
    source_root = str(Path(__file__).resolve().parents[2])
    sys.path = [entry for entry in sys.path if entry != source_root]
    try:
        importlib.import_module("streamlit")
    except ImportError:
        pytest.skip("Streamlit dashboard extra is not installed")
    finally:
        sys.path.insert(0, source_root)
    from streamlit.testing.v1 import AppTest

    script = Path(__file__).resolve().parents[2] / "streamlit.py"
    app = AppTest.from_file(str(script), default_timeout=30).run()
    assert not app.exception
    index = script.parents[1] / "assets" / "phase3" / "run_index.json"
    from market_forecast.dashboard.artifacts import load_catalog

    try:
        load_catalog(index.parent)
    except (FileNotFoundError, ValueError):
        assert any(title.value == "Experiment artifacts needed" for title in app.title)
        return
    for page in ("Data Explorer", "Time-Series Analysis", "Experiment Setup", "Backtest Results",
                 "Model Comparison", "Future Forecast", "Experiment History"):
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page
        if page == "Model Comparison":
            import pandas as pd

            shown = app.dataframe[0].value
            saved = pd.read_csv(index.parent / "comparison.csv")
            first = shown.iloc[0]
            source = saved[(saved.asset_id == first.asset_id) & (saved.horizon == first.horizon) &
                           (saved.model == first.model)].iloc[0]
            assert first.rmse_mean == source.rmse_mean
            assert first.final_test_rmse == source.final_test_rmse
