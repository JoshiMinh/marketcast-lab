from __future__ import annotations

from pathlib import Path
import importlib
import sys

import pandas as pd
import pytest


def _app():
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
    return AppTest.from_file(str(script), default_timeout=30).run()


def _has_heading(app, title: str) -> bool:
    return any(f"<h1>{title}</h1>" in block.value and "<svg" in block.value for block in app.markdown)


def test_dashboard_views_and_saved_metrics() -> None:
    app = _app()
    assert not app.exception
    assert app.sidebar.radio[0].options == ["Overview", "Forecasts", "Data & Runs"]
    assert _has_heading(app, "Model evidence")
    assert len(app.get("vega_lite_chart")) == 1

    root = Path(__file__).resolve().parents[3] / "assets" / "phase3"
    saved = pd.read_csv(root / "recommendations.csv")
    asset = app.sidebar.selectbox[0].value
    horizon = app.sidebar.selectbox[1].value
    selected = saved[(saved.asset_id == asset) & (saved.horizon == horizon)].iloc[0]
    assert app.metric[0].value == selected.model.replace("_", " ").title().replace("Arima", "ARIMA")
    assert app.metric[1].value == f"{selected.rmse_mean:,.4g}"

    app.sidebar.selectbox[0].set_value("equity:SPY").run()
    app.sidebar.selectbox[1].set_value(20).run()
    selected = saved[(saved.asset_id == "equity:SPY") & (saved.horizon == 20)].iloc[0]
    assert app.metric[1].value == f"{selected.rmse_mean:,.4g}"

    app.sidebar.radio[0].set_value("Forecasts").run()
    assert not app.exception
    assert _has_heading(app, "Forecasts")
    forecast_view = next(widget for widget in app.radio if widget.label == "Forecast view")
    assert forecast_view.options == ["Historical backtest", "Saved future scenario"]
    assert len(app.get("vega_lite_chart")) == 1
    forecast_view.set_value("Saved future scenario").run()
    assert not app.exception
    assert any("Speculative scenario" in caption.value for caption in app.caption)
    assert len(app.get("vega_lite_chart")) == 1

    app.sidebar.radio[0].set_value("Data & Runs").run()
    assert not app.exception
    assert _has_heading(app, "Data & runs")


def test_comparison_survives_missing_run_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parents[3] / "assets" / "phase3"
    for name in ("comparison.csv", "recommendations.csv"):
        (tmp_path / name).write_bytes((root / name).read_bytes())
    monkeypatch.setenv("MARKETCAST_ARTIFACT_ROOT", str(tmp_path))
    app = _app()
    assert not app.exception
    assert _has_heading(app, "Model evidence")
    app.sidebar.radio[0].set_value("Forecasts").run()
    assert not app.exception
    assert any("full saved run files" in info.value for info in app.info)
    next(widget for widget in app.radio if widget.label == "Forecast view").set_value("Saved future scenario").run()
    assert not app.exception
    assert len(app.get("vega_lite_chart")) == 1
    app.sidebar.radio[0].set_value("Data & Runs").run()
    assert not app.exception
    assert any("Full run files are unavailable" in info.value for info in app.info)


def test_missing_future_scenario_is_clear(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARKETCAST_PHASE4_ROOT", str(tmp_path))
    app = _app()
    app.sidebar.radio[0].set_value("Forecasts").run()
    next(widget for widget in app.radio if widget.label == "Forecast view").set_value("Saved future scenario").run()
    assert not app.exception
    assert any("No saved future scenario" in info.value for info in app.info)
