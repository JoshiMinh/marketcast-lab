"""Read-only dashboard for saved MarketCast Lab experiments."""
from __future__ import annotations

import json
from urllib.parse import quote
import os
from pathlib import Path
import sys

import altair as alt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) in sys.path:
    sys.path.remove(str(SOURCE_ROOT))
    sys.path.append(str(SOURCE_ROOT))

import streamlit as st

from market_forecast.dashboard.artifacts import ArtifactCatalog, RunRecord, load_catalog

st.set_page_config(page_title="MarketCast Lab", page_icon=str(PROJECT_ROOT / ".streamlit" / "favicon.svg"),
                   layout="wide", initial_sidebar_state="auto")

PAGES = ("Overview", "Forecasts", "Data & Runs")
ACCENT = "#55c6b3"
MUTED = "#8193a8"
ICON_PATHS = {
    "activity": '<path d="M3 12h5l3-7 4 14 3-7h3"/>',
    "overview": '<path d="M4 20V11h4v9M10 20V5h4v15M16 20v-7h4v7M3 20h18"/>',
    "forecast": '<path d="M3 17l6-6 4 3 8-8M16 6h5v5"/>',
    "data": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.7 4 3 9 3s9-1.3 9-3V5M3 12c0 1.7 4 3 9 3s9-1.3 9-3"/>',
    "check": '<path d="m5 12 4 4L19 6"/><circle cx="12" cy="12" r="10"/>',
    "calendar": '<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M16 3v4M8 3v4M3 11h18"/>',
    "lock": '<rect x="5" y="10" width="14" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3M12 15v2"/>',
}


def _stamp(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


@st.cache_data(show_spinner=False)
def _csv(path: str, stamp: tuple[int, int]) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _json(path: str, stamp: tuple[int, int]) -> dict | list:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def _predictions(run: RunRecord, stamp: tuple[int, int]) -> pd.DataFrame:
    return run.predictions()


@st.cache_data(show_spinner=False)
def _source(run: RunRecord, source_stamp: tuple[int, int], config_stamp: tuple[int, int],
            manifest_stamp: tuple[int, int]) -> pd.DataFrame:
    return run.source_data()


def _read_csv(path: Path) -> pd.DataFrame:
    return _csv(str(path), _stamp(path))


def _read_json(path: Path) -> dict | list:
    return _json(str(path), _stamp(path))


def _artifact_path(variable: str, default: Path) -> Path:
    path = Path(os.environ.get(variable, str(default)))
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _load_catalog(root: Path) -> ArtifactCatalog | None:
    index = root / "run_index.json"
    if not index.exists():
        return None
    try:
        return load_catalog(root)
    except (FileNotFoundError, ValueError, KeyError, StopIteration):
        return None


def _run(catalog: ArtifactCatalog | None, asset: str) -> RunRecord | None:
    return next((run for run in catalog.runs if run.asset_id == asset), None) if catalog else None


def _asset_label(asset: str) -> str:
    return asset.split(":", 1)[-1].replace("-CUSHING-SPOT", " Cushing spot").replace("-", "/")


def _model_label(model: str) -> str:
    label = model.replace("_", " ").title()
    for name in ("ARIMA", "SARIMA", "RNN", "GRU", "LSTM"):
        label = label.replace(name.title(), name)
    return label


def _number(value: float) -> str:
    return f"{value:,.4g}" if pd.notna(value) else "Unavailable"


def _chart(chart: alt.Chart, height: int) -> alt.Chart:
    return chart.properties(height=height).configure_view(stroke=None).configure_axis(
        labelColor="#a9b7c6", titleColor="#a9b7c6", gridColor="#283342", domain=False, tickColor="#283342")


def _icon(name: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" aria-hidden="true" focusable="false" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" '
            f'stroke-linejoin="round">{ICON_PATHS[name]}</svg>')


def _page_heading(title: str, context: str, icon: str) -> None:
    st.markdown(f'<div class="page-heading">{_icon(icon)}<h1>{title}</h1></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-context">{context}<span class="study-badge">{_icon("lock")} Saved study</span></div>', unsafe_allow_html=True)


def _section_heading(title: str, icon: str) -> None:
    st.markdown(f'<div class="section-heading">{_icon(icon)}<h2>{title}</h2></div>', unsafe_allow_html=True)


def _style() -> None:
    st.markdown("""<style>
    :root {--mc-accent: #55c6b3; --mc-muted: #a4b2c3; --mc-border: #293444; --mc-surface: #121b27;}
    .block-container {max-width: 1600px; padding: 4.5rem 2rem 2rem;}
    h1 {font-size: 1.85rem !important; font-weight: 650 !important; letter-spacing: -.035em;}
    h2 {font-size: 1rem !important; font-weight: 600 !important;}
    .page-heading, .section-heading, .brand-heading {display: flex; align-items: center; gap: .75rem;}
    .page-heading h1, .section-heading h2 {margin: 0 !important; padding: 0 !important;}
    .page-heading svg {width: 1.75rem; height: 1.75rem; color: var(--mc-accent); flex: none;}
    .page-context {display: flex; align-items: center; gap: 1rem; flex-wrap: wrap; color: var(--mc-muted); font-size: .85rem; margin: .5rem 0 1.5rem;}
    .study-badge {display: inline-flex; align-items: center; gap: .4rem; border-left: 1px solid var(--mc-border); padding-left: 1rem; font-size: .75rem;}
    .study-badge svg {width: .85rem; height: .85rem;}
    .section-heading {margin: .25rem 0 .5rem;}
    .section-heading svg {width: 1.1rem; height: 1.1rem; color: var(--mc-accent); flex: none;}
    .brand-heading {font-size: 1.05rem; font-weight: 650; letter-spacing: -.025em;}
    .brand-heading svg {width: 1.5rem; height: 1.5rem; color: var(--mc-accent); flex: none;}
    [data-testid="stSidebar"][aria-expanded="true"] {min-width: 248px !important; max-width: 248px !important; border-right: 1px solid var(--mc-border);}
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {padding-top: 1.5rem;}
    [data-testid="stSidebar"] hr {margin: 1.25rem 0; border-color: var(--mc-border);}
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-testid="stRadioOption"] {width: 100%; padding: .5rem .75rem; border-radius: .25rem; margin: .15rem 0;}
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-testid="stRadioOption"][data-selected="true"] {background: #1b3035;}
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-testid="stRadioOption"] p {display: flex; align-items: center; gap: .6rem; font-size: .85rem;}
    [data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child {display: none;}
    [data-testid="stRadioOption"][data-focus-visible="true"] {outline: 2px solid var(--mc-accent); outline-offset: 2px;}
    [data-testid="stMetric"] {background: var(--mc-surface); border: 1px solid var(--mc-border); border-radius: .35rem; padding: 1rem;}
    [data-testid="stMetricLabel"] p {font-size: .8rem !important; color: var(--mc-muted); white-space: normal !important;}
    [data-testid="stMetricValue"] {font-size: 1.6rem; font-weight: 550; letter-spacing: -.025em;}
    [data-testid="stVerticalBlockBorderWrapper"] {border-color: var(--mc-border) !important; border-radius: .5rem !important;}
    [data-testid="stCaptionContainer"] {color: var(--mc-muted);}
    [data-testid="stExpander"] {border-color: var(--mc-border); border-radius: .35rem;}
    [data-testid="stExpander"] summary {font-size: .85rem;}
    .sidebar-note {color: var(--mc-muted); font-size: .75rem; line-height: 1.7;}
    .sidebar-note svg {width: .85rem; height: .85rem; vertical-align: -.1rem; margin-right: .35rem;}
    button:focus-visible, input:focus-visible {outline: 2px solid var(--mc-accent) !important; outline-offset: 3px;}
    @media (max-width: 900px) {.block-container {padding-left: 1rem; padding-right: 1rem;} [data-testid="stHorizontalBlock"] {flex-wrap: wrap;} [data-testid="stColumn"] {min-width: min(100%, 220px); flex: 1 1 220px;}}
    @media (max-width: 1000px) {.st-key-overview-panels [data-testid="stColumn"] {min-width: 100%; flex: 1 1 100%;}}
    @media (max-width: 480px) {.block-container {padding: 4rem .75rem 1rem;} h1 {font-size: 1.5rem !important;} .page-context {gap: .5rem;} [data-testid="stColumn"] {min-width: 100%;} .study-badge {border: 0; padding-left: 0;}}
    </style>""", unsafe_allow_html=True)
    navigation_css = []
    for index, name in enumerate(("overview", "forecast", "data"), 1):
        svg = _icon(name).replace('currentColor', '#a4b2c3')
        uri = "data:image/svg+xml," + quote(svg)
        navigation_css.append(
            f'[data-testid="stSidebar"] [data-testid="stRadio"] label[data-testid="stRadioOption"]:has(input[value="{index - 1}"]) p::before '
            f'{{content: ""; width: 1rem; height: 1rem; flex: none; background: url("{uri}") center/contain no-repeat;}}'
        )
    st.markdown("<style>" + "".join(navigation_css) + "</style>", unsafe_allow_html=True)


def _overview(root: Path, summary: pd.DataFrame, asset: str, horizon: int) -> None:
    _page_heading("Model evidence", f"{_asset_label(asset)} · {horizon} observed sessions · saved validation study", "overview")
    rows = summary[(summary.asset_id == asset) & (summary.horizon == horizon)].sort_values("rmse_mean").copy()
    if rows.empty:
        st.info("No saved comparison is available for this selection.")
        return
    recs = _read_csv(root / "recommendations.csv")
    chosen = recs[(recs.asset_id == asset) & (recs.horizon == horizon)]
    selected = chosen.iloc[0] if not chosen.empty else rows.iloc[0]
    baseline = rows[rows.model == "last_value"]
    baseline_rmse = float(baseline.iloc[0].rmse_mean) if not baseline.empty else float("nan")
    improvement = (1 - float(selected.rmse_mean) / baseline_rmse) * 100 if baseline_rmse > 0 else float("nan")
    a, b, c = st.columns(3)
    a.metric("Recommended model", _model_label(str(selected.model)))
    b.metric("Validation RMSE", _number(float(selected.rmse_mean)))
    c.metric("Versus last value", f"{improvement:+.1f}%" if pd.notna(improvement) else "Unavailable",
             help="Positive means lower validation RMSE than the last-value baseline.")

    st.write("")
    with st.container(key="overview-panels"):
        comparison_panel, holdout_panel = st.columns([3, 1.15], gap="large")
        with comparison_panel, st.container(border=True):
            _section_heading("Validation RMSE by model", "activity")
            plot = rows.assign(model_label=rows.model.map(_model_label), recommended=rows.model.eq(selected.model))
            chart = alt.Chart(plot).mark_bar(cornerRadiusEnd=3).encode(
                x=alt.X("rmse_mean:Q", title="Mean RMSE across validation folds", scale=alt.Scale(zero=True)),
                y=alt.Y("model_label:N", title=None, sort=alt.SortField(field="rmse_mean", order="ascending")),
                color=alt.condition("datum.recommended", alt.value(ACCENT), alt.value(MUTED)),
                tooltip=[alt.Tooltip("model_label:N", title="Model"),
                         alt.Tooltip("rmse_mean:Q", title="Validation RMSE", format=".4g"),
                         alt.Tooltip("rmse_std:Q", title="Fold spread", format=".4g"),
                         alt.Tooltip("fit_seconds_mean:Q", title="Mean fit time (s)", format=".3g")])
            st.altair_chart(_chart(chart, min(400, max(260, len(rows) * 26))), width="stretch")
            st.caption("Lower is better. The highlighted model follows the saved validation-only recommendation rule.")
        with holdout_panel, st.container(border=True):
            _section_heading("Locked holdout", "lock")
            st.metric("Recommended model RMSE", _number(float(selected.final_test_rmse)))
            st.metric("Last-value RMSE", _number(float(baseline.iloc[0].final_test_rmse)) if not baseline.empty else "Unavailable")
            st.metric("Validation fold spread", _number(float(selected.rmse_std)))
            st.caption("Holdout scores check the selected model; they do not choose it. Raw RMSE is comparable only within an asset and horizon.")
    with st.expander("All model scores and selection details"):
        st.dataframe(rows[["model", "rmse_mean", "rmse_std", "mae_mean", "mase_mean", "final_test_rmse", "fit_seconds_mean"]],
                     width="stretch", hide_index=True)
        st.caption("Non-baseline recommendations require at least 5% improvement in validation mean RMSE and mean plus one fold standard deviation.")


def _historical(run: RunRecord | None, horizon: int) -> None:
    if run is None:
        st.info("Detailed backtests need the full saved run files. Overview remains available from the committed comparison snapshot.")
        return
    metrics = _read_csv(run.directory / "metrics.csv")
    metrics = metrics[metrics.horizon == horizon]
    if metrics.empty:
        st.info("No backtest is saved for this horizon.")
        return
    controls = st.columns([2, 2, 1])
    model = controls[0].selectbox("Model", sorted(metrics.model.unique()), format_func=_model_label)
    partition = controls[1].selectbox("Evaluation period", ("validation", "final_test"),
                                       format_func=lambda value: "Walk-forward validation" if value == "validation" else "Locked holdout")
    folds = sorted(metrics.loc[metrics.partition == "validation", "fold"].unique())
    fold = controls[2].selectbox("Fold", folds) if partition == "validation" else 0
    selected = metrics[(metrics.model == model) & (metrics.partition == partition) & (metrics.fold == fold)]
    if selected.empty:
        st.info("No score is saved for this selection.")
        return
    manifest = _read_json(run.directory / "data_manifest.json")
    predictions = _predictions(run, _stamp(run.directory / manifest["prediction_file"]))
    predictions = predictions[(predictions.model == model) & (predictions.horizon == horizon) &
                              (predictions.partition == partition) & (predictions.fold == fold)].sort_values("timestamp")
    if predictions.empty:
        st.info("No prediction rows are saved for this selection.")
        return
    row = selected.iloc[0]
    a, b, c = st.columns(3)
    a.metric("RMSE", _number(row.rmse))
    b.metric("MAE", _number(row.mae))
    c.metric("Fit + inference", f"{row.fit_seconds + row.inference_seconds:.3g}s")
    plot = predictions.melt(id_vars="timestamp", value_vars=["actual", "prediction"], var_name="series", value_name="value")
    chart = alt.Chart(plot).mark_line(point=True).encode(
        x=alt.X("timestamp:T", title="Recorded target date"), y=alt.Y("value:Q", title=manifest["target_column"]),
        color=alt.Color("series:N", scale=alt.Scale(domain=["actual", "prediction"], range=[MUTED, ACCENT]), title=None),
        tooltip=[alt.Tooltip("timestamp:T", title="Target date"), "series:N", alt.Tooltip("value:Q", format=".4g")]).interactive(bind_y=False)
    st.altair_chart(_chart(chart, 330), width="stretch")
    st.caption("Historical out-of-sample predictions on observed dates. These are not future forecasts.")
    with st.expander("Prediction rows and residual diagnostics"):
        st.dataframe(predictions[["timestamp", "step", "actual", "prediction"]], width="stretch", hide_index=True)
        diagnostics = _read_json(run.directory / "diagnostics.json")["models"]
        matching = [item for item in diagnostics if item.get("model") == model and item.get("fold") == fold]
        if matching:
            st.json(matching[-1])


def _scenario(asset: str, horizon: int) -> None:
    root = _artifact_path("MARKETCAST_PHASE4_ROOT", PROJECT_ROOT / "assets" / "phase4")
    path = root / "future_forecasts.csv"
    if not path.exists():
        st.info("No saved future scenario is available. Generate one with `python main.py build-future-scenarios`.")
        return
    frame = _read_csv(path)
    selected = frame[(frame.asset_id == asset) & (frame.horizon == horizon)].sort_values("step")
    if selected.empty:
        st.info("No scenario is saved for this asset and horizon.")
        return
    origin = selected.iloc[0]
    st.caption(f"Speculative scenario from {str(origin.origin_timestamp)[:10]} · {_model_label(str(origin.model))} · {horizon} observed sessions")
    a, b = st.columns(2)
    a.metric("First projected value", _number(float(selected.iloc[0].prediction)))
    b.metric("Final projected value", _number(float(selected.iloc[-1].prediction)))
    chart = alt.Chart(selected).mark_line(point=True, color=ACCENT).encode(
        x=alt.X("step:Q", title="Observed sessions after origin", axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("prediction:Q", title="Projected value", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("step:Q", title="Step"), alt.Tooltip("prediction:Q", title="Projection", format=".4g")])
    st.altair_chart(_chart(chart, 330), width="stretch")
    st.caption("Steps have no invented calendar dates. Current market conditions may differ from the saved origin.")
    with st.expander("Scenario values and provenance"):
        st.dataframe(selected[["step", "prediction", "model", "origin_timestamp", "source_run_id"]],
                     width="stretch", hide_index=True)


def _forecasts(catalog: ArtifactCatalog | None, asset: str, horizon: int) -> None:
    _page_heading("Forecasts", f"{_asset_label(asset)} · {horizon} observed sessions", "forecast")
    view = st.radio("Forecast view", ("Historical backtest", "Saved future scenario"), horizontal=True)
    if view == "Historical backtest":
        _historical(_run(catalog, asset), horizon)
    else:
        _scenario(asset, horizon)


def _data_and_runs(catalog: ArtifactCatalog | None, asset: str) -> None:
    _page_heading("Data & runs", f"{_asset_label(asset)} · source history, diagnostics and saved experiment details", "data")
    run = _run(catalog, asset)
    if run is None:
        st.info("Full run files are unavailable. Generate the study to inspect source history and diagnostics; saved comparisons remain on Overview.")
        return
    manifest = _read_json(run.directory / "data_manifest.json")
    config = _read_json(run.directory / "config.json")
    quality = _read_json(run.directory / "quality_report.json")
    source_path = Path(config["data_path"])
    if not source_path.is_absolute():
        source_path = PROJECT_ROOT / source_path
    try:
        frame = _source(run, _stamp(source_path), _stamp(run.directory / "config.json"),
                        _stamp(run.directory / "data_manifest.json"))
        target = manifest["target_column"]
        chart_frame = frame[["timestamp", target]].tail(1500)
        chart = alt.Chart(chart_frame).mark_line(color=ACCENT).encode(
            x=alt.X("timestamp:T", title="Observed date"), y=alt.Y(f"{target}:Q", title=target),
            tooltip=[alt.Tooltip("timestamp:T", title="Date"), alt.Tooltip(f"{target}:Q", title=target, format=".4g")]).interactive(bind_y=False)
        st.altair_chart(_chart(chart, 330), width="stretch")
    except (FileNotFoundError, ValueError, KeyError) as exc:
        figure = run.directory / "figures" / "eda.png"
        if figure.exists():
            st.image(str(figure), caption="Saved source analysis figure")
        st.caption(f"Live source chart unavailable: {exc}")
    st.caption(f"{manifest['source_name']} · {manifest['target_semantics']} · observed dates only")
    a, b, c = st.columns(3)
    a.metric("Observations", f"{quality['row_count']:,}")
    b.metric("Duplicate keys", quality["duplicate_key_count"])
    c.metric("Return outlier flags", len(quality["close_return_outlier_dates"]))
    with st.expander("Time series diagnostics"):
        eda = _read_json(run.directory / "eda.json")
        st.write(eda["decision"])
        st.caption(f"Level ADF p-value: {eda['price']['adf_pvalue']:.3g} · KPSS p-value: {eda['price']['kpss_pvalue']:.3g}")
        st.json(eda)
    with st.expander("Experiment setup and provenance"):
        st.dataframe(pd.DataFrame(manifest["folds"])[["fold", "train_start_date", "train_end_date", "test_start_date", "test_end_date"]],
                     width="stretch", hide_index=True)
        st.caption(f"Run {run.run_id} · holdout {manifest['final_test_start_date'][:10]} to {manifest['final_test_end_date'][:10]}")
        st.json(config)
        st.json(quality)
    with st.expander("Run history and diagnostics"):
        records = pd.DataFrame([{"asset": item.asset_id, "run_id": item.run_id,
                                 "wall_seconds": item.wall_seconds} for item in catalog.runs])
        st.dataframe(records, width="stretch", hide_index=True)
        st.write("Warnings", _read_json(run.directory / "warnings.json"))
        st.write("Failures", _read_json(run.directory / "failures.json"))
        st.json(_read_json(run.directory / "environment.json"))


def main() -> None:
    _style()
    root = _artifact_path("MARKETCAST_ARTIFACT_ROOT", PROJECT_ROOT / "assets" / "phase3")
    try:
        summary = _read_csv(root / "comparison.csv")
    except (FileNotFoundError, pd.errors.EmptyDataError) as exc:
        st.title("Experiment artifacts needed")
        st.error(f"Comparison snapshot unavailable: {exc}")
        st.code("python main.py four-markets --config configs/four_asset_full.json", language="powershell")
        st.stop()
    if summary.empty:
        st.title("Experiment artifacts needed")
        st.info("The saved comparison has no model results.")
        st.stop()
    with st.sidebar:
        st.markdown(f'<div class="brand-heading">{_icon("activity")}<span>MarketCast Lab</span></div>',
                    unsafe_allow_html=True)
        st.caption("FORECAST RESEARCH")
        page = st.radio("View", PAGES)
        st.divider()
        asset = st.selectbox("Asset", sorted(summary.asset_id.unique()), format_func=_asset_label)
        horizons = sorted(int(value) for value in summary.loc[summary.asset_id == asset, "horizon"].unique())
        horizon = st.selectbox("Horizon · observed sessions", horizons, index=horizons.index(5) if 5 in horizons else 0)
        st.divider()
        st.markdown(f'<div class="sidebar-note">{_icon("calendar")} Study through 14 Oct 2025<br>{_icon("lock")} Validation before holdout<br><br>Educational analysis.<br>Not financial advice.</div>', unsafe_allow_html=True)
    catalog = _load_catalog(root) if page != "Overview" else None
    if (root / "SYNTHETIC_FIXTURE.txt").exists():
        st.warning("Synthetic offline fixture. Values are demonstration data, not market evidence.")
    if page == "Overview":
        _overview(root, summary, asset, horizon)
    elif page == "Forecasts":
        _forecasts(catalog, asset, horizon)
    else:
        _data_and_runs(catalog, asset)


if __name__ == "__main__":
    main()
