"""Artifact-only MarketCast Lab dashboard. No model is fitted while rendering."""
from __future__ import annotations

from pathlib import Path
import json
import os
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) in sys.path:
    sys.path.remove(str(SOURCE_ROOT))
    sys.path.append(str(SOURCE_ROOT))

import streamlit as st

from market_forecast.dashboard.artifacts import load_catalog, ArtifactCatalog, RunRecord

st.set_page_config(page_title="MarketCast Lab", page_icon="ðŸ“ˆ", layout="wide")

PAGES = ("Data Explorer", "Time-Series Analysis", "Experiment Setup", "Backtest Results",
         "Model Comparison", "Future Forecast", "Experiment History")


@st.cache_data(show_spinner=False)
def _catalog() -> ArtifactCatalog:
    return load_catalog(os.environ.get("MARKETCAST_ARTIFACT_ROOT", str(PROJECT_ROOT / "assets" / "phase3")))


def _asset_select(catalog: ArtifactCatalog, label: str = "Asset") -> RunRecord:
    asset_id = st.selectbox(label, [run.asset_id for run in catalog.runs])
    return catalog.for_asset(asset_id)


def _run_caption(run: RunRecord) -> None:
    manifest = run.json("data_manifest.json")
    st.caption(f"Run {run.run_id} Â· {manifest['source_name']} Â· target: {manifest['target_column']} Â· "
               f"holdout: {manifest['final_test_start_date'][:10]} to {manifest['final_test_end_date'][:10]}")


def _study_commands(catalog: ArtifactCatalog) -> str:
    path = catalog.root / "study_manifest.json"
    if path.exists():
        study = json.loads(path.read_text(encoding="utf-8"))
        if study.get("synthetic"):
            return (f"python main.py offline-fixture --assets {study['assets_path']} --output {catalog.root}\n"
                    f"python main.py audit-four-markets --output {catalog.root}")
        return (f"python main.py four-markets --config {study['config_path']} --assets {study['assets_path']} --output {catalog.root}\n"
                f"python main.py audit-four-markets --output {catalog.root}")
    return ("python main.py four-markets --config configs/four_asset_full.json\n"
            "python main.py audit-four-markets")


def _data_explorer(catalog: ArtifactCatalog) -> None:
    st.title("Data Explorer")
    run = _asset_select(catalog)
    _run_caption(run)
    manifest = run.json("data_manifest.json")
    quality = run.json("quality_report.json")
    a, b, c = st.columns(3)
    a.metric("Observations", f"{quality['row_count']:,}")
    b.metric("Duplicate keys", quality["duplicate_key_count"])
    c.metric("Return outlier flags", len(quality["close_return_outlier_dates"]))
    st.write(f"**Source:** {manifest['source_name']}")
    st.write(f"**Target:** {manifest['target_semantics']}")
    st.write(f"**Calendar:** {manifest['calendar']}. Dates without a source observation remain absent.")
    try:
        frame = run.source_data()
        target = manifest["target_column"]
        st.line_chart(frame.set_index("timestamp")[[target]], height=320)
        st.dataframe(frame.tail(20), width="stretch", hide_index=True)
    except (FileNotFoundError, ValueError) as exc:
        st.info(str(exc))
        st.image(str(run.directory / "figures" / "eda.png"), caption="Saved EDA figure from the experiment")
    with st.expander("Quality, provenance and transformations"):
        st.json(quality)


def _time_series_analysis(catalog: ArtifactCatalog) -> None:
    st.title("Time-Series Analysis")
    run = _asset_select(catalog)
    _run_caption(run)
    eda = run.json("eda.json")
    st.write(eda["decision"])
    a, b, c = st.columns(3)
    a.metric("ADF p-value, level", f"{eda['price']['adf_pvalue']:.3g}")
    b.metric("KPSS p-value, level", f"{eda['price']['kpss_pvalue']:.3g}")
    c.metric("Change standard deviation", f"{eda['return_std']:.3g}")
    st.caption(f"Change measure: {eda.get('change_kind', 'fractional return')}")
    st.image(str(run.directory / "figures" / "eda.png"), caption="Target level, changes and rolling statistics; holdout excluded")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Autocorrelation")
        st.line_chart(pd.DataFrame({"price ACF": eda["price"]["acf"],
                                    "return ACF": eda["returns"]["acf"]}))
    with col2:
        st.subheader("Partial autocorrelation")
        st.line_chart(pd.DataFrame({"price PACF": eda["price"]["pacf"],
                                    "return PACF": eda["returns"]["pacf"]}))
    with st.expander("Decomposition and full EDA diagnostics"):
        st.json(eda)


def _experiment_setup(catalog: ArtifactCatalog) -> None:
    st.title("Experiment Setup")
    run = _asset_select(catalog)
    _run_caption(run)
    config = run.json("config.json")
    manifest = run.json("data_manifest.json")
    st.write("This view describes a saved experiment. Rendering never starts training.")
    st.dataframe(pd.DataFrame(manifest["folds"])[["fold", "train_start_date", "train_end_date",
                                                 "test_start_date", "test_end_date"]],
                 width="stretch", hide_index=True)
    a, b, c = st.columns(3)
    a.metric("Validation folds", config["folds"])
    b.metric("Horizons", ", ".join(map(str, config["horizons"])))
    c.metric("Models", len(config["models"]))
    st.code(_study_commands(catalog), language="powershell")
    with st.expander("Saved configuration and selected parameters"):
        st.json(config)
        st.json(manifest["selected_model_parameters"])


def _backtest_results(catalog: ArtifactCatalog) -> None:
    st.title("Backtest Results")
    run = _asset_select(catalog)
    _run_caption(run)
    metrics = run.metrics()
    models = sorted(metrics.model.unique())
    model = st.selectbox("Model", models)
    horizon = st.selectbox("Horizon in observed sessions", sorted(metrics.horizon.unique()))
    partition = st.radio("Evaluation period", ["validation", "final_test"], horizontal=True,
                         format_func=lambda value: "Walk-forward validation" if value == "validation" else "Locked final holdout")
    fold = st.selectbox("Validation fold", [1, 2, 3]) if partition == "validation" else 0
    selected = metrics[(metrics.model == model) & (metrics.horizon == horizon) &
                       (metrics.partition == partition) & (metrics.fold == fold)]
    predictions = run.predictions()
    predictions = predictions[(predictions.model == model) & (predictions.horizon == horizon) &
                              (predictions.partition == partition) & (predictions.fold == fold)]
    if selected.empty or predictions.empty:
        st.warning("No saved prediction for this combination. Check the run failures.")
        return
    st.info("Historical out-of-sample predictions on recorded target dates. This chart is not a future forecast.")
    row = selected.iloc[0]
    a, b, c, d = st.columns(4)
    a.metric("RMSE", f"{row.rmse:.4g}")
    b.metric("MAE", f"{row.mae:.4g}")
    c.metric("MASE", f"{row.mase:.3g}")
    d.metric("Fit + inference", f"{row.fit_seconds + row.inference_seconds:.3g}s")
    chart = predictions.set_index("timestamp")[["actual", "prediction"]]
    st.line_chart(chart, height=350)
    st.dataframe(predictions[["timestamp", "step", "actual", "prediction"]],
                 width="stretch", hide_index=True)
    st.write(f"Interval coverage: {'unavailable' if pd.isna(row.interval_coverage) else f'{row.interval_coverage:.1%}'}")
    diagnostics = run.json("diagnostics.json")["models"]
    matching = [item for item in diagnostics if item.get("model") == model and item.get("fold") == fold]
    if matching:
        with st.expander("Residual diagnostics and model complexity"):
            st.json(matching[-1])


def _model_comparison(catalog: ArtifactCatalog) -> None:
    st.title("Model Comparison")
    st.caption("Validation ranking, fold spread, holdout, runtime and available interval coverage")
    summary = catalog.table("comparison.csv")
    folds = catalog.table("fold_metrics.csv")
    recs = catalog.table("recommendations.csv")
    classes = ["All"] + sorted(summary.asset_class.unique())
    asset_class = st.selectbox("Asset class", classes)
    if asset_class != "All":
        summary = summary[summary.asset_class == asset_class]
        folds = folds[folds.asset_class == asset_class]
    assets = ["All"] + sorted(summary.asset_id.unique())
    asset = st.selectbox("Asset", assets)
    if asset != "All":
        summary = summary[summary.asset_id == asset]
        folds = folds[folds.asset_id == asset]
    horizon = st.selectbox("Horizon", ["All", 1, 5, 20])
    if horizon != "All":
        summary = summary[summary.horizon == horizon]
        folds = folds[folds.horizon == horizon]
    model_choices = sorted(summary.model.unique())
    models = st.multiselect("Models", model_choices, default=model_choices)
    summary = summary[summary.model.isin(models)]
    folds = folds[folds.model.isin(models)]
    fold = st.selectbox("Fold", ["Aggregate", 1, 2, 3, "Final holdout"])
    if fold == "Aggregate":
        columns = ["asset_id", "horizon", "model", "rmse_mean", "rmse_std", "mae_mean",
                   "mase_mean", "final_test_rmse", "fit_seconds_mean", "interval_coverage_mean"]
        st.dataframe(summary[columns].sort_values(["asset_id", "horizon", "rmse_mean"]),
                     width="stretch", hide_index=True)
    else:
        subset = folds[folds.partition == ("final_test" if fold == "Final holdout" else "validation")]
        if fold != "Final holdout":
            subset = subset[subset.fold == fold]
        st.dataframe(subset[["asset_id", "horizon", "model", "fold", "rmse", "mae", "mase",
                             "fit_seconds", "inference_seconds", "interval_coverage", "run_id"]],
                     width="stretch", hide_index=True)
    st.subheader("Recommended models")
    chosen = recs[(recs.asset_id.isin(summary.asset_id.unique())) &
                  (recs.horizon.isin(summary.horizon.unique()))]
    st.dataframe(chosen[["asset_id", "horizon", "model", "rmse_mean", "rmse_std", "final_test_rmse"]],
                 width="stretch", hide_index=True)
    st.caption("A non-baseline model must improve validation mean RMSE and mean plus one fold standard deviation by at least 5%. The holdout is not used for selection.")
    st.image(str(catalog.root / "fold_variability.png"), caption="Fold RMSE divided by same-fold last-value RMSE")
    with st.expander("Residual diagnostics and model complexity across markets"):
        st.dataframe(catalog.table("residual_diagnostics.csv"), width="stretch", hide_index=True)


def _future_forecast(catalog: ArtifactCatalog) -> None:
    st.title("Future Forecast")
    st.warning("Speculative scenario from a saved historical origin. These values are separate from backtest and holdout scores.")
    path = Path(os.environ.get("MARKETCAST_PHASE4_ROOT", str(PROJECT_ROOT / "assets" / "phase4"))) / "future_forecasts.csv"
    if not path.exists():
        st.info("No scenario has been generated. Run `python main.py build-future-scenarios` explicitly, then refresh this page.")
        return
    frame = pd.read_csv(path)
    asset = st.selectbox("Asset", sorted(frame.asset_id.unique()))
    horizon = st.selectbox("Scenario length in observed sessions", [1, 5, 20])
    selected = frame[(frame.asset_id == asset) & (frame.horizon == horizon)]
    st.caption(f"Origin: {selected.origin_timestamp.iloc[0]}; source run: {selected.source_run_id.iloc[0]}; model: {selected.model.iloc[0]}")
    st.line_chart(selected.set_index("step")[["prediction"]], height=320)
    st.dataframe(selected[["step", "prediction", "model", "origin_timestamp"]],
                 width="stretch", hide_index=True)
    st.caption("Steps have no fabricated calendar dates. Current market conditions may differ from this saved origin.")


def _experiment_history(catalog: ArtifactCatalog) -> None:
    st.title("Experiment History")
    records = pd.DataFrame([{"run_id": run.run_id, "asset": run.asset_id,
                             "class": run.asset_class, "wall_seconds": run.wall_seconds,
                             "warnings": len(run.json("warnings.json")),
                             "failures": len(run.json("failures.json"))} for run in catalog.runs])
    st.dataframe(records, width="stretch", hide_index=True)
    run_id = st.selectbox("Inspect run", records.run_id.tolist())
    run = catalog.run(run_id)
    _run_caption(run)
    st.write("**Warnings**", run.json("warnings.json"))
    st.write("**Failures**", run.json("failures.json"))
    with st.expander("Config, data manifest and environment"):
        st.json(run.json("config.json"))
        st.json(run.json("data_manifest.json"))
        st.json(run.json("environment.json"))


def main() -> None:
    st.sidebar.title("MarketCast Lab")
    st.sidebar.caption("Artifact-driven daily forecasting study")
    page = st.sidebar.radio("View", PAGES)
    st.sidebar.divider()
    st.sidebar.caption("Educational analysis only. Forecasts are not financial advice.")
    try:
        catalog = _catalog()
    except (FileNotFoundError, ValueError) as exc:
        st.title("Experiment artifacts needed")
        st.error(str(exc))
        st.code("python main.py four-markets --config configs/four_asset_full.json\n"
                "python main.py audit-four-markets", language="powershell")
        fallback = PROJECT_ROOT / "images" / "relative_rmse.png"
        if fallback.exists():
            st.image(str(fallback), caption="Saved demonstration figure; generate the run artifacts for interactive evidence")
        st.stop()
    if (catalog.root / "SYNTHETIC_FIXTURE.txt").exists():
        st.warning("Synthetic offline fixture. All values and conclusions here are software demonstration data, not market evidence.")
    render = {"Data Explorer": _data_explorer, "Time-Series Analysis": _time_series_analysis,
              "Experiment Setup": _experiment_setup, "Backtest Results": _backtest_results,
              "Model Comparison": _model_comparison, "Future Forecast": _future_forecast,
              "Experiment History": _experiment_history}
    render[page](catalog)
    st.divider()
    st.caption("Historical tests and speculative scenarios are labeled separately. MarketCast Lab is not financial advice.")


if __name__ == "__main__":
    main()
