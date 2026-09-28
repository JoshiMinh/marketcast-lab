"""Deterministic course deliverables built from saved Phase 3 artifacts."""
from __future__ import annotations

from pathlib import Path
import json
import shutil
from datetime import datetime, timezone

import pandas as pd

from market_forecast.dashboard.artifacts import ArtifactCatalog, load_catalog
from market_forecast.models import default_registry
from market_forecast.reports import audit_cross_market_report


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame[columns].copy()
    for column in selected:
        if pd.api.types.is_float_dtype(selected[column]):
            selected[column] = selected[column].map(lambda value: "—" if pd.isna(value) else f"{value:.4g}")
    rows = ["| " + " | ".join(map(str, columns)) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    rows.extend("| " + " | ".join(str(value) for value in row) + " |" for row in selected.itertuples(index=False, name=None))
    return "\n".join(rows)


ORIGINAL_ASSETS = {"crypto:BTC-USD", "equity:SPY", "forex:EUR-USD", "oil:WTI-CUSHING-SPOT"}


def _original_study(catalog: ArtifactCatalog) -> bool:
    return {run.asset_id for run in catalog.runs} == ORIGINAL_ASSETS


def _build_extended_report(catalog: ArtifactCatalog, output: Path) -> Path:
    audit = audit_cross_market_report(catalog.root)
    recommendations = catalog.table("recommendations.csv")
    sources = pd.DataFrame([{
        "asset": run.asset_id,
        "source": run.json("data_manifest.json")["source_name"],
        "target": run.json("data_manifest.json")["target_semantics"],
        "unit": run.json("data_manifest.json").get("unit", "USD"),
        "reuse": run.json("data_manifest.json")["license_redistribution_status"],
    } for run in catalog.runs])
    columns = ["asset_id", "horizon", "model", "rmse_mean", "rmse_std", "final_test_rmse"]
    text = (f"# MarketCast Lab: {len(catalog.runs)} daily series\n\n"
            f"{audit['metric_rows_reproduced']} saved metric rows reproduced from predictions. "
            "Each asset has separate observed dates, validation folds, and a locked holdout. "
            "Targets include adjusted prices, spot/reference values, and possibly percentage yields; "
            "raw RMSE values across different units are not comparable.\n\n"
            "## Sources and target semantics\n\n"
            + _markdown_table(sources, ["asset", "source", "target", "unit", "reuse"])
            + "\n\n## Validation selections and holdout\n\n"
            + _markdown_table(recommendations, columns)
            + "\n\nSelection requires at least 5% improvement over last value in both mean RMSE "
              "and mean plus one fold standard deviation. Holdout scores are displayed after selection. "
              "Inspect the saved manifests for exact source hashes, fold dates, and model parameters. "
              "This is historical model evaluation, not investment advice.\n")
    if (catalog.root / "SYNTHETIC_FIXTURE.txt").exists():
        text = "**SYNTHETIC OFFLINE FIXTURE — SOFTWARE CHECK ONLY.**\n\n" + text
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output


def _build_extended_presentation(catalog: ArtifactCatalog, output: Path) -> Path:
    rows = catalog.table("recommendations.csv")
    lines = ["# MarketCast Lab", f"{len(catalog.runs)} daily series; saved backtests.",
             "\n---\n", "# Sources", *[f"- {run.asset_id}: {run.json('data_manifest.json')['source_name']}" for run in catalog.runs],
             "\n---\n", "# Selected models",
             _markdown_table(rows, ["asset_id", "horizon", "model", "rmse_mean", "final_test_rmse"]),
             "\n---\n", "# Interpretation", "Compare models within an asset and target unit.",
             "Holdout results follow validation-only selection. Historical analysis only."]
    if (catalog.root / "SYNTHETIC_FIXTURE.txt").exists():
        lines[:0] = ["# SYNTHETIC OFFLINE FIXTURE", "Software check only.", "\n---\n"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def build_scientific_report(catalog: ArtifactCatalog, output: Path) -> Path:
    if not _original_study(catalog):
        return _build_extended_report(catalog, output)
    audit = audit_cross_market_report(catalog.root)
    comparison = catalog.table("comparison.csv")
    recommendations = catalog.table("recommendations.csv").sort_values(["asset_class", "asset_id", "horizon"])
    folds = catalog.table("fold_metrics.csv")
    residuals = catalog.table("residual_diagnostics.csv")
    failures = catalog.table("failures.csv")
    sources = []
    quality_rows = []
    eda_rows = []
    boundary_rows = []
    warning_rows = []
    for run in catalog.runs:
        manifest = run.json("data_manifest.json")
        quality = run.json("quality_report.json")
        eda = run.json("eda.json")
        warnings = run.json("warnings.json")
        sources.append({"asset": run.asset_id, "provider": manifest["source_name"],
                        "target": manifest["target_column"], "reuse": manifest["license_redistribution_status"],
                        "source_url": manifest["source_url"] or "unknown"})
        quality_rows.append({"asset": run.asset_id, "rows": quality["row_count"],
                             "first": quality["start"][:10], "last": quality["end"][:10],
                             "duplicate keys": quality["duplicate_key_count"],
                             "invalid OHLC": quality["inconsistent_ohlc_row_count"],
                             "outlier flags": len(quality["close_return_outlier_dates"])})
        eda_rows.append({"asset": run.asset_id, "ADF p": eda["price"]["adf_pvalue"],
                         "KPSS p": eda["price"]["kpss_pvalue"],
                         "return std": eda["return_std"], "period": eda["seasonal_period"]})
        boundary_rows.append({"asset": run.asset_id,
                              "first validation": manifest["folds"][0]["test_start_date"][:10],
                              "last validation": manifest["folds"][-1]["test_end_date"][:10],
                              "holdout": f"{manifest['final_test_start_date'][:10]} to {manifest['final_test_end_date'][:10]}"})
        warning_rows.extend({"asset": run.asset_id, "warning": warning} for warning in warnings)
    baseline = comparison[comparison.model == "last_value"][["asset_id", "horizon", "final_test_rmse"]].rename(
        columns={"final_test_rmse": "last_value_holdout_rmse"})
    selected = recommendations.merge(baseline, on=["asset_id", "horizon"])
    selected["holdout_vs_last_value_pct"] = 100 * (selected.final_test_rmse / selected.last_value_holdout_rmse - 1)
    selected["fold_cv"] = selected.rmse_std / selected.rmse_mean
    intervals = comparison[comparison.interval_coverage_mean.notna()][["asset_id", "horizon", "model", "interval_coverage_mean", "final_test_interval_coverage"]]
    diagnostics = residuals.groupby(["asset_id", "model"], as_index=False).agg(
        ljung_box_pvalue_mean=("ljung_box_pvalue", "mean"),
        parameter_count=("parameter_count", "max"))
    result = f"""# MarketCast Lab: forecasting across four daily markets

## Abstract and question

How do statistical, lag-based machine-learning and recurrent neural models compare across crypto, equity, foreign exchange and crude oil, and how does that comparison change at 1, 5 and 20 observed-session horizons? Four immutable Phase 3 runs provide {len(folds)} fold and holdout metric rows. The audit reproduced {audit['metric_rows_reproduced']} MAE/RMSE rows from saved predictions. Results are historical model evaluation, not investment advice.

## Concepts and target

**Trend** is sustained movement in the level; **seasonality** is a recurring pattern tied to a defensible period; **cycles** are broader, irregular swings; **noise** is movement unexplained by the fitted structure. **Stationarity** means the distributional properties used by a model do not change over time. Price levels often challenge this assumption, so the EDA tests levels and returns, while ARIMA uses first differencing. **Target** is the future observed price: adjusted close for SPY, close for BTC and the ECB/EIA point observations. **Horizon** is the next 1, 5 or 20 observed market dates. **Recursive strategy** means each later forecast step uses prior predictions, not newly observed outcomes.

## Sources, provenance and rights

{_markdown_table(pd.DataFrame(sources), ['asset','provider','target','reuse','source_url'])}

Every run manifest records the source URL or the known gap, raw SHA-256, retrieval time basis, symbol mapping, currency, timezone, adjustment, target semantics and calendar. SPY raw data is retained only in the ignored local cache. The bundled BTC upstream source and original acquisition time are unknown. WTI is an EIA Cushing spot assessment, so futures rolling and roll jumps do not apply.

## Cleaning and quality

{_markdown_table(pd.DataFrame(quality_rows), ['asset','rows','first','last','duplicate keys','invalid OHLC','outlier flags'])}

Dates are sorted and duplicate asset/date keys rejected. Missing source prices are dropped, with no synthetic holiday or weekend records. ECB and EIA provide point prices, so their OHLC and volume fields are null. The quality JSON files also contain missing-value counts, observed-date gap distributions, weekday gaps that include holidays, a >10% close-return outlier flag, provenance and transformations. The BTC and SPY OHLC bars remain source values; SPY's model target uses its separate adjusted close.

## Exploratory analysis and transformation decision

{_markdown_table(pd.DataFrame(eda_rows), ['asset','ADF p','KPSS p','return std','period'])}

Each run has price/return plots, 20-observation rolling mean and volatility, exploratory decomposition, level/return ACF and PACF, ADF and KPSS results. Its `eda.md` states the asset-specific transformation decision. The weekly period is seven consecutive observations for BTC and a five-observation working-week proxy for the other series. Holiday gaps mean this proxy is exploratory. Forecasts are evaluated in price units. Scalers for lag regressors and sequence networks fit on each training fold only; evaluation never constructs features across assets.

## Validation design and models

{_markdown_table(pd.DataFrame(boundary_rows), ['asset','first validation','last validation','holdout'])}

The shared contract uses the final 180 observations per asset through 2025-10-14, three expanding validation folds of 20 observations, then a locked 20-observation holdout. These dates differ because the markets have different observation calendars. Each model sees the same folds within an asset. The matrix contains last value, drift, seasonal naive, exponential smoothing, Holt-Winters, ARIMA, SARIMA, Ridge, Lasso, Random Forest, XGBoost, RNN, LSTM and GRU. All use horizons 1, 5 and 20.

Ridge/Lasso standardize training lag features inside a scikit-learn pipeline. Tree models use lag windows without scaling. The PyTorch RNN, LSTM and GRU each take a 10-observation univariate window, one recurrent layer and a linear output. LSTM uses gated memory and GRU a smaller gated state; the vanilla RNN has no gates. The saved configuration limits them to two training epochs and four hidden units. Early stopping is available but this run is a compute-bounded comparison, not a broad architecture search. Simple ARIMA/SARIMA orders are fixed; the holdout is never used for tuning.

## Metrics and results

MAE averages absolute errors, RMSE emphasizes larger errors, sMAPE scales by forecast and actual magnitude, and MASE compares to an in-sample naive difference. Mean and standard deviation use the three validation folds. Runtime is training plus inference time. AIC/BIC apply to fitted statistical models; residual ACF and Ljung-Box p-values are saved separately. Only ARIMA/SARIMA supply intervals, so other coverage cells are blank.

{_markdown_table(selected, ['asset_id','horizon','model','rmse_mean','rmse_std','fold_cv','final_test_rmse','last_value_holdout_rmse','holdout_vs_last_value_pct','fit_seconds_mean','inference_seconds_mean','run_id'])}

![Relative validation RMSE](../phase3/relative_rmse.png)

![Fold variation by asset and horizon](../phase3/fold_variability.png)

The recommendation rule is fixed before viewing the holdout: a non-baseline model must lower both validation mean RMSE and mean plus one fold standard deviation by at least 5% versus last value. The selected rows above are generated from `recommendations.csv`. This rule favors simple models when apparent gains are small relative to fold variation. The `holdout_vs_last_value_pct` column exposes validation gains that reverse on the final period, particularly longer BTC and WTI forecasts.

## Residuals, uncertainty and complexity

{_markdown_table(diagnostics[diagnostics.model.isin(selected.model.unique())].head(20), ['asset_id','model','ljung_box_pvalue_mean','parameter_count'])}

ARIMA/SARIMA interval coverage is available for {len(intervals)} asset/model/horizon combinations in `comparison.csv`; coverage is measured on short fold windows and should not be read as calibrated long-run uncertainty.

{_markdown_table(intervals, ['asset_id','horizon','model','interval_coverage_mean','final_test_interval_coverage'])}

Runtime in the selected-model table and network parameter counts in saved diagnostics show the extra compute required by complex methods. There were {len(failures)} recorded model failures. Warnings retained in the runs include:

{_markdown_table(pd.DataFrame(warning_rows), ['asset','warning'])}

## Interpretation, limitations and threats to validity

There is no stable universal winner. For SPY and EUR/USD, the last-value forecast survives the improvement and variability threshold across horizons. Oil's validation folds support Ridge at one observation and XGBoost at longer horizons, but the longer-horizon holdout weakens that conclusion. BTC's 20-observation XGBoost selection similarly loses to last value on the holdout. Fold variability and one short locked test window limit ranking confidence.

The source data have different calendars, publication mechanisms and adjustment methods; raw RMSE cannot be compared across their price scales. The ECB reference fixing is not a continuous FX close. Yahoo adjustments can be revised. Spot oil has no roll discontinuity, but spot assessments can be revised or unavailable. Non-stationarity and regime changes threaten any fitted relationship. The experiment does not test transaction costs, trading rules, probabilistic calibration beyond statistical intervals or broader asset coverage. The BTC source license and SPY redistribution status constrain publication of underlying data.

## Reproducibility and conclusion

The report uses the four run IDs in `../phase3/run_index.json`. Each metric row in `../phase3/fold_metrics.csv` names its run ID, config, data manifest and prediction file. Regenerate with `python main.py four-markets --config configs/four_asset_full.json`, audit with `python main.py audit-four-markets`, and rebuild this report with `python main.py build-deliverables`. The dashboard reads the same runs. The evidence supports asset- and horizon-specific model choices with explicit holdout caveats, not a trading recommendation.
"""
    if (catalog.root / "SYNTHETIC_FIXTURE.txt").exists():
        result = "**SYNTHETIC OFFLINE FIXTURE — SOFTWARE CHECK ONLY. NO MARKET CONCLUSIONS.**\n\n" + result
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result, encoding="utf-8")
    return output


def build_presentation(catalog: ArtifactCatalog, output: Path) -> Path:
    """Create a concise Markdown slide deck; content comes from saved results."""
    if not _original_study(catalog):
        return _build_extended_presentation(catalog, output)
    recommendation = catalog.table("recommendations.csv")
    baseline = catalog.table("comparison.csv")
    source_lines = [f"- {run.asset_id}: {run.json('data_manifest.json')['source_name']}" for run in catalog.runs]
    picks = recommendation.pivot(index="asset_id", columns="horizon", values="model")
    pick_table = _markdown_table(picks.reset_index(), ["asset_id", 1, 5, 20])
    holds = recommendation.merge(baseline[baseline.model == "last_value"][["asset_id", "horizon", "final_test_rmse"]],
                                 on=["asset_id", "horizon"], suffixes=("", "_baseline"))
    reversal = holds[(holds.model != "last_value") & (holds.final_test_rmse > holds.final_test_rmse_baseline)]
    lines = ["# MarketCast Lab", "Four daily markets. Fourteen models. Three forecast horizons.",
             "\n---\n", "# Research question", "How does forecast performance change across asset class and horizon?",
             "Recursive predictions for the next 1, 5 and 20 observed sessions.",
             "\n---\n", "# Data and calendars", *source_lines,
             "BTC trades seven days. SPY, ECB reference FX and WTI spot retain their own observed dates.",
             "\n---\n", "# Evaluation design", "Three expanding validation folds and one locked 20-observation holdout per asset.",
             "The same model matrix and target timestamps apply within each asset.",
             "\n---\n", "# Model families", "Naive and seasonal baselines; exponential smoothing and ARIMA; lag regression and trees; PyTorch RNN, LSTM and GRU.",
             "Small networks and fixed statistical orders bound compute.",
             "\n---\n", "# Validation recommendations", pick_table,
             "Selection requires a 5% gain in mean RMSE and mean plus fold standard deviation versus last value.",
             "\n---\n", "# Stability matters", "See `../phase3/fold_variability.png` and `../phase3/relative_rmse.png`.",
             f"{len(reversal)} selected complex-model cases underperformed last value on the locked holdout.",
             "\n---\n", "# Limits", "One recent window, one instrument per class and limited tuning.",
             "SPY adjusted prices, ECB fixing and EIA spot assessments have different semantics.",
             "Statistical intervals do not establish calibrated uncertainty.",
             "\n---\n", "# Reproduce", "`python main.py four-markets --config configs/four_asset_full.json`",
             "`python main.py audit-four-markets`", "All displayed results trace to `../phase3/run_index.json`.",
             "\n---\n", "# Conclusion", "Choose by asset and horizon; inspect fold variation and holdout reversals.",
             "Historical analysis only. No trading advice."]
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = ["# SYNTHETIC OFFLINE FIXTURE", "Software check only. No market conclusions.", "\n---\n"] if (catalog.root / "SYNTHETIC_FIXTURE.txt").exists() else []
    output.write_text("\n".join(prefix + lines) + "\n", encoding="utf-8")
    return output


def _build_extended_powerpoint(catalog: ArtifactCatalog, output: Path) -> Path:
    from pptx import Presentation
    from pptx.util import Inches, Pt

    deck = Presentation()
    deck.slide_width, deck.slide_height = Inches(13.333), Inches(7.5)

    def add_slide(title: str, body: str) -> None:
        slide = deck.slides.add_slide(deck.slide_layouts[6])
        title_box = slide.shapes.add_textbox(Inches(.8), Inches(.5), Inches(11.8), Inches(.8))
        title_box.text_frame.text = title
        title_box.text_frame.paragraphs[0].font.size = Pt(30)
        box = slide.shapes.add_textbox(Inches(.9), Inches(1.6), Inches(11.5), Inches(5.3))
        box.text_frame.word_wrap = True
        box.text_frame.text = body
        for paragraph in box.text_frame.paragraphs:
            paragraph.font.size = Pt(19)

    warning = "Synthetic fixture: software check only." if (catalog.root / "SYNTHETIC_FIXTURE.txt").exists() else "Historical evaluation only."
    add_slide("MarketCast Lab", f"{len(catalog.runs)} daily series\n{warning}")
    for start in range(0, len(catalog.runs), 6):
        group = catalog.runs[start:start + 6]
        add_slide("Sources and targets", "\n".join(
            f"{run.asset_id}: {run.json('data_manifest.json')['target_semantics']}"
            for run in group))
    recommendations = catalog.table("recommendations.csv")
    for run in catalog.runs:
        selected = recommendations[recommendations.asset_id == run.asset_id]
        add_slide(run.asset_id, "\n".join(
            f"Horizon {int(row.horizon)}: {row.model}; validation RMSE {row.rmse_mean:.4g}; holdout RMSE {row.final_test_rmse:.4g}"
            for row in selected.itertuples()))
    add_slide("Interpretation", "Compare models within each target unit. Validation selects the model; holdout tests that choice.\n"
              "Inspect saved manifests for source hashes, calendar, and fold dates.\nHistorical analysis is not investment advice.")
    output.parent.mkdir(parents=True, exist_ok=True)
    deck.save(output)
    return output


def build_powerpoint(catalog: ArtifactCatalog, output: Path) -> Path:
    if not _original_study(catalog):
        return _build_extended_powerpoint(catalog, output)
    """Create a nine-slide offline deck from the same saved comparison tables."""
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor

    deck = Presentation()
    deck.slide_width, deck.slide_height = Inches(13.333), Inches(7.5)
    navy, teal, ink, muted = RGBColor(19, 43, 54), RGBColor(0, 115, 121), RGBColor(29, 43, 48), RGBColor(91, 107, 113)
    white = RGBColor(255, 255, 255)

    def textbox(slide, x, y, w, h, value, size=20, bold=False, color=ink):
        shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = shape.text_frame; tf.clear(); tf.word_wrap = True
        for number, line in enumerate(value.split("\n")):
            p = tf.paragraphs[0] if number == 0 else tf.add_paragraph()
            p.text = line; p.font.name = "Aptos"; p.font.size = Pt(size)
            p.font.bold = bold; p.font.color.rgb = color
            p.space_after = Pt(10)
        return shape

    def slide(title, subtitle=""):
        item = deck.slides.add_slide(deck.slide_layouts[6])
        item.background.fill.solid(); item.background.fill.fore_color.rgb = white
        band = item.shapes.add_shape(1, 0, 0, deck.slide_width, Inches(.14))
        band.fill.solid(); band.fill.fore_color.rgb = teal; band.line.fill.background()
        textbox(item, .75, .5, 11.8, .65, title, 30, True, navy)
        if subtitle:
            textbox(item, .78, 1.27, 11.8, .65, subtitle, 15, False, muted)
        textbox(item, .8, 7.08, 11.6, .22, "MarketCast Lab  ·  Educational analysis  ·  Source: saved Phase 3 runs", 9, False, muted)
        return item

    intro = slide("MarketCast Lab", "Four daily markets · fourteen models · three forecast horizons")
    if (catalog.root / "SYNTHETIC_FIXTURE.txt").exists():
        textbox(intro, .8, 1.45, 10.9, .5, "SYNTHETIC OFFLINE FIXTURE · NO MARKET CONCLUSIONS", 16, True, teal)
    textbox(intro, .8, 2.05, 10.9, 2.2, "How does forecast behavior change across asset class and horizon?", 33, True, navy)
    textbox(intro, .8, 5.6, 11.5, .8, "Historical evaluation, not investment advice", 18, False, teal)

    item = slide("Four series, four observation calendars")
    assets = [("BTC/USD", "Seven-day crypto close"), ("SPY", "Adjusted ETF close"),
              ("EUR/USD", "ECB reference fixing"), ("WTI Cushing", "EIA spot assessment")]
    for index, (name, meaning) in enumerate(assets):
        y = 1.85 + index * 1.13
        textbox(item, .9, y, 3.0, .5, name, 22, True, navy)
        textbox(item, 4.2, y, 7.8, .5, meaning, 20)
    textbox(item, .9, 6.52, 11.5, .4, "Horizon counts future observed sessions. Missing market dates stay missing.", 15, False, teal)

    item = slide("One defensible evaluation contract")
    textbox(item, .9, 1.8, 11.3, 3.8,
            "Three expanding validation folds per asset\nTwenty observations per fold and locked holdout\nSame target timestamps across models within an asset\nRecursive forecasts at 1, 5 and 20 observed sessions", 22)
    audited = audit_cross_market_report(catalog.root)
    textbox(item, .9, 6.12, 11.4, .55,
            f"{audited['metric_rows_reproduced']} saved metric rows reproduced from fold predictions", 19, True, teal)

    item = slide("Model families and controlled compute")
    textbox(item, .9, 1.8, 11.3, 4.6,
            "Baselines: last value, drift, seasonal naive\nStatistical: exponential smoothing, Holt-Winters, ARIMA, SARIMA\nLag models: Ridge, Lasso, Random Forest, XGBoost\nPyTorch sequences: RNN, LSTM, GRU\nSmall networks and fixed statistical orders bound runtime", 21)

    picks = catalog.table("recommendations.csv").pivot(index="asset_id", columns="horizon", values="model")
    item = slide("Recommendations differ by asset and horizon", "Selected on validation mean RMSE and fold variability")
    table = item.shapes.add_table(5, 4, Inches(.75), Inches(1.9), Inches(11.85), Inches(3.65)).table
    table.columns[0].width = Inches(4.05)
    for idx, label in enumerate(("Asset", "1", "5", "20")):
        table.cell(0, idx).text = label
    for row_index, run in enumerate(catalog.runs, start=1):
        table.cell(row_index, 0).text = run.asset_id
        for col_index, horizon in enumerate((1, 5, 20), start=1):
            table.cell(row_index, col_index).text = str(picks.loc[run.asset_id, horizon])
    for row in table.rows:
        for cell in row.cells:
            cell.margin_left = Inches(.14); cell.margin_top = Inches(.1)
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.name = "Aptos"; paragraph.font.size = Pt(16)
                paragraph.font.color.rgb = ink
    textbox(item, .8, 6.1, 11.8, .55, "Complex models need at least a 5% validation gain on mean and mean plus one standard deviation.", 16, False, teal)

    item = slide("Validation gains can reverse on holdout", "Fold spread and final-test results must be read together")
    item.shapes.add_picture(str(catalog.root / "relative_rmse.png"), Inches(1.65), Inches(1.55), width=Inches(10.0))
    textbox(item, .85, 6.68, 11.7, .28, "BTC and WTI longer-horizon XGBoost selections weakened on the locked holdout.", 13, False, teal)

    item = slide("Failure cases and limits")
    textbox(item, .9, 1.75, 11.4, 4.9,
            "Three folds and one short holdout leave ranking uncertainty.\nRecursive 20-step errors can compound.\nECB fixes and EIA spot assessments differ from exchange closes.\nBTC provenance and SPY redistribution remain unresolved.\nOne FX statistical fit produced a convergence warning.", 21)

    item = slide("Reproduce and inspect")
    textbox(item, .9, 1.86, 11.3, 2.6,
            "python main.py four-markets --config configs/four_asset_full.json\npython main.py audit-four-markets\npython main.py build-deliverables", 21)
    textbox(item, .9, 5.3, 11.4, .9, "The dashboard and report use the same four run IDs in artifacts/phase3/run_index.json.", 20, True, teal)

    item = slide("Conclusion")
    textbox(item, .9, 2.0, 11.3, 2.9,
            "Select by asset and horizon.\nInspect fold variation before trusting a small gain.\nTreat a holdout reversal as evidence of instability.", 25, True, navy)
    textbox(item, .9, 6.03, 11.1, .6, "Historical analysis only. No trading recommendation.", 17, False, teal)
    output.parent.mkdir(parents=True, exist_ok=True)
    deck.save(output)
    return output


def build_deliverables(index_root: str | Path = "artifacts/phase3", output_root: str | Path = "artifacts/phase4") -> Path:
    catalog = load_catalog(index_root)
    output = Path(output_root)
    if not output.is_absolute():
        output = catalog.root.parents[1] / output
    output.mkdir(parents=True, exist_ok=True)
    build_scientific_report(catalog, output / "scientific_report.md")
    build_presentation(catalog, output / "presentation.md")
    try:
        build_powerpoint(catalog, output / "presentation.pptx")
    except ImportError:
        pass  # Markdown slides remain available without the optional presentation extra.
    for name in ("relative_rmse.png", "fold_variability.png"):
        shutil.copy2(catalog.root / name, output / name)
    return output


def build_future_scenarios(index_root: str | Path = "artifacts/phase3", output_root: str | Path = "artifacts/phase4") -> Path:
    """Explicitly refit selected models to the saved study endpoint for scenario display."""
    catalog = load_catalog(index_root)
    output = Path(output_root)
    if not output.is_absolute():
        output = catalog.root.parents[1] / output
    output.mkdir(parents=True, exist_ok=True)
    recommendations = catalog.table("recommendations.csv")
    registry = default_registry()
    rows = []
    for run in catalog.runs:
        config = run.json("config.json")
        manifest = run.json("data_manifest.json")
        frame = run.source_data()
        if config["max_train_rows"]:
            frame = frame.iloc[-int(config["max_train_rows"]):]
        target = manifest["target_column"]
        values = frame[target].to_numpy(dtype=float)
        for selected in recommendations[recommendations.asset_id == run.asset_id].itertuples():
            params = dict(manifest["selected_model_parameters"].get(selected.model, {}))
            if selected.model in {"ridge", "lasso", "random_forest", "xgboost", "rnn", "lstm", "gru"}:
                params.setdefault("seed", config["seed"])
            model = registry.create(selected.model, params).fit(values)
            prediction = model.predict(int(selected.horizon))
            rows.extend({"asset_id": run.asset_id, "horizon": int(selected.horizon),
                         "step": step, "prediction": float(value), "model": selected.model,
                         "origin_timestamp": frame.timestamp.iloc[-1].isoformat(),
                         "source_run_id": run.run_id, "target": target}
                        for step, value in enumerate(prediction, start=1))
    path = output / "future_forecasts.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    (output / "future_manifest.json").write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_index": str(catalog.root / "run_index.json"),
        "method": "explicit refit on each asset's 180 observations through the saved cutoff; recursive scenario",
        "calendar_dates": "omitted; steps count observations without invented future holidays",
        "purpose": "speculative educational scenario, separate from historical backtest",
    }, indent=2), encoding="utf-8")
    return path
