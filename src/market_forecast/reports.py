from __future__ import annotations

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def build_comparison(run_dir: str | Path) -> pd.DataFrame:
    run = Path(run_dir)
    metrics = pd.read_csv(run / "metrics.csv")
    validation = metrics[metrics["partition"] == "validation"]
    grouped = validation.groupby(["model", "horizon"], as_index=False)
    comparison = grouped.agg(
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"), smape_mean=("smape", "mean"),
        mase_mean=("mase", "mean"), fit_seconds_mean=("fit_seconds", "mean"),
    )
    holdout = metrics[metrics["partition"] == "final_test"][["model", "horizon", "rmse", "mae"]]
    holdout = holdout.rename(columns={"rmse": "final_test_rmse", "mae": "final_test_mae"})
    return comparison.merge(holdout, on=["model", "horizon"], how="left")


def write_report(run_dir: str | Path) -> Path:
    run = Path(run_dir)
    comparison = build_comparison(run)
    comparison.to_csv(run / "comparison.csv", index=False)
    figure_path = run / "figures" / "rmse_comparison.png"
    pivot = comparison.pivot(index="model", columns="horizon", values="rmse_mean")
    pivot.plot(kind="bar", figsize=(10, 5), ylabel="Validation RMSE")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=140)
    plt.close()
    report = run / "report.md"
    report.write_text(
        "# Experiment comparison\n\nGenerated only from saved run metrics.\n\n"
        + "```text\n" + comparison.to_string(index=False) + "\n```\n",
        encoding="utf-8",
    )
    return report


def write_asset_eda(run_dir: str | Path, frame: pd.DataFrame, target: str, period: int) -> None:
    """Describe training/validation history only; the locked holdout stays unseen."""
    from market_forecast.analysis import series_diagnostics
    run = Path(run_dir)
    values = frame[target].astype(float)
    returns = values.pct_change().dropna()
    stats = series_diagnostics(values.to_numpy(), seasonal_period=period)
    with np.errstate(all="ignore"):
        return_stats = series_diagnostics(returns.to_numpy(), seasonal_period=period)
    weekly = ("Seven consecutive observations correspond to one crypto week."
              if frame.asset_class.iloc[0] == "crypto" else
              "Five observations approximate one business week; holidays make this only an exploratory proxy.")
    stationarity = ("ADF rejects a unit root at 5%, but KPSS should also be considered."
                    if stats["adf_pvalue"] < .05 else
                    "ADF does not reject a unit root at 5%; first differencing is a reasonable ARIMA candidate.")
    decision = (f"{stationarity} {weekly} Evaluate level prices so all models share target semantics. "
                "Return volatility is descriptive; scaling and model transforms are fit within each training fold. "
                "The seasonal decomposition is exploratory and does not establish a stable calendar effect.")
    payload = {
        "observations": len(frame), "start": frame.timestamp.iloc[0].isoformat(),
        "end": frame.timestamp.iloc[-1].isoformat(), "target": target,
        "price": stats, "returns": return_stats,
        "rolling_mean_20_latest": float(values.rolling(20).mean().iloc[-1]),
        "rolling_volatility_20_latest": float(returns.rolling(20).std().iloc[-1]),
        "return_std": float(returns.std()),
        "seasonal_period": period,
        "decision": decision,
    }
    (run / "eda.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (run / "eda.md").write_text(f"# {frame.asset_id.iloc[0]} EDA\n\n{decision}\n\n"
                                  f"ADF p={stats['adf_pvalue']:.4g}; KPSS p={stats['kpss_pvalue']:.4g}; "
                                  f"return standard deviation={returns.std():.4g}. See eda.json and figures/eda.png.\n",
                                  encoding="utf-8")
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(frame.timestamp, values); axes[0].set_ylabel(target)
    axes[1].plot(frame.timestamp.iloc[1:], returns); axes[1].set_ylabel("Return")
    axes[2].plot(frame.timestamp, values.rolling(20).mean(), label="20-observation mean")
    axes[2].plot(frame.timestamp, values.rolling(20).std(), label="20-observation std")
    axes[2].legend(); fig.tight_layout()
    fig.savefig(run / "figures" / "eda.png", dpi=130); plt.close(fig)


def write_cross_market_report(output: str | Path) -> Path:
    destination = Path(output)
    runs = json.loads((destination / "run_index.json").read_text(encoding="utf-8"))
    failures, fold_rows, residual_rows = [], [], []
    for entry in runs:
        run = Path(entry["run_dir"])
        manifest = json.loads((run / "data_manifest.json").read_text(encoding="utf-8"))
        metric = pd.read_csv(run / "metrics.csv")
        for key, value in entry.items():
            metric[key] = value
        metric["prediction_file"] = str(run / manifest["prediction_file"])
        metric["config_file"] = str(run / "config.json")
        metric["manifest_file"] = str(run / "data_manifest.json")
        fold_rows.append(metric)
        failure = json.loads((run / "failures.json").read_text(encoding="utf-8"))
        failures.extend([{"asset_id": entry["asset_id"], "run_id": entry["run_id"], **item} for item in failure])
        diagnostics = json.loads((run / "diagnostics.json").read_text(encoding="utf-8"))
        residual_rows.extend([{"asset_id": entry["asset_id"], "run_id": entry["run_id"],
                               "model": item["model"], "fold": item["fold"],
                               "ljung_box_pvalue": item.get("ljung_box_pvalue"),
                               "aic": item.get("aic"), "bic": item.get("bic"),
                               "parameter_count": item.get("parameter_count")}
                              for item in diagnostics["models"]])
    all_metrics = pd.concat(fold_rows, ignore_index=True)
    all_metrics.to_csv(destination / "fold_metrics.csv", index=False)
    pd.DataFrame(residual_rows).to_csv(destination / "residual_diagnostics.csv", index=False)
    keys = ["asset_id", "asset_class", "run_id", "horizon", "model"]
    validation = all_metrics[all_metrics.partition == "validation"]
    summary = validation.groupby(keys, as_index=False).agg(
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"), smape_mean=("smape", "mean"),
        mase_mean=("mase", "mean"), fit_seconds_mean=("fit_seconds", "mean"),
        inference_seconds_mean=("inference_seconds", "mean"),
        interval_coverage_mean=("interval_coverage", "mean"), completed_folds=("fold", "nunique"))
    final = all_metrics[all_metrics.partition == "final_test"][keys + ["rmse", "mae", "interval_coverage"]]
    final = final.rename(columns={"rmse": "final_test_rmse", "mae": "final_test_mae",
                                  "interval_coverage": "final_test_interval_coverage"})
    summary = summary.merge(final, on=keys, how="left")
    summary.to_csv(destination / "comparison.csv", index=False)
    pd.DataFrame(failures, columns=["asset_id", "run_id", "model", "fold", "error_type", "message"]).to_csv(
        destination / "failures.csv", index=False)
    baseline = summary.loc[summary.model == "last_value", ["asset_id", "horizon", "rmse_mean"]].rename(columns={"rmse_mean": "baseline_rmse"})
    relative = summary.merge(baseline, on=["asset_id", "horizon"])
    relative["rmse_ratio"] = relative.rmse_mean / relative.baseline_rmse
    fig, ax = plt.subplots(figsize=(12, 6))
    relative.pivot_table(index=["asset_id", "horizon"], columns="model", values="rmse_ratio").plot.bar(ax=ax)
    ax.set_ylabel("Validation RMSE / last-value RMSE"); fig.tight_layout()
    fig.savefig(destination / "relative_rmse.png", dpi=130); plt.close(fig)
    fig, axes = plt.subplots(4, 3, figsize=(16, 13), sharex=True)
    asset_order = [entry["asset_id"] for entry in runs]
    for row_index, asset_id in enumerate(asset_order):
        for col_index, horizon in enumerate((1, 5, 20)):
            ax = axes[row_index, col_index]
            subset = validation[(validation.asset_id == asset_id) & (validation.horizon == horizon)]
            fold_base = subset[subset.model == "last_value"][["fold", "rmse"]]
            base_map = dict(zip(fold_base["fold"], fold_base["rmse"]))
            for model_name, group in subset.groupby("model"):
                ax.plot(group.fold, [value / base_map[fold] for fold, value in zip(group.fold, group.rmse)],
                        marker=".", alpha=.65, linewidth=.8, label=model_name)
            ax.axhline(1, color="black", linewidth=.6)
            ax.set_title(f"{asset_id}, h={horizon}")
            ax.set_xticks([1, 2, 3])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=8)
    fig.text(.01, .5, "Fold RMSE / last-value fold RMSE", rotation=90, va="center")
    fig.tight_layout(rect=[.025, .08, 1, 1])
    fig.savefig(destination / "fold_variability.png", dpi=130); plt.close(fig)
    recommendations = summary[summary.completed_folds == 3].copy()
    recommendations["selection_score"] = recommendations.rmse_mean + recommendations.rmse_std.fillna(0)
    base_score = recommendations[recommendations.model == "last_value"][["asset_id", "horizon", "selection_score", "rmse_mean"]].rename(
        columns={"selection_score": "baseline_score", "rmse_mean": "baseline_mean"})
    recommendations = recommendations.merge(base_score, on=["asset_id", "horizon"])
    recommendations = recommendations[(recommendations.model == "last_value") |
                                  ((recommendations.selection_score <= .95 * recommendations.baseline_score) &
                                   (recommendations.rmse_mean <= .95 * recommendations.baseline_mean))]
    recommendations = recommendations.sort_values("selection_score").groupby(["asset_id", "horizon"]).head(1)
    recommendations.to_csv(destination / "recommendations.csv", index=False)
    lines = ["# Phase 3 cross-market evidence", "", "A non-baseline model is selected only if validation mean RMSE and mean plus one fold standard deviation both improve by at least 5% over last value. The locked holdout is shown only after selection.",
             "Horizon is the next 1, 5, or 20 observed sessions, so wall-clock spans differ by market.",
             "The 20-observation holdout and preceding three 20-observation validation blocks use identical row counts; dates are in each run manifest.",
             "", "## Recommendations", "", "```text", recommendations[["asset_id", "horizon", "model", "rmse_mean", "rmse_std", "final_test_rmse"]].to_string(index=False), "```",
             "", "## Stability and failure cases", "",
             "The fold standard deviations in recommendations.csv are often large relative to mean RMSE; treat small differences as inconclusive.",
             "Compare selected final_test_rmse with last-value holdout in comparison.csv: several validation gains reverse on the holdout, particularly longer oil and BTC forecasts.",
             "A 20-observation recursive forecast can compound model error; one-step results are a different strategy and are not pooled here.",
             "", "## Failures", "", f"{len(failures)} recorded run failures. See failures.csv and each run's warnings.json for convergence or storage warnings.", "", "## Traceability", "",
             "Every row in fold_metrics.csv carries run_id, config_file, manifest_file and prediction_file."]
    (destination / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
    return destination


def audit_cross_market_report(output: str | Path) -> dict[str, int]:
    """Check that each saved metric is reproducible from its fold predictions."""
    destination = Path(output)
    runs = json.loads((destination / "run_index.json").read_text(encoding="utf-8"))
    checked = 0
    for entry in runs:
        run = Path(entry["run_dir"])
        manifest = json.loads((run / "data_manifest.json").read_text(encoding="utf-8"))
        predictions_path = run / manifest["prediction_file"]
        predictions = (pd.read_parquet(predictions_path) if predictions_path.suffix == ".parquet"
                       else pd.read_csv(predictions_path))
        metrics = pd.read_csv(run / "metrics.csv")
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        if predictions.asset_id.unique().tolist() != [entry["asset_id"]]:
            raise AssertionError(f"Cross-asset predictions in {run}")
        expected = {(model, horizon, "validation", fold)
                    for model in config["models"] for horizon in config["horizons"]
                    for fold in range(1, config["folds"] + 1)}
        expected |= {(model, horizon, "final_test", 0)
                     for model in config["models"] for horizon in config["horizons"]}
        observed = set(zip(metrics.model, metrics.horizon, metrics.partition, metrics.fold))
        if len(metrics) != len(expected) or observed != expected:
            raise AssertionError(f"Incomplete model/horizon/fold matrix in {run}")
        alignment = predictions.groupby(["partition", "fold", "horizon", "step"])["timestamp"].nunique()
        if (alignment != 1).any():
            raise AssertionError(f"Models disagree on target dates in {run}")
        for row in metrics.itertuples():
            selected = predictions[(predictions.model == row.model) &
                                   (predictions.fold == row.fold) &
                                   (predictions.partition == row.partition) &
                                   (predictions.horizon == row.horizon)]
            if len(selected) != row.horizon:
                raise AssertionError(f"Prediction count mismatch in {run}: {row.model}, {row.horizon}")
            error = selected.actual.to_numpy() - selected.prediction.to_numpy()
            if not np.isclose(np.sqrt(np.mean(error ** 2)), row.rmse, atol=1e-5):
                raise AssertionError(f"RMSE mismatch in {run}: {row.model}, {row.horizon}")
            if not np.isclose(np.mean(np.abs(error)), row.mae, atol=1e-5):
                raise AssertionError(f"MAE mismatch in {run}: {row.model}, {row.horizon}")
            checked += 1
    return {"runs": len(runs), "metric_rows_reproduced": checked}
