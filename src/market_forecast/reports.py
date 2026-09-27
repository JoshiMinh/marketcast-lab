from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


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
        "# BTC experiment comparison\n\nGenerated only from saved run metrics.\n\n"
        + "```text\n" + comparison.to_string(index=False) + "\n```\n",
        encoding="utf-8",
    )
    return report
