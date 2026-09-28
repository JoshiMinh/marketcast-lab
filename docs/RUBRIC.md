# Assessment evidence checklist

## Data analysis (30%)

- [x] Four documented source paths, license statuses and target meanings: `docs/SOURCES.md`, four `data_manifest.json` files.
- [x] Canonical schema, source calendars, missingness and outlier flags: `docs/DATA_DICTIONARY.md`, four `quality_report.json` files.
- [x] Price/return EDA, rolling statistics, ACF/PACF, ADF/KPSS and transformation decisions: four `eda.json`, `eda.md` and `figures/eda.png` files.

## Model building (30%)

- [x] Common model registry and 14 required families: `src/market_forecast/models/`, `configs/four_asset_full.json`.
- [x] Three expanding folds, locked holdout, horizons 1/5/20, training-only transforms and asset separation: `src/market_forecast/evaluation/`, tests, fold manifests.
- [x] Bounded CPU configuration, saved parameters and failure/warning records: run configs, diagnostics and `warnings.json`/`failures.json`.

## Evaluation and presentation (40%)

- [x] Traceable fold and holdout metrics, variability, runtime, residuals and interval coverage: `artifacts/phase3/` tables and figures.
- [x] Per-asset and per-horizon recommendations with holdout reversals: `recommendations.csv`, generated scientific report.
- [x] Seven-view artifact-only dashboard, explicit scenario forecast, presentation and demo: `src/streamlit.py`, `artifacts/phase4/`, `docs/DEMO.md`.
- [x] Reproduction audit: `python main.py audit-four-markets` checks all 672 saved MAE/RMSE rows.

Open evidence limits: BTC upstream provenance and SPY raw redistribution rights are unresolved. Three folds and one short holdout provide limited stability evidence. The FX convergence warning and incomplete interval coverage should be visible in assessment discussion.
