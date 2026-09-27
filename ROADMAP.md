# MarketCast Lab — Four-Phase Implementation Roadmap

## Mission

Convert the existing cryptocurrency forecasting prototype into a reproducible academic platform that compares time-series forecasting methods across cryptocurrency, equities, foreign exchange, and crude oil.

The final project must answer:

> How do statistical, machine-learning, and deep-learning forecasting methods compare across financial asset classes with different calendars, volatility, trend, and seasonality?

This is an analysis and comparison system, not a trading or financial-advice system.

## How to use this roadmap

Complete the phases in order. Each phase below is a self-contained prompt for an AI coding agent. Start a fresh agent session for each phase and give it the entire prompt. Do not begin the next phase until the current phase's acceptance checks pass.

Agents must inspect the current repository before editing, preserve useful legacy behavior, avoid unrelated changes, and leave the repository runnable. If a required data source cannot legally be committed, add a retrieval adapter, manifest, and documentation instead.

## Progress

- [x] Phase 1 — Establish a trustworthy single-asset foundation.
- [x] Phase 2 — Build evaluation and all required model families.
- [ ] Phase 3 — Generalize to four markets and produce defensible results.
- [ ] Phase 4 — Deliver the dashboard, report, and reproducibility package.

Phase 1 and Phase 2 were verified with the automated test suite and reproducible BTC experiment configurations. Generated run directories remain local and are excluded from version control; configurations, code, tests, and reporting logic are committed.

## Fixed project contract

These requirements apply to every phase:

- Use daily data initially for one representative asset per class: BTC/USD, SPY or AAPL, EUR/USD, and a documented crude-oil series.
- Keep observations separated by `asset_id`; never calculate features across asset boundaries.
- Preserve chronological order. Fit scalers, transformations, feature selectors, and models only on the applicable training fold.
- Reserve a locked final test period. Never use it for feature selection, tuning, or model selection.
- Evaluate horizons 1, 5, and 20 with at least three comparable walk-forward folds.
- Include last-value, drift, and seasonal-naive baselines where meaningful.
- Report MAE, RMSE, sMAPE, MASE, and runtime. Report AIC/BIC for applicable statistical models and residual diagnostics where possible.
- Implement RNN, LSTM, and GRU in PyTorch, not TensorFlow.
- Make experiments deterministic where libraries permit and record seeds, dependencies, data range, fold boundaries, parameters, forecast strategy, and code revision.
- Keep notebooks and Streamlit as consumers of shared `src/` code. They must not contain separate training pipelines.
- Prefer a smaller valid experiment matrix over broad but methodologically weak tuning.

The canonical long-format data fields are:

`asset_id`, `symbol`, `asset_class`, `timestamp`, `frequency`, `open`, `high`, `low`, `close`, `adjusted_close`, `volume`, `currency`, `provider`, and `retrieved_at`.

Each immutable experiment run should produce:

```text
artifacts/runs/<run_id>/
├── config.json
├── data_manifest.json
├── environment.json
├── metrics.csv
├── fold_predictions.parquet
├── diagnostics.json
├── model/
└── figures/
```

---

## Phase 1 prompt — Establish a trustworthy single-asset foundation

```text
You are implementing Phase 1 of MarketCast Lab in the current repository.

GOAL
Turn the existing prototype into a tested, reproducible, leakage-safe BTC forecasting foundation. Do not add other markets or advanced models yet. Preserve working legacy entry points where practical, but shared library code becomes the source of truth.

FIRST, INSPECT
- Read README.md, ROADMAP.md, requirements.txt, main.py, and every file under src/.
- Run git status and preserve all user changes.
- Run the existing CLI/dashboard smoke path if feasible and record current failures or behavioral limitations.
- Identify where rows from different symbols are mixed, where features are calculated before asset selection, and where preprocessing is duplicated.

IMPLEMENT STEP BY STEP
1. Add a supported Python version and pyproject.toml with runtime and development dependencies. Keep requirements.txt compatible temporarily if existing entry points need it.
2. Create the `src/market_forecast/` package with focused modules for configuration, data schema/validation, features, baselines, evaluation, experiments, and dashboard integration. Migrate incrementally rather than deleting working code upfront.
3. Normalize the bundled crypto data into the canonical long-format contract. Create stable `asset_id` values, sort by timestamp, reject duplicate asset/timestamp keys, and produce a machine-readable quality report.
4. Select one asset before truncation, transformation, splitting, lagging, rolling calculations, or model training. Make every feature operation explicitly per-asset and past-only.
5. Implement last-value and drift forecasts plus seasonal naive only when a defensible seasonal period exists.
6. Implement chronological train/validation/final-test boundaries. All transforms must expose fit/transform behavior so tests can prove they are fit only on training data.
7. Remove or route around duplicated forecasting/preprocessing logic in the CLI and Streamlit app. UI code should call shared functions.
8. Add unit tests for canonical schema validation, sorting, duplicate detection, cross-asset lag isolation, past-only rolling windows, no split overlap, and train-only scaling.
9. Add one integration test that loads the bundled data, selects BTC, runs a baseline forecast, and writes a minimal result artifact.
10. Update README.md with setup commands, current limitations, and the exact trustworthy BTC workflow.

REQUIRED OUTPUTS
- pyproject.toml and a package under src/market_forecast/
- Canonical schema and validation report code
- Leakage-safe per-asset feature code
- Naive baseline implementations
- Unit and integration tests
- Reproducible BTC smoke command
- A short migration note identifying legacy code still awaiting replacement

ACCEPTANCE CHECKS
- A clean environment can install the project and run tests using documented commands.
- BTC is selected before any feature engineering or split.
- Tests fail if a lag/rolling value crosses an asset or temporal boundary.
- No scaler or transform sees validation/final-test values during fit.
- One deterministic BTC baseline run completes and saves its configuration, boundaries, predictions, and metrics.
- Existing user-facing entry points either still work or clearly redirect to their replacements.

SCOPE CONTROL
Do not add equity, forex, or oil providers. Do not build PyTorch networks. Do not polish the dashboard. Finish correctness and tests first.

At completion, report changed files, commands run, test results, remaining risks, and the exact command the Phase 2 agent should run first.
```

---

## Phase 2 prompt — Build evaluation and all required model families

```text
You are implementing Phase 2 of MarketCast Lab. Phase 1 should already provide a canonical, tested, single-asset BTC pipeline. Verify its acceptance checks before editing; repair regressions if necessary.

GOAL
Create one reproducible experiment engine that fairly evaluates baselines, statistical models, lag-based machine learning, and PyTorch sequence models on identical BTC folds and horizons.

IMPLEMENT STEP BY STEP
1. Implement configurable expanding-window backtesting first; optionally add rolling windows afterward. Use at least three chronological validation folds and a locked final test period.
2. Explicitly distinguish one-step forecasts using observed history from recursive or direct multi-step forecasts. Store the strategy with every run and align predictions exactly to target timestamps for horizons 1, 5, and 20.
3. Implement fold-aware MAE, MSE, RMSE, sMAPE, MASE, mean error, training time, and inference time. Guard MAPE around zero. Aggregate mean and variability without discarding fold-level results.
4. Add residual plots/data, residual ACF, Ljung-Box testing, and interval coverage when a model supplies intervals.
5. Define a shared ForecastModel adapter contract (`fit`, `predict`, `get_params`, `save`) and a model registry. The evaluator must not contain large model-specific branches.
6. Add statistical analysis: rolling statistics, decomposition, ADF/KPSS, ACF/PACF, exponential smoothing/Holt-Winters, ARIMA, and SARIMA when diagnostics justify seasonality. Select candidates with validation evidence plus AIC/BIC, never final-test performance.
7. Add past-only lag, rolling, trend, calendar, and volatility features. Use scikit-learn Pipeline/ColumnTransformer behavior so preprocessing is fit within each fold. Implement Ridge or Lasso, Random Forest, and XGBoost or LightGBM (choose one and document why).
8. Replace TensorFlow sequence training with reusable PyTorch window datasets and vanilla RNN, LSTM, and GRU models. Add deterministic seeds, CPU support, device selection, early stopping, checkpoints, learning curves, and parameter counts.
9. Use bounded, documented tuning spaces. Tune only on validation folds. Provide fast smoke configurations for CI and fuller experiment configurations for final runs.
10. Implement immutable run artifacts with config, data manifest, environment, fold predictions, metrics, diagnostics, models, figures, warnings, and failures. Add CLI commands to run and compare experiments.
11. Add unit tests for folds, metrics, forecast alignment, window generation, artifact round trips, and model registry behavior. Add CPU-only integration smoke tests for one statistical, one scikit-learn, and each PyTorch architecture.
12. Create an analysis/report generator from saved artifacts; notebooks may demonstrate results but must call package functions.

REQUIRED OUTPUTS
- Shared model interface and registry
- Walk-forward backtester and locked-test workflow
- Statistical, ML, RNN, LSTM, and GRU implementations
- Fold-aware metrics and diagnostics
- Immutable run artifacts and comparison command
- Fast test configs and full BTC experiment configs
- Generated BTC comparison table/figures from artifacts

ACCEPTANCE CHECKS
- Every model is evaluated on the same stored fold boundaries and target timestamps.
- Re-running a smoke configuration with the same seed produces equivalent splits and metrics within documented numerical tolerance.
- Hyperparameter selection cannot access the final test set.
- A complete BTC run includes baselines, exponential smoothing, ARIMA/SARIMA when justified, Ridge/Lasso, Random Forest, boosting, RNN, LSTM, and GRU.
- Result tables report fold mean/variability, final holdout performance, runtime, and diagnostic availability.
- Tests detect intentionally introduced leakage or forecast misalignment.

SCOPE CONTROL
Do not broaden to other markets or redesign Streamlit. Optimize for methodological correctness and a manageable CPU-capable experiment workflow.

At completion, report changed files, experiment/test commands, runtime expectations, test results, the best-supported BTC findings without overstating them, and prerequisites for Phase 3.
```

---

## Phase 3 prompt — Generalize to four markets and produce defensible results

```text
You are implementing Phase 3 of MarketCast Lab. Phase 2 should already run the full model contract on BTC. Verify that shared folds, leakage tests, and artifact reproduction pass before adding providers.

GOAL
Run the same defensible experiment contract on one crypto, one equity, one forex pair, and one crude-oil series, then explain how model behavior changes by asset class and forecast horizon.

IMPLEMENT STEP BY STEP
1. Add provider adapters behind a shared provider interface for the selected equity, forex, and oil sources. Keep the CSV provider. Store source URL/name, retrieval time, license/redistribution status, symbol mapping, timezone, currency, and adjustment method in manifests.
2. Select final instruments: BTC/USD; SPY or AAPL; EUR/USD; and a documented oil spot benchmark or continuous futures series. If using futures, explain roll construction and identify roll-related jumps.
3. Normalize provider output to the canonical schema without erasing market-specific calendars. Do not fill exchange holidays or weekends as if they were ordinary observations. Use adjusted equity prices when appropriate and document target semantics.
4. Generate a quality report per asset: row count/range, duplicate keys, missing values/timestamps, frequency consistency, invalid OHLC values, outlier flags, provenance, and transformations.
5. Produce EDA per asset: price/returns, rolling statistics, trend/seasonal decomposition when justified, ACF/PACF, ADF/KPSS, volatility observations, and a written decision for transformations/seasonal periods.
6. Define common comparable folds/horizons while respecting each asset's observation calendar. Record exact boundaries. If a common setting is invalid for one market, document and justify the smallest necessary exception.
7. Run the required experiment matrix for horizons 1, 5, and 20: last value, drift, seasonal naive when valid, exponential smoothing/Holt-Winters, ARIMA/SARIMA when valid, Ridge/Lasso, Random Forest, boosting, PyTorch RNN, LSTM, and GRU.
8. If compute is constrained, reduce tuning combinations or training epochs with early stopping; do not remove required model families. Preserve smoke/full configurations and record failures instead of silently excluding them.
9. Generate cross-market tables and figures by asset, class, horizon, model, and fold. Include mean/variability, final holdout, runtime, residual diagnostics, and interval coverage where available.
10. Analyze strengths, weaknesses, failure cases, stability, and complexity. Recommend a model per asset and horizon based on evidence; do not declare one universal winner.
11. Add provider/normalization integration tests and a small four-asset end-to-end regression fixture with fixed expected split dates and metric tolerances.

REQUIRED OUTPUTS
- Four documented provider/data paths and manifests
- Four data-quality reports and EDA outputs
- Complete saved experiment artifacts for the minimum matrix
- Cross-market comparison tables and figures
- Evidence-based model recommendations by asset/horizon
- A limitations note covering data source, calendars, oil rolls, uncertainty, compute, and non-stationarity

ACCEPTANCE CHECKS
- One crypto, equity, forex, and oil series passes the same schema and evaluation interfaces.
- No feature or model run crosses assets.
- Calendar handling is explicit and no artificial holiday observations are introduced.
- Every reported number traces to a run ID, config, data manifest, and fold prediction file.
- Comparisons use common folds/horizons or clearly label justified exceptions.
- Conclusions include variability and failure cases, not only the lowest mean RMSE.

SCOPE CONTROL
Use one asset per class and daily frequency. Do not add intraday data, live trading, portfolio optimization, global panel models, Transformers, GARCH, or broad asset coverage unless all acceptance checks are already complete.

At completion, report data sources, licenses/redistribution constraints, commands and runtimes, failed runs, coverage of the required matrix, key findings, and exact artifacts needed by Phase 4.
```

---

## Phase 4 prompt — Deliver the dashboard, report, and reproducibility package

```text
You are implementing Phase 4, the final delivery phase of MarketCast Lab. Treat saved Phase 3 artifacts as the source of truth. Verify traceability and required matrix coverage before changing presentation code.

GOAL
Turn the validated experiment system and results into a clear Streamlit application, scientific report, presentation/demo, and clean reproducibility workflow suitable for course assessment.

IMPLEMENT STEP BY STEP
1. Audit the repository against this roadmap and create a gap checklist. Resolve missing correctness/evidence items before visual polish.
2. Refactor the Streamlit app into: Data Explorer, Time-Series Analysis, Experiment Setup, Backtest Results, Model Comparison, Future Forecast, and Experiment History.
3. Make the dashboard read validated datasets and immutable run artifacts through shared package functions. It must not retrain implicitly during rendering or duplicate metrics/forecast logic.
4. Show fold-aware forecasts, mean and variability of metrics, residual diagnostics, interval coverage, runtime, run metadata, and limitations. Clearly separate historical backtests from speculative future forecasts and display a non-advisory disclaimer.
5. Make comparison views filterable by asset, asset class, horizon, fold, and model. Present stability and complexity alongside accuracy; explain why the recommended model may differ by asset/horizon.
6. Generate report tables and figures directly from saved artifacts. Write the scientific report with: problem/research question, time-series concepts, data/provenance, cleaning, transformations, EDA, validation design, models, tuning, metrics, results, diagnostics, cross-market interpretation, limitations, threats to validity, conclusions, and recommendations.
7. Explicitly define trend, seasonality, cycles, noise, stationarity, target, horizon, and forecast strategy. Explain RNN/LSTM/GRU architecture choices and why preprocessing avoids leakage.
8. Prepare a concise presentation and a short deterministic demo scenario that loads existing artifacts. Include fallback screenshots/figures in case training or network access is unavailable.
9. Finish README setup, data acquisition, experiment, report, dashboard, and troubleshooting instructions. Add data dictionary, source/license register, configuration reference, and expected runtime/hardware notes.
10. Run unit, integration, regression, CLI, and dashboard startup smoke tests from a clean environment. Run the documented reproduction path and fix discrepancies.
11. Verify every claim and displayed number against saved artifacts. Remove stale TensorFlow instructions/artifacts or clearly label them as legacy and excluded from results.
12. Produce a final assessment checklist mapped to: data analysis (30%), model building (30%), and evaluation/presentation (40%).

REQUIRED OUTPUTS
- Artifact-driven Streamlit dashboard
- Final scientific report with generated tables/figures
- Presentation and deterministic demo script/scenario
- Complete setup, data, experiment, and reproduction documentation
- Data dictionary and source/license register
- Final automated test and reproducibility results
- Rubric/evidence checklist

ACCEPTANCE CHECKS
- Another student can follow README instructions from a clean clone, obtain permitted data or use documented fixtures, run the smoke workflow, build the report, and launch the dashboard.
- The dashboard reproduces the report's principal results from the same run IDs.
- All required course concepts, model families, horizons, metrics, diagnostics, and four asset classes are evidenced.
- Backtests and future forecasts are visually and textually distinct.
- Every substantive conclusion links to saved results; no manually copied metric is the sole source of a claim.
- The final recommendation is per asset/horizon and acknowledges uncertainty, instability, and non-advisory use.

SCOPE CONTROL
Do not add new model families or markets during final delivery unless needed to close a documented requirement. Prioritize reproducibility, interpretation, and a reliable demo over extra features.

At completion, report final deliverables, reproduction commands, test results, known limitations, rubric coverage, and any claim that still lacks sufficient evidence.
```

## Suggested execution order and stopping rules

1. **Phase 1 — Trust the data path.** Stop if asset/time leakage tests do not pass.
2. **Phase 2 — Trust the comparison engine.** Stop if folds differ unintentionally between models or the final test participates in tuning.
3. **Phase 3 — Trust the cross-market evidence.** Stop if provenance, calendars, or run traceability are incomplete.
4. **Phase 4 — Communicate and reproduce.** Stop before submission if a clean-clone smoke run or claim-to-artifact audit fails.

If time is short, reduce tuning breadth, dataset length, or optional visualizations. Do not remove required model families, chronological validation, locked testing, core diagnostics, or traceability.

## Final definition of done

- Four representative daily market series use the documented common schema.
- Tests protect asset boundaries, temporal boundaries, transformations, folds, metrics, serialization, and representative model runs.
- Naive, statistical, lag-based ML, RNN, LSTM, and GRU models run through one evaluation contract.
- Results cover horizons 1, 5, and 20 across at least three walk-forward folds and one locked final test.
- Statistical assumptions, residuals, runtime, uncertainty availability, and fold variability are reported.
- Every result is traceable to configuration, data manifest, environment, predictions, and run ID.
- The dashboard consumes artifacts and distinguishes evaluation from future speculation.
- The report compares markets and model families, documents limitations, and recommends models by asset/horizon.
- A clean, documented smoke workflow reproduces the core evidence.
