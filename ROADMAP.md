# Multi-Market Time Series Forecasting Lab - Roadmap

## 1. Project decision

This repository should be **converted and upgraded**, not discarded. The current crypto application already provides a useful Streamlit interface, command-line entry point, plotting code, and initial ARIMA/Prophet/LSTM/GRU implementations. Those parts should become the starting point for a broader, academically rigorous time-series forecasting project.

The former name, `crypto-price-forecast-ml`, was too narrow for the intended scope.

### Recommended name

**Repository:** `marketcast-lab`  
**Application:** **MarketCast Lab**  
**Academic title:** **MarketCast Lab: A Unified Framework for Time Series Forecasting Across Financial Markets**

Alternative repository names:

- `financial-time-series-lab`
- `cross-market-forecasting`
- `unified-time-series-forecasting`

`marketcast-lab` communicates a cross-market forecasting laboratory without claiming that one universal model can reliably predict every market.

## 2. Project vision

Build a reproducible platform that imports time series from several financial markets, analyzes their statistical characteristics, trains multiple forecasting families, evaluates them with leakage-safe backtesting, and explains which approaches work best for each market and forecast horizon.

The project should support representative assets from:

- Cryptocurrency
- Stocks or stock indices
- Foreign exchange
- Commodities, especially crude oil

The central research question is:

> How do statistical, machine-learning, and deep-learning forecasting methods compare across financial asset classes with different calendars, volatility, trend, and seasonality characteristics?

This is a comparison and analysis system, not a trading recommendation system.

## 3. Course alignment

The finished project must demonstrate all major subject outcomes.

| Course area | Required evidence in the project |
| --- | --- |
| Time-series fundamentals | Explain trend, seasonality, cycles, noise, stationarity, and forecast horizons. |
| Data preparation | Detect missing timestamps and values; handle duplicates and outliers; apply scaling, log/Box-Cox transforms, smoothing, and differencing where justified. |
| Exploratory analysis | Plot series and returns; perform decomposition; inspect rolling statistics, ACF, and PACF. |
| Statistical forecasting | Implement naive baselines, exponential smoothing, AR/ARIMA, and SARIMA where seasonality exists. |
| Machine learning | Generate lag and rolling-window features; compare Ridge/Lasso, Random Forest, and a boosting model. |
| Deep learning | Implement RNN, LSTM, and GRU in PyTorch, including multi-step forecasting. |
| Correct evaluation | Use chronological splits and walk-forward or `TimeSeriesSplit` validation without future leakage. |
| Metrics and diagnostics | Report MAE, MSE, RMSE, MAPE/sMAPE, MASE, AIC/BIC where applicable, residual diagnostics, and runtime. |
| Model comparison | Compare models by asset, asset class, horizon, and fold; explain strengths and limitations. |
| Communication | Provide a clear dashboard, reproducible notebook/report, experiment tables, conclusions, and documented limitations. |

Because evaluation, comparison, explanation, and model selection are central grading concerns, dashboard polish must not take priority over methodological correctness.

## 4. Scope

### Minimum viable academic scope

Use one representative instrument from each market:

| Asset class | Suggested instrument | Example symbol | Important consideration |
| --- | --- | --- | --- |
| Crypto | Bitcoin | `BTC-USD` | Trades continuously, 7 days per week. |
| Stock/index | S&P 500 ETF or Apple | `SPY` or `AAPL` | Exchange calendar, holidays, adjusted prices. |
| Forex | Euro/US dollar | `EURUSD` | Approximately 24/5, provider-dependent timestamps. |
| Commodity | Crude oil | `CL` continuous futures or a documented spot benchmark | Futures rolls can create artificial price jumps. |

Start with **daily frequency** for a fair, manageable comparison. Intraday data can be an optional extension after the daily pipeline is correct.

### Explicit non-goals for the first release

- Live automated trading
- Portfolio execution or brokerage integration
- Guaranteed price predictions
- High-frequency forecasting
- Training one global model across every asset before single-asset baselines are validated
- Supporting every possible data provider

## 5. Reuse, replace, and upgrade

### Reuse

- Streamlit navigation, visual style, and Plotly charts
- CLI entry-point concept
- Metric helper functions after adding tests
- ARIMA, Prophet, LSTM, GRU, and ensemble code as references
- Saved-artifact concept
- Existing crypto dataset as one documented data source

### Refactor heavily

- Model interfaces and model registry
- Training orchestration
- Configuration handling
- Result serialization
- Dataset loading and validation
- Feature generation
- Streamlit business logic

### Replace

- Global feature engineering across mixed assets
- Training on rows from multiple symbols as though they form one series
- A single fixed 80/20 evaluation as the main evidence
- TensorFlow sequence models, replacing them with PyTorch for course alignment
- Hard-coded ARIMA order `(5, 1, 0)` as the only statistical configuration
- Fixed five-epoch neural-network training
- Unweighted averaging as the only ensemble strategy

## 6. Correctness issues that must be resolved first

1. Feature calculations must be grouped by `asset_id` and ordered by timestamp.
2. A model run must never cross from one instrument into another unless it is an explicitly designed global/panel model.
3. Data must be split before fitting scalers, transformations, feature selectors, or model parameters.
4. Lag and rolling features must use past values only.
5. Hyperparameter selection must use validation folds, never the final test set.
6. Final test data must remain untouched until the selected model is evaluated once.
7. Results must always identify dataset version, asset, frequency, target, horizon, split dates, features, model parameters, seed, and code version.
8. Backtest forecasts must distinguish one-step forecasts using observed history from recursive multi-step forecasts using prior predictions.

## 7. Target architecture

```text
marketcast-lab/
├── README.md
├── ROADMAP.md
├── pyproject.toml
├── configs/
│   ├── assets.yaml
│   ├── experiments.yaml
│   └── models.yaml
├── data/
│   ├── raw/                 # Immutable source downloads; normally gitignored
│   ├── interim/             # Normalized provider output
│   └── processed/           # Validated modeling datasets
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_eda_stationarity.ipynb
│   ├── 03_statistical_models.ipynb
│   ├── 04_ml_models.ipynb
│   └── 05_deep_learning.ipynb
├── src/market_forecast/
│   ├── cli.py
│   ├── config.py
│   ├── data/
│   │   ├── schema.py
│   │   ├── validation.py
│   │   ├── registry.py
│   │   └── providers/
│   │       ├── base.py
│   │       ├── csv_provider.py
│   │       └── market_provider.py
│   ├── analysis/
│   │   ├── decomposition.py
│   │   ├── stationarity.py
│   │   └── diagnostics.py
│   ├── features/
│   │   ├── transforms.py
│   │   ├── lags.py
│   │   └── technical.py
│   ├── models/
│   │   ├── base.py
│   │   ├── baselines.py
│   │   ├── statistical.py
│   │   ├── machine_learning.py
│   │   ├── deep_learning.py
│   │   └── registry.py
│   ├── evaluation/
│   │   ├── splits.py
│   │   ├── backtest.py
│   │   ├── metrics.py
│   │   └── comparison.py
│   ├── experiments/
│   │   ├── runner.py
│   │   └── artifacts.py
│   └── dashboard/
│       ├── app.py
│       ├── pages/
│       └── charts.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── artifacts/               # Models and reproducible experiment outputs
├── reports/
│   ├── figures/
│   └── final_report.md
└── scripts/
    ├── fetch_data.py
    ├── run_experiments.py
    └── launch_dashboard.py
```

Notebooks should call functions from `src/`; they must not become a second, independent implementation of the pipeline.

## 8. Canonical data contract

All providers should normalize data into a common long-format schema:

| Column | Meaning |
| --- | --- |
| `asset_id` | Stable internal identifier, such as `crypto:BTC-USD`. |
| `symbol` | Provider-facing ticker or symbol. |
| `asset_class` | `crypto`, `equity`, `forex`, or `commodity`. |
| `timestamp` | Timezone-aware timestamp before daily normalization. |
| `frequency` | Daily, hourly, or another explicit interval. |
| `open`, `high`, `low`, `close` | Raw OHLC values. |
| `adjusted_close` | Corporate-action-adjusted value when available. |
| `volume` | Nullable because it may not be meaningful for some forex sources. |
| `currency` | Quote currency. |
| `provider` | Data origin. |
| `retrieved_at` | Data lineage timestamp. |

Each processed dataset should have a validation report containing:

- Row count and time range
- Duplicate-key count
- Missing-value and missing-timestamp summary
- Sampling-frequency consistency
- Non-positive or impossible OHLC checks
- Outlier flags
- Provider and retrieval metadata
- Transformations applied

## 9. Model interface

Every model family should use a shared conceptual interface:

```python
class ForecastModel:
    def fit(self, train_frame, validation_frame=None): ...
    def predict(self, horizon, context=None): ...
    def get_params(self): ...
    def save(self, path): ...
```

Model adapters may internally use statsmodels, scikit-learn, or PyTorch. The evaluation layer should interact with the interface rather than contain model-specific branches.

Required model groups:

1. **Baselines:** last value, drift, moving average, and seasonal naive where meaningful.
2. **Statistical:** Simple Exponential Smoothing, Holt/Holt-Winters, ARIMA, and SARIMA.
3. **Machine learning:** Ridge, Lasso, Random Forest, and XGBoost or LightGBM using leakage-safe lag/rolling features.
4. **Deep learning:** vanilla RNN, LSTM, and GRU implemented in PyTorch.
5. **Optional extensions:** Prophet, GARCH for volatility, attention/Transformer, and validated ensembles.

## 10. Evaluation protocol

### Data partitions

Use three chronological regions:

- **Training/validation region:** walk-forward folds for tuning and model selection
- **Final test region:** locked until selection is complete
- **Optional live holdout:** the most recent period for the final demonstration

### Backtesting

Implement expanding-window backtesting first:

```text
Fold 1: [train------][validate]
Fold 2: [train------------][validate]
Fold 3: [train------------------][validate]
                                      [final test]
```

Evaluate at multiple horizons, such as 1, 5, and 20 daily steps. Use equal folds and horizons when comparing models.

### Metrics

Required:

- MAE
- MSE and RMSE
- MAPE where the target is safely away from zero
- sMAPE
- MASE against a defined naive baseline
- AIC and BIC for applicable statistical models
- Training and inference time

Recommended diagnostics:

- Mean error/bias
- Residual ACF and Ljung-Box test
- Directional accuracy as a secondary metric, not the main forecasting score
- Confidence/prediction interval coverage when intervals are produced
- Diebold-Mariano testing for selected final comparisons, if time permits

Aggregate results by model, asset, asset class, horizon, and fold. Report both the mean and variability; a single best-case metric is insufficient.

## 11. Experiment reproducibility

Each experiment should produce an immutable run directory:

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

The run metadata should include:

- Run ID and timestamp
- Git commit when available
- Python and dependency versions
- Random seeds
- Asset and data range
- Feature and transform configuration
- Model parameters
- Split/fold boundaries
- Forecast strategy and horizon
- Warnings and failures

Start with JSON/CSV/Parquet artifacts. Add MLflow only if experiment volume makes the simple format difficult to manage.

## 12. Dashboard requirements

The upgraded Streamlit application should include:

1. **Data Explorer** - asset selection, date range, missing-data summary, raw/adjusted price, returns, and volume.
2. **Time-Series Analysis** - decomposition, rolling statistics, ACF/PACF, stationarity tests, and transformations.
3. **Experiment Setup** - asset, target, frequency, horizon, split strategy, models, and feature configuration.
4. **Backtest Results** - fold-aware forecasts, metrics, residual plots, interval coverage, and runtime.
5. **Model Comparison** - rankings by asset and horizon with stability and limitations, not only the lowest RMSE.
6. **Future Forecast** - clearly separated from historical backtesting, with uncertainty and a non-advisory disclaimer.
7. **Experiment History** - saved configurations and reproducible results.

The dashboard should read results created by the experiment pipeline. It should not contain a duplicate training implementation.

## 13. Implementation phases

### Phase 0 - Preserve and baseline the current project

- Create a migration branch.
- Record the current application behavior with screenshots and smoke tests.
- Add a dependency lock using `pyproject.toml` and a supported Python version.
- Document the current crypto dataset source and license.
- Save a reproducible baseline run before changing model behavior.

**Exit criteria:** the legacy dashboard and CLI can be started reproducibly, and known defects are documented.

### Phase 1 - Repair single-asset correctness

- Introduce `asset_id` and the canonical schema.
- Validate and sort each series independently.
- Group all feature engineering by asset.
- Select an asset before truncation, transformation, and splitting.
- Add naive forecasts.
- Add unit tests that fail if values leak across assets or time boundaries.
- Remove duplicated forecasting logic between training and Streamlit.

**Exit criteria:** BTC can be trained and evaluated end-to-end with no cross-asset contamination and passes leakage tests.

### Phase 2 - Build the evaluation framework

- Implement expanding-window and rolling-window splitters.
- Lock a final test period.
- Add fold-aware predictions and metric aggregation.
- Add MAE, RMSE, sMAPE, MASE, runtime, and statistical diagnostics.
- Define one-step and recursive multi-step protocols.
- Persist complete run metadata and predictions.

**Exit criteria:** repeated backtests are deterministic, comparable, and reproducible from one command.

### Phase 3 - Complete statistical analysis and models

- Add rolling statistics and seasonal decomposition.
- Add ADF and KPSS stationarity tests.
- Add ACF/PACF and residual diagnostics.
- Implement exponential smoothing and Holt-Winters.
- Implement ARIMA/SARIMA candidate selection using validation plus AIC/BIC.
- Document when transformations and seasonal periods are appropriate.

**Exit criteria:** the statistical-model report includes assumptions, diagnostics, selection reasoning, and final backtest results.

### Phase 4 - Add lag-based machine learning

- Build past-only lag, rolling, calendar, trend, and volatility features.
- Use scikit-learn `Pipeline` objects so transformations fit on training folds only.
- Implement Ridge, Lasso, Random Forest, and XGBoost/LightGBM.
- Tune a small justified hyperparameter space through time-series validation.
- Add feature-importance or coefficient analysis with suitable caveats.

**Exit criteria:** ML models are compared fairly against naive and ARIMA-family baselines on identical folds.

### Phase 5 - Migrate deep learning to PyTorch

- Implement reusable dataset/window builders.
- Implement vanilla RNN, LSTM, and GRU models.
- Add deterministic seeds, early stopping, checkpointing, learning-rate controls, and device selection.
- Support direct or recursive multi-step forecasting and document the choice.
- Tune lookback, hidden size, layers, dropout, batch size, optimizer, and learning rate on validation data.
- Record learning curves and parameter counts.

**Exit criteria:** all three sequence models train reproducibly and are compared with statistical and ML baselines.

### Phase 6 - Generalize to multiple markets

- Add provider adapters for equities, forex, and commodities.
- Normalize timezones and daily boundaries.
- Preserve trading calendars rather than filling market holidays as ordinary observations.
- Use adjusted prices for equity analysis when appropriate.
- Document oil-series construction and futures-roll handling.
- Run the same evaluation contract for each representative asset.

**Exit criteria:** at least one crypto, equity, forex, and oil series completes the same experiment suite with comparable result artifacts.

### Phase 7 - Upgrade the dashboard

- Rename crypto-specific UI elements to generic asset terminology.
- Split the interface into analysis, experiments, comparison, and forecasting pages.
- Read experiment artifacts rather than retrain implicitly inside UI rendering.
- Add fold-level visualizations, residual diagnostics, uncertainty, and run metadata.
- Clearly mark historical evaluation versus speculative future forecasts.

**Exit criteria:** the dashboard can reproduce the report's primary figures and explain the selected model for each asset/horizon.

### Phase 8 - Testing, reporting, and presentation

- Reach strong test coverage for data boundaries, transformations, splitters, metrics, artifact loading, and model smoke tests.
- Run all final experiments from clean configuration files.
- Produce tables and figures directly from saved artifacts.
- Write the scientific report with methods, experiments, limitations, and conclusions.
- Prepare a concise live demo and presentation.
- Verify every claim against saved results.

**Exit criteria:** another student can clone the repository, obtain or load the documented data, reproduce the selected experiments, and understand the conclusions.

## 14. Testing strategy

### Unit tests

- Canonical-schema validation
- No cross-asset lag/rolling contamination
- No train/test overlap
- Scaler and transform fitting only on training data
- Correct fold boundaries
- Metric values on known examples
- Multi-step forecast length and alignment
- Artifact serialization round trips

### Integration tests

- CSV provider to processed dataset
- One complete statistical-model experiment
- One complete scikit-learn experiment
- Small CPU-only PyTorch training run
- Saved run to dashboard loading

### Regression tests

- Fixed small dataset with expected split dates and metric tolerances
- CLI smoke tests
- Dashboard import/startup smoke test

## 15. Suggested command-line experience

```bash
# Validate and prepare configured datasets
python -m market_forecast.cli data prepare --config configs/assets.yaml

# Analyze one asset
python -m market_forecast.cli analyze --asset crypto:BTC-USD

# Run a reproducible experiment suite
python -m market_forecast.cli experiment run --config configs/experiments.yaml

# Compare completed runs
python -m market_forecast.cli report build --latest

# Launch the dashboard
streamlit run src/market_forecast/dashboard/app.py
```

## 16. Milestones and suggested schedule

| Week | Deliverable |
| --- | --- |
| 1 | Preserve baseline, define schema, repair per-asset processing, add naive baseline. |
| 2 | Build walk-forward evaluation, metrics, artifacts, and tests. |
| 3 | Complete EDA, stationarity, decomposition, Holt-Winters, ARIMA/SARIMA. |
| 4 | Add lag-feature Ridge/Lasso, Random Forest, and boosting models. |
| 5 | Implement and validate PyTorch RNN/LSTM/GRU. |
| 6 | Add stock, forex, and oil data adapters; run cross-market experiments. |
| 7 | Upgrade Streamlit and generate final comparison figures. |
| 8 | Complete report, presentation, reproducibility audit, and rehearsal. |

If less time is available, reduce the number of assets and tuning combinations before removing model families or correct validation.

## 17. Definition of done

The project is complete when:

- The repository and application use the new multi-market identity.
- At least four representative market series use a documented common schema.
- No preprocessing, feature, scaling, or evaluation leakage remains.
- Naive, statistical, ML, RNN, LSTM, and GRU models are evaluated.
- Deep-learning models are implemented in PyTorch.
- Walk-forward results cover multiple horizons and include variability across folds.
- Statistical assumptions and residuals are analyzed.
- Every reported result can be traced to a saved configuration and data manifest.
- Tests protect asset and time boundaries.
- The dashboard distinguishes backtests from future forecasts and states limitations.
- The report compares model families, explains results, identifies limitations, and recommends methods by market/horizon.
- Setup and reproduction commands work from a clean environment.

## 18. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Scope becomes too large | Begin with one asset per class and daily data; treat extra assets as extensions. |
| Neural models consume excessive time | Use bounded search spaces, early stopping, CPU smoke configurations, and saved checkpoints. |
| Unfair comparison between markets | Use a shared evaluation contract while preserving each market's calendar and target semantics. |
| Oil data contains contract-roll artifacts | Use a documented continuous series and explain its adjustment method. |
| MAPE becomes misleading | Include sMAPE and MASE and explain metric limitations. |
| Dashboard duplicates pipeline logic | Make the UI consume saved experiment artifacts and shared library functions. |
| Strong-looking results are caused by leakage | Add automated boundary tests and manually audit fold dates and transformations. |
| Data cannot be redistributed | Store download instructions and manifests rather than unlicensed raw data. |

## 19. Immediate next actions

Execute these in order:

1. Rename the repository to `marketcast-lab` and update project-facing text.
2. Create the package structure and dependency configuration while preserving the legacy entry points temporarily.
3. Write tests demonstrating the current cross-asset contamination bug.
4. Fix per-asset selection and feature engineering.
5. Implement the canonical schema, data validation, and naive baselines.
6. Implement the walk-forward evaluator and artifact contract.
7. Migrate sequence models from TensorFlow to PyTorch.
8. Add the remaining statistical and ML model families.
9. Add one equity, one forex, and one oil dataset through provider adapters.
10. Run the final experiment matrix before completing the dashboard and report.

The first technical milestone is deliberately narrow: **produce one trustworthy, reproducible BTC experiment before adding another market**. Once that path is correct, new assets become adapters and configurations rather than new one-off applications.
