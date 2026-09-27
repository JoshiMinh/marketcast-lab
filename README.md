# MarketCast Lab

Comparative time-series forecasting across cryptocurrency, equities, foreign exchange, and commodities.

> **Migration status:** MarketCast Lab is being upgraded from a cryptocurrency forecasting prototype into a reproducible multi-market course project. The current application runs on the bundled crypto dataset; the leakage-safe evaluation framework, PyTorch models, and additional market adapters are planned in the [roadmap](ROADMAP.md).

![Current Streamlit application](preview.png)

## Purpose

MarketCast Lab is an academic forecasting laboratory for answering a practical question:

> How do statistical, machine-learning, and deep-learning forecasting methods compare across markets with different trends, seasonal patterns, volatility, and trading calendars?

The finished system will provide one consistent workflow for:

1. Importing and validating market time series.
2. Exploring trend, seasonality, stationarity, autocorrelation, and volatility.
3. Applying leakage-safe transformations and lag/rolling features.
4. Training statistical, machine-learning, and PyTorch sequence models.
5. Evaluating multiple forecast horizons with walk-forward validation.
6. Comparing results by asset, model, horizon, error, stability, and runtime.
7. Presenting reproducible experiments through reports and a Streamlit dashboard.

This is an educational analysis tool, not financial advice or an automated trading system.

## Course coverage

The target implementation is designed for the Time Series Data Analysis course (`AI2029`) and covers:

- Time-series visualization, cleaning, scaling, log/Box-Cox transformation, smoothing, and differencing.
- Trend and seasonal decomposition, ACF/PACF, ADF/KPSS testing, and residual diagnostics.
- Naive baselines, exponential smoothing, Holt-Winters, ARIMA, and SARIMA.
- Lag-feature Ridge/Lasso, Random Forest, and XGBoost or LightGBM models.
- RNN, LSTM, and GRU models implemented in PyTorch.
- `TimeSeriesSplit` or explicit walk-forward validation.
- MAE, MSE, RMSE, MAPE/sMAPE, MASE, AIC/BIC, runtime, and interval diagnostics where applicable.
- Evidence-based comparison, interpretation, limitations, and final model selection.

See [ROADMAP.md](ROADMAP.md) for the complete course-to-deliverable mapping and implementation plan.

## Planned market scope

The minimum final comparison uses daily data for one representative instrument per asset class:

| Asset class | Initial candidate | Key consideration |
| --- | --- | --- |
| Cryptocurrency | BTC/USD | Continuous 7-day market |
| Equity | SPY or AAPL | Exchange calendar and adjusted prices |
| Foreign exchange | EUR/USD | Approximately 24/5 trading |
| Commodity | Crude oil | Contract-roll or benchmark methodology |

Daily frequency keeps the cross-market comparison manageable and defensible. More assets and intraday frequencies are extensions, not initial requirements.

## Current prototype

The repository currently provides:

- A historical cryptocurrency CSV dataset.
- EMA, moving average, RSI, MACD, Bollinger Band, return, and volatility features.
- ARIMA and optional Prophet forecasts.
- TensorFlow/Keras LSTM and GRU prototypes.
- A simple mean ensemble.
- A CLI training workflow.
- A Streamlit dashboard with historical charts, backtest metrics, and forward projections.
- RMSE, MAE, and MAPE reporting.

### Important current limitations

The prototype is not yet the final course implementation:

- The CLI can mix rows from different crypto assets into one apparent series.
- Technical indicators are computed before grouping by asset.
- The dashboard truncates the combined dataset before asset selection.
- Evaluation uses one fixed holdout instead of walk-forward validation.
- It lacks required statistical analysis, classical baselines, lag-based ML models, and residual diagnostics.
- Sequence models use TensorFlow rather than the course-aligned PyTorch implementation.
- Model configuration, run metadata, tests, and data provenance are incomplete.

Until Phase 1 and Phase 2 of the roadmap are complete, generated forecasts should be treated as interface demonstrations rather than reliable experimental results.

## Current project structure

```text
marketcast-lab/
├── data/
│   └── crypto_statistics_data.csv
├── results/
├── src/
│   ├── data.py
│   ├── models.py
│   ├── train.py
│   └── streamlit.py
├── main.py
├── marketcast_lab.ipynb
├── requirements.txt
├── README.md
└── ROADMAP.md
```

The roadmap defines the target package structure. Existing entry points will remain usable during migration where practical.

## Run the current prototype

### 1. Create and activate an environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Train models

Interactive CLI:

```powershell
python main.py
```

Non-interactive example:

```powershell
python main.py train --models all --optimizer adam
```

### 4. Launch the dashboard

```powershell
streamlit run src/streamlit.py
```

Model artifacts are written under `results/<optimizer>/`.

## Target workflow

The upgraded system will separate experimentation from presentation:

```text
provider data
  -> canonical per-asset schema
  -> validation and preprocessing
  -> exploratory/statistical analysis
  -> leakage-safe features and temporal folds
  -> model training and forecasting
  -> immutable run artifacts
  -> comparison report and Streamlit dashboard
```

The dashboard will consume saved experiment artifacts instead of maintaining a second training implementation.

## Reproducibility principles

- Preserve temporal order and market-specific calendars.
- Fit every transformation only on the relevant training fold.
- Never calculate lags or rolling features across asset boundaries.
- Keep the final test period isolated from tuning and model selection.
- Record data source, date range, features, parameters, seeds, folds, software versions, and code revision.
- Compare all models on the same assets, folds, horizons, and metrics.
- Include naive baselines; a complex model is useful only when it beats a simple alternative consistently.

## Roadmap

Development is organized into eight phases:

1. Preserve and document the current baseline.
2. Repair single-asset correctness and add boundary tests.
3. Build walk-forward evaluation and reproducible artifacts.
4. Complete statistical analysis and forecasting models.
5. Add lag-based machine-learning models.
6. Migrate RNN/LSTM/GRU models to PyTorch.
7. Add equity, forex, and oil data adapters and upgrade the dashboard.
8. Produce final experiments, report, presentation, and reproducibility audit.

Detailed tasks, exit criteria, risks, and the definition of done are maintained in [ROADMAP.md](ROADMAP.md).

## Expected final deliverables

- Reusable Python package and command-line interface.
- Streamlit analysis and comparison dashboard.
- Versioned experiment configurations and result artifacts.
- Data audit and exploratory-analysis notebook.
- Statistical, ML, and deep-learning experiment notebooks or generated reports.
- Automated unit, integration, leakage, and regression tests.
- Cross-market results tables and figures.
- Scientific final report and presentation materials.
- Reproduction instructions for a clean environment.

## License and data

A project license and per-dataset source/license documentation must be added before redistribution. Raw market data should not be committed unless its license permits redistribution; otherwise, provide retrieval scripts and data manifests.

## Project status

**Current milestone:** Phase 1 - repair the single-asset pipeline before adding new markets.

The first trustworthy deliverable will be a reproducible BTC experiment with correct per-asset features, temporal validation, naive baselines, and saved fold predictions.
