# MarketCast Lab

MarketCast Lab compares daily forecasts for BTC/USD, SPY, EUR/USD, and WTI Cushing spot oil. It evaluates 14 model families at 1, 5, and 20 observed-session horizons with three expanding validation folds and a locked holdout. The dashboard, report, and slides read saved runs; they do not train while rendering. This course study is educational, not financial advice.

![MarketCast Lab preview](images/screenshots.png)

## Setup and commands

Use Python 3.11 to 3.13. In PowerShell, from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,boosting,dashboard,presentation]"
python -m pytest -q
```

For a network-free BTC smoke run, use `python main.py experiment --config configs/smoke.json`. For the four-market study, run:

```powershell
python main.py four-markets --config configs/four_asset_full.json
python main.py audit-four-markets
python main.py build-deliverables
python main.py build-future-scenarios
python -m streamlit run src/streamlit.py
```

The first study run downloads SPY from Yahoo Finance, EUR/USD from the ECB, and WTI spot from the EIA; provider responses are cached in ignored `data/raw/`. The bundled CSV supplies BTC. The audit recomputes MAE and RMSE from saved predictions. The original four-market run produced 672 metric rows (14 models x 3 horizons x 4 evaluation windows x 4 assets), with no model failures and one FX convergence warning. Source revisions can change a new run.

For a fully offline software check, use synthetic data and separate outputs:

```powershell
python main.py offline-fixture
python main.py audit-four-markets --output assets/offline-fixture
python main.py build-deliverables --index assets/offline-fixture --output assets/offline-phase4
$env:MARKETCAST_ARTIFACT_ROOT='assets/offline-fixture'
$env:MARKETCAST_PHASE4_ROOT='assets/offline-phase4'
python -m streamlit run src/streamlit.py
```

The dashboard labels synthetic data; it is not market evidence. To include ETH, QQQ, EUR/JPY, Henry Hub gas, and the US ten-year Treasury yield, pass `--assets configs/assets_extended.json` to `four-markets` or `offline-fixture` and give that study its own `--output` directory. `python main.py --help` lists all commands.

## Method and evidence

The saved study ends on 2025-10-14 and uses each asset's last 180 observed dates. Holidays and weekends are not inserted. SPY targets adjusted close; BTC targets close; ECB and EIA provide reference or spot point values. WTI is a Cushing cash spot benchmark, not a continuous futures contract. Different target units make raw RMSE comparable within an asset, not across assets.

The model matrix includes last value, drift, seasonal naive, exponential smoothing, Holt-Winters, ARIMA, SARIMA, Ridge, Lasso, Random Forest, XGBoost, and PyTorch RNN/LSTM/GRU. A non-baseline recommendation requires at least 5% improvement over last value in both validation mean RMSE and mean plus one fold standard deviation. The holdout checks a selection after validation; it does not choose models. Long-horizon BTC and oil validation gains reversed on the original holdout. Three folds and one short holdout limit ranking confidence.

Each run saves its config, source URL or provenance gap, raw SHA-256, target semantics, quality report, fold dates, parameters, predictions, metrics, diagnostics, warnings, and models. The canonical data has one row per asset and observed UTC date; duplicate keys are rejected, missing point-series OHLC stays null, and transforms fit only on training history. Prediction rows record asset, model, fold, partition, horizon, step, timestamp, actual, and forecast. `assets/phase3/fold_metrics.csv` links summary rows to run IDs and source files.

| Series | Source and target | Publication limit |
| --- | --- | --- |
| BTC/USD | Bundled `data/crypto_statistics_data.csv`, close | Upstream source and license unknown |
| SPY | [Yahoo Finance](https://finance.yahoo.com/quote/SPY/history/), adjusted close | Raw-feed redistribution rights unverified |
| EUR/USD | [ECB reference series](https://data.ecb.europa.eu/data/datasets/EXR/EXR.D.USD.EUR.SP00.A), USD per euro | Cite ECB and disclose transformations |
| WTI Cushing | [EIA daily spot series](https://www.eia.gov/dnav/pet/hist/RWTCD.htm), USD per barrel | Cite EIA; this is a spot assessment |

The expanded catalog also uses bundled ETH, Yahoo QQQ, ECB EUR/JPY, EIA Henry Hub gas, and FRED DGS10 yield. Their exact provider URLs and units are recorded in each run manifest. The study does not test trading costs or calibrated predictive uncertainty. Forecast scenarios start from the historical cutoff and omit invented future market dates.

## Files and demonstration

- `src/market_forecast/` contains providers, models, evaluation, runs, artifact readers, and publication code. `src/tests/` contains the tests; `src/streamlit.py` is the artifact-driven dashboard.
- `configs/` contains experiment settings and asset catalogs. `data/` contains the bundled BTC series and ignored provider caches.
- `assets/phase3/` contains the committed comparison snapshot and figures; `assets/phase4/` contains the saved presentation, figures, and scenario output. Full run directories under `assets/runs/` are ignored and must be regenerated to audit the snapshot interactively.
- `assets/legacy/adam/` holds outputs from the retired TensorFlow/Prophet trainer. They are excluded from the audited study. The root notebook is historical, not a supported training entry point.
- `images/` holds the screenshot and fallback comparison figures: [relative RMSE](images/relative_rmse.png) and [fold variability](images/fold_variability.png).

For a demo, audit the saved predictions, build deliverables, open `assets/phase4/presentation.pptx`, and inspect the dashboard's Experiment History, Backtest Results, Model Comparison, and Future Forecast pages. On a clean clone the committed comparison and slides are available, but full runs may be absent; the dashboard explains how to generate them. The fallback figures are saved examples, not newly reproduced results.

The supported CLI is `main.py` or the installed `marketcast` command. The former interactive `train` and `ui` commands were retired. Prediction files fall back to CSV when Parquet support is unavailable. GitHub Actions are disabled; reproduction uses the local commands above.
