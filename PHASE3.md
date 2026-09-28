# Phase 3: four-market experiment

## Reproduce

Install Python 3.11–3.13 and `python -m pip install -e ".[dev,boosting]"`.
From the repository root run:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest -q
python main.py four-markets --config configs/four_asset_full.json
python main.py audit-four-markets
```

The first four-market run downloads three sources into ignored `data/raw/`. Later runs use those snapshots unless the cache is removed. Keep raw source files locally, especially SPY. The bundled BTC CSV is already in `data/`. On the development machine the four runs took 4.6, 2.4, 2.4 and 2.2 seconds respectively (about 17 seconds including report generation); hardware and library versions change runtime.

The source cutoff is **2025-10-14**, the end of the bundled BTC series. Each asset uses its last 180 observed dates ending at or before that cutoff, three expanding validation folds of 20 observations, and a locked 20-observation holdout. Forecast horizons are 1, 5 and 20 *future observations*, not a fixed number of elapsed days. Each run manifest lists exact dates and indices. The different market calendars create the smallest necessary exception to identical calendar dates. No weekends or holidays are inserted. The end date is common; the first validation date differs by asset.

The outcome is raw close for BTC and daily ECB/EIA point series; for SPY it is Yahoo's adjusted close, while the raw OHLC fields remain unadjusted. ECB and EIA publish one point per date; unavailable OHLC and volume fields are null. The ECB rate is USD per euro at the ECB reference fixing, rather than a 24-hour FX closing price. EIA WTI is a Cushing cash spot benchmark in USD per barrel, **not** a continuous futures contract. No roll construction or roll jumps apply. The oil adapter limits the study to 2023 onward; the full EIA history includes the negative 2020 WTI price, which should not be silently discarded in a longer study.

## Sources and use

| Asset | Provider and source | Reuse status |
| --- | --- | --- |
| BTC/USD | Bundled `data/crypto_statistics_data.csv`; upstream source is unknown | License and redistribution rights unknown; retain the existing file without claiming a new license |
| SPY | [Yahoo Finance SPY chart](https://finance.yahoo.com/quote/SPY/history/), obtained through the daily chart endpoint | Adjusted prices are used locally for research. Redistribution permission for the raw feed is unverified; raw response is ignored by Git |
| EUR/USD | [ECB exchange-rate series D.USD.EUR.SP00.A](https://data.ecb.europa.eu/data/datasets/EXR/EXR.D.USD.EUR.SP00.A) | [ECB statistics reuse policy](https://www.ecb.europa.eu/stats/ecb_statistics/governance_and_quality_framework/html/usage_policy.en.html) permits reuse with source citation and disclosure of modifications |
| WTI Cushing spot | [EIA daily WTI spot series](https://www.eia.gov/dnav/pet/hist/RWTCD.htm) and its official XLS download | [EIA reuse policy](https://www.eia.gov/about/copyrights_reuse.php) permits distribution of its government data with attribution |

Each run's `data_manifest.json` records the exact URL, retrieval time, SHA-256 of the raw response, local path, license status, symbol mapping, timezone, currency, adjustment and target semantics. ECB/EIA publication schedules and dates are distinct from exchange trading calendars. Yahoo adjusted close is a vendor supplied adjusted price; the exact revision history is not frozen beyond the cached response hash.
For the bundled BTC file, the recorded timestamp is only the file modification time; the original acquisition time and URL are unknown.

## Evidence and interpretation

`artifacts/phase3/run_index.json` identifies the four immutable runs. Each run contains `config.json`, `data_manifest.json`, `environment.json`, `quality_report.json`, `eda.json`, `figures/eda.png`, `metrics.csv`, `fold_predictions.csv` or Parquet, `diagnostics.json`, `failures.json`, saved models and a comparison figure. `artifacts/phase3/fold_metrics.csv` attaches run ID, config, manifest and prediction path to every number. `comparison.csv`, `recommendations.csv`, `residual_diagnostics.csv`, `relative_rmse.png` and `analysis.md` are generated from those runs.

The minimum matrix has 14 models × 3 horizons × (3 validation folds + 1 holdout) × 4 assets = **672 metric rows**. This includes last value, drift, weekly observation-count seasonal naive, exponential smoothing, Holt-Winters, ARIMA, SARIMA, Ridge, Lasso, Random Forest, XGBoost, and PyTorch RNN/LSTM/GRU. The 5 or 7 observation seasonal period is an exploratory weekly hypothesis, not proof of seasonality. EDA supplies price and return ACF/PACF, ADF/KPSS, decomposition, rolling mean/volatility and a transformation decision. The simple ARIMA/SARIMA orders are fixed for compute control; no holdout-driven tuning is performed. Two training epochs and small networks make this a minimum matrix rather than a full tuning study.

Recommendations use validation mean RMSE plus one fold standard deviation. A complex model is eligible only if both that score and mean RMSE improve at least 5% over last value. The holdout does not choose models. In this snapshot, last value remains the robust selection for all EUR/USD and SPY horizons and BTC horizon 5; BTC horizon 1 selects Holt-Winters and horizon 20 XGBoost; oil selects Ridge at horizon 1 and XGBoost at horizons 5 and 20. See `recommendations.csv` for fold variability and holdout values. BTC horizon 20 and oil horizons 5/20 show holdout deterioration against last value despite validation gains. The 1-step oil Ridge gain is more convincing, but three validation folds remain a small sample. One statsmodels convergence warning appeared for FX; no run failures were recorded. The warning may indicate an unstable statistical fit even when a forecast is returned.

## Limits for Phase 4

These are short, recent windows with one instrument per class. Regimes, adjusted-price revisions, spot assessments, different observation calendars and different price scales limit generalization. Relative RMSE and MASE support cross-class reading; raw RMSE is only comparable within an asset. Recursive 20-observation predictions accumulate error. ARIMA/SARIMA intervals are the only available interval estimates; coverage from one 20-observation holdout is weak uncertainty evidence. Historical non-stationarity can invalidate fitted relationships. The BTC upstream provenance and SPY redistribution rights require resolution before publishing source datasets. Phase 4 should consume the saved run index and run directories, present the recorded warnings, show fold variability and holdout reversals, and avoid retraining in the dashboard.
