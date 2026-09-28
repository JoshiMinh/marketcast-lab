# MarketCast Lab

MarketCast Lab compares daily price forecasts for BTC/USD, SPY, EUR/USD, and WTI Cushing spot oil. A shared experiment engine evaluates 14 model families at 1, 5, and 20 **observed-session** horizons with three expanding validation folds and a separate final holdout. The Streamlit dashboard, comparison tables, report, and slides consume saved experiment artifacts. They do not train while rendering.

This is a course research project. Its forecasts are educational and are not financial advice.

## Start from a clean clone

Use Python 3.11–3.13. In PowerShell, from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,boosting,dashboard,presentation]"
python -m pytest -q
```

The bundled BTC CSV supports a quick experiment without network access:

```powershell
python main.py experiment --config configs/smoke.json
```

An expanded catalog includes ETH, QQQ, EUR/JPY, Henry Hub natural gas and the US ten-year Treasury yield. Run it into separate artifacts with `python main.py four-markets --assets configs/assets_extended.json --output artifacts/extended`; for a network-free software check use `python main.py offline-fixture --assets configs/assets_extended.json --output artifacts/offline-extended`. Audit with `python main.py audit-four-markets --output artifacts/extended` (or the offline output). The extended report and slides are generated from that run index with `build-deliverables --index artifacts/extended --output artifacts/extended-report`. The original four-asset study remains the default and its committed results are unchanged. New provider downloads require network access and are cached under ignored `data/raw/`.

For a **four-asset, fully offline software check**, generate deterministic synthetic data and its own artifacts:

```powershell
python main.py offline-fixture
python main.py audit-four-markets --output artifacts/offline-fixture
python main.py build-deliverables --index artifacts/offline-fixture --output artifacts/offline-phase4
$env:MARKETCAST_ARTIFACT_ROOT='artifacts/offline-fixture'
$env:MARKETCAST_PHASE4_ROOT='artifacts/offline-phase4'
python -m streamlit run src/streamlit.py
```

The dashboard labels these outputs as synthetic. They check the software path; they provide no evidence about real market performance. In a new PowerShell session, the dashboard uses the real artifact paths again.

## Reproduce the four-market study

The first command retrieves SPY from Yahoo Finance, EUR/USD from the ECB, and WTI spot from the EIA; it caches source snapshots under ignored `data/raw/`. The bundled CSV supplies BTC. Network access and provider availability are needed for first acquisition.

```powershell
python main.py four-markets --config configs/four_asset_full.json
python main.py audit-four-markets
python main.py build-deliverables
python main.py build-future-scenarios
python -m streamlit run src/streamlit.py
```

`audit-four-markets` recomputes saved MAE and RMSE from fold prediction files. The four-market configuration runs 14 families for each asset at horizons 1, 5, and 20: last value, drift, seasonal naive, exponential smoothing, Holt-Winters, ARIMA, SARIMA, Ridge, Lasso, Random Forest, XGBoost, RNN, LSTM, and GRU. A model failure or warning is recorded in the run rather than silently removed. See [configuration and runtime notes](docs/CONFIGURATION.md).

Generated run directories, provider downloads, and checkpoints are intentionally ignored by Git. The committed [Phase 3 comparison snapshot](artifacts/phase3/analysis.md), [scientific report](artifacts/phase4/scientific_report.md), and [presentation](artifacts/phase4/presentation.pptx) show the completed local study; **reproduce the commands above to obtain the immutable run files needed to audit or explore those numbers interactively**. Each run includes its config, source hash and manifest, quality report, diagnostics, metrics, and fold predictions. A newly generated run has a new ID and can differ when a provider revises historical data. The dashboard shows an acquisition message if the referenced run files are absent.

## Navigate the results

The dashboard has Data Explorer, Time-Series Analysis, Experiment Setup, Backtest Results, Model Comparison, Future Forecast, and Experiment History pages. Backtest plots show saved actual and predicted values by fold. Comparison views expose validation mean and spread, final holdout, runtime, residual diagnostics, and available interval coverage. Future Forecast reads a separately generated scenario file; it is visibly distinct from backtests and does not imply known future market holidays. The saved scenario's origin is the study cutoff, 2025-10-14, so it is a historical-origin demonstration rather than a current-market quote.

The target is adjusted close for SPY and source close/reference value for the other assets. Only dates observed by each provider enter the series; exchange holidays and weekends are never manufactured as ordinary rows. WTI is an EIA **spot** benchmark, so no futures roll is involved. See [methods and limitations](PHASE3.md), the [data dictionary](docs/DATA_DICTIONARY.md), and the [source/license register](docs/SOURCES.md). SPY raw-feed redistribution rights and the bundled BTC CSV's original license are unverified, so raw SPY downloads are not committed and BTC provenance remains a limitation.

## Assessment and demo

- [Scientific report](artifacts/phase4/scientific_report.md) and [Markdown slides](artifacts/phase4/presentation.md), both generated from saved artifacts
- [Deterministic demo and fallback figures](docs/DEMO.md)
- [Data dictionary](docs/DATA_DICTIONARY.md), [source/license register](docs/SOURCES.md), and [configuration reference](docs/CONFIGURATION.md)
- [Phase 4 evidence checklist](PHASE4_GAPS.md) and [assessment rubric map](docs/RUBRIC.md)

The development-machine four-market run took roughly 18 seconds on Windows/Python 3.13/CPU after dependencies were installed; PyTorch installation can take considerably longer. The last local audit reproduced 672 metric rows from fold predictions, with zero model failures and one recorded FX convergence warning. These counts describe the saved study, not a guarantee for a new provider snapshot.

## Troubleshooting

- **No dashboard runs:** run the four-market and audit commands above, or use the offline fixture. The committed report alone does not contain the full prediction files.
- **Provider download fails:** retry the command. Cached snapshots in `data/raw/` are reused; for a network-free demonstration, use the synthetic fixture.
- **Parquet unavailable:** predictions fall back to CSV, and the run records that warning. The audit reads either format.
- **Global pytest plugin interferes:** use the clean virtual environment. If needed, set `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'` before running pytest.
- **Dashboard dependency missing:** install the `dashboard` extra in the setup command. Legacy TensorFlow/Prophet code in `src/models.py` and `src/train.py` is excluded from the four-market results; `python main.py train` does not reproduce this study.

GitHub Actions are disabled for this repository. Reproduction uses the local commands above.
