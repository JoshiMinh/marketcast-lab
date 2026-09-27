# Phase 1 migration note

## Shared source of truth added

The `market_forecast` package now owns:

- canonical crypto normalization, validation, selection, and quality reports;
- per-asset, past-only lag and rolling features;
- chronological train/validation/final-test partitions;
- an auditable train-only scaler;
- last-value, drift, and seasonal-naive baselines;
- deterministic Phase 1 artifact generation; and
- dashboard data preparation helpers.

The legacy training path selects `crypto:BTC-USD` before feature engineering and rejects mixed-asset input. The legacy dashboard now filters the chosen crypto before feature engineering and truncation.

## Phase 2 additions

- One registry and `ForecastModel` interface now cover baselines, statistical models, lag regressors, XGBoost, and PyTorch sequence models.
- Expanding-window folds share target timestamps across models and leave the final test region untouched during tuning.
- Recursive and observed-history one-step strategies are separate execution paths.
- Runs use unique directories and retain fold predictions, metrics, diagnostics, model files, failures, warnings, comparison outputs, and environment/data manifests.
- TensorFlow is no longer used by the trusted experiment pipeline.

## Intentionally retained for later replacement

- `src/models.py` still contains TensorFlow LSTM/GRU, Prophet, and fixed-order ARIMA code for the legacy interface only.
- `src/train.py` and `src/streamlit.py` still contain legacy advanced-model forecasting and fixed-holdout sequence preparation; their outputs are excluded from trusted evidence.
- Existing files under `results/` are legacy artifacts and are not accepted as Phase 1 evidence.
- The Streamlit presentation remains crypto-specific and is not being redesigned in Phase 1.
- The notebook predates the canonical package and must not be treated as a second implementation.

Phase 3 should add licensed equity, forex, and oil provider adapters without changing the shared evaluation contract. Phase 4 should move Streamlit entirely onto saved Phase 2/3 artifacts and then remove the legacy TensorFlow path.
