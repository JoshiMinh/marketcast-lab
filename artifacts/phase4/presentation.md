# MarketCast Lab
Four daily markets. Fourteen models. Three forecast horizons.

---

# Research question
How does forecast performance change across asset class and horizon?
Recursive predictions for the next 1, 5 and 20 observed sessions.

---

# Data and calendars
- crypto:BTC-USD: Bundled cryptocurrency statistics CSV; upstream provenance unknown
- equity:SPY: Yahoo Finance chart, SPDR S&P 500 ETF Trust
- forex:EUR-USD: ECB euro foreign exchange reference rate
- oil:WTI-CUSHING-SPOT: EIA Cushing, OK WTI spot price FOB
BTC trades seven days. SPY, ECB reference FX and WTI spot retain their own observed dates.

---

# Evaluation design
Three expanding validation folds and one locked 20-observation holdout per asset.
The same model matrix and target timestamps apply within each asset.

---

# Model families
Naive and seasonal baselines; exponential smoothing and ARIMA; lag regression and trees; PyTorch RNN, LSTM and GRU.
Small networks and fixed statistical orders bound compute.

---

# Validation recommendations
| asset_id | 1 | 5 | 20 |
| --- | --- | --- | --- |
| crypto:BTC-USD | holt_winters | last_value | xgboost |
| equity:SPY | last_value | last_value | last_value |
| forex:EUR-USD | last_value | last_value | last_value |
| oil:WTI-CUSHING-SPOT | ridge | xgboost | xgboost |
Selection requires a 5% gain in mean RMSE and mean plus fold standard deviation versus last value.

---

# Stability matters
See `../phase3/fold_variability.png` and `../phase3/relative_rmse.png`.
3 selected complex-model cases underperformed last value on the locked holdout.

---

# Limits
One recent window, one instrument per class and limited tuning.
SPY adjusted prices, ECB fixing and EIA spot assessments have different semantics.
Statistical intervals do not establish calibrated uncertainty.

---

# Reproduce
`python main.py four-markets --config configs/four_asset_full.json`
`python main.py audit-four-markets`
All displayed results trace to `../phase3/run_index.json`.

---

# Conclusion
Choose by asset and horizon; inspect fold variation and holdout reversals.
Historical analysis only. No trading advice.
