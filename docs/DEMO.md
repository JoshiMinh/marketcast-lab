# Deterministic demonstration

1. Run `python main.py audit-four-markets` and show the four run IDs and 672 reproduced metric rows.
2. Run `python main.py build-deliverables`, then open `artifacts/phase4/presentation.pptx` or `presentation.md`.
3. Launch `python -m streamlit run src/streamlit.py`. In **Experiment History**, show the SPY target and FX convergence warning. In **Data Explorer**, show the missing OHLC fields for ECB and EIA and their observed calendars.
4. In **Backtest Results**, select BTC, XGBoost, horizon 20, final holdout. Read the saved actual and predicted values. In **Model Comparison**, show its validation mean and fold spread beside final-test RMSE and last value.
5. In **Future Forecast**, show that the scenario origin is 2025-10-14 and that no invented future holiday dates appear. Explain that `build-future-scenarios` was run explicitly; the dashboard did not train.
6. Open `artifacts/phase4/scientific_report.md` and use its run-ID table to trace a displayed number through `artifacts/phase3/fold_metrics.csv` to the run config, data manifest and fold prediction file.

If training or the network is unavailable, use the saved deck plus the committed fallback figures `docs/demo/relative_rmse.png` and `docs/demo/fold_variability.png`. The bundled BTC smoke command (`python main.py experiment --config configs/smoke.json`) works without external data. If four-market run directories are unavailable on a clean clone, the dashboard still starts and explains how to acquire them; do not present the fallback figures as newly reproduced results.

For an interactive offline demo, run `python main.py offline-fixture`, audit it with `python main.py audit-four-markets --output artifacts/offline-fixture`, and build its separate report with `python main.py build-deliverables --index artifacts/offline-fixture --output artifacts/offline-phase4`. Set `MARKETCAST_ARTIFACT_ROOT=artifacts/offline-fixture` and `MARKETCAST_PHASE4_ROOT=artifacts/offline-phase4` before launching Streamlit. The prominent synthetic label means these outputs verify the software workflow only.
