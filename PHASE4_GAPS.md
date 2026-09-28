# Phase 4 audit and gap checklist

Audit basis: `ROADMAP.md`, `artifacts/phase3/run_index.json`, four run manifests, and `python main.py audit-four-markets` (4 runs and 672 metric rows reproduced from predictions).

| Requirement | Initial status | Resolution target |
| --- | --- | --- |
| Four assets, 14 families, horizons 1/5/20, three folds and holdout | Pass | Preserve run IDs and artifact audit |
| Traceable numbers, data manifests and prediction files | Pass | Shared artifact reader for dashboard and report |
| Calendar and target semantics | Pass | Surface source notes and observed-session horizons |
| Data Explorer, Time-Series Analysis, Experiment Setup, Backtest Results, Model Comparison, Future Forecast, Experiment History | Resolved | `src/streamlit.py` has seven artifact-driven views |
| Legacy TensorFlow predictions excluded from final evidence | Resolved | New dashboard has no legacy imports; README labels the old CLI |
| Scientific report generated from saved artifacts | Resolved | `python main.py build-deliverables` writes `artifacts/phase4/scientific_report.md` |
| Presentation and offline demo | Resolved | Generated PPTX/Markdown deck, `docs/DEMO.md` and `docs/demo/` figures |
| Clean-clone setup and source/license reference | Resolved | README, data dictionary, source register and config reference |
| Automated dashboard startup and reproducibility checks | Resolved | Fresh virtual environment: 30 tests passed, 672 metrics audited, Streamlit health returned HTTP 200 |
| Assessment rubric mapping | Resolved | `docs/RUBRIC.md` maps 30/30/40 evidence |

Remaining scientific limits carried forward from Phase 3: BTC upstream provenance is unknown; SPY raw redistribution rights are unverified; results use one short historical window; weekly seasonality is exploratory; an FX statistical fit raised a convergence warning; 20-observation oil and BTC validation gains can reverse on holdout; statistical intervals alone cannot establish reliable uncertainty coverage.

Final clean-environment reproduction on Windows/Python 3.13 used `.venv`, installed `.[dev,boosting,dashboard,presentation]`, passed 30 tests, completed the four-market run in 17.76 seconds, audited all 672 metric rows, regenerated the report/slides/scenarios, and received HTTP 200 from the Streamlit health endpoint. An offline synthetic four-asset fixture also ran the full matrix and reproduced 672 rows without network access.
