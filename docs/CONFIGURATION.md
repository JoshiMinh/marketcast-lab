# Configuration reference

`configs/four_asset_full.json` is the minimum four-market matrix. `configs/smoke.json` is a smaller offline BTC smoke test. `configs/btc_full.json` is the older fuller BTC tuning study.

The asset matrix lives in `configs/assets_four.json`. To run the expanded nine-series study, use `--assets configs/assets_extended.json` with `four-markets` or `offline-fixture`. Each catalog entry supplies an asset ID, provider, cache path, seasonal period and optional provider settings. The original four-series catalog remains the default. Keep each study's `--output` directory separate so its run index and summaries remain coherent.

`python main.py offline-fixture` creates deterministic synthetic canonical CSVs under ignored `data/fixtures/generated/` and four separate runs under `artifacts/offline-fixture/`. The fixture uses the same folds, models and horizon configuration; its report is watermarked and must not be used as market evidence.

| Key | Meaning |
| --- | --- |
| `provider`, `asset_id`, `data_path` | One provider and one asset per immutable run; the asset catalog supplies these |
| `provider_options` | Source symbol or series, quote currency, unit or URL for a reusable provider adapter |
| `end_date` | Shared study cutoff, 2025-10-14 in the saved Phase 3 study |
| `models` | Registry names of the 14 required model families |
| `horizons` | Observed-session forecast lengths 1, 5 and 20 |
| `folds` | Number of expanding validation folds, three |
| `final_test_size` | Locked holdout size in observations, 20 |
| `min_train_size` | Minimum first-fold history, 100 |
| `max_train_rows` | Tail observations retained per asset, 180 |
| `seasonal_period` | Exploratory observation-count weekly proxy, 7 for BTC and 5 otherwise |
| `strategy` | `recursive` or `one_step_observed`; never pool strategies |
| `seed` | Random seed for scikit-learn and PyTorch models |
| `model_params` | Bounded parameters and network epochs |
| `tuning_grid` | Optional validation-only candidate list, absent in the Phase 3 minimum matrix |

The four-market command records exact fold dates, targets, selected parameters and source hash in each run manifest. To change a setting, create a new config and a new run; do not edit an existing run directory. The report and dashboard consume the run index, not the config file alone.
