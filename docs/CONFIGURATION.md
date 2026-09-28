# Configuration reference

`configs/four_asset_full.json` is the minimum four-market matrix. `configs/smoke.json` is a smaller offline BTC smoke test. `configs/btc_full.json` is the older fuller BTC tuning study.

`python main.py offline-fixture` creates deterministic synthetic canonical CSVs under ignored `data/fixtures/generated/` and four separate runs under `artifacts/offline-fixture/`. The fixture uses the same folds, models and horizon configuration; its report is watermarked and must not be used as market evidence.

| Key | Meaning |
| --- | --- |
| `provider`, `asset_id`, `data_path` | One provider and one asset per immutable run; the four-market command supplies these per class |
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
