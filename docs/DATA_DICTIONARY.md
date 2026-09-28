# Data dictionary

The canonical long table has one row per `asset_id` and observed `timestamp`. Duplicate keys are rejected. All timestamps are UTC midnight labels for each provider's local observation date; they are not intraday execution times.

| Field | Meaning | Missingness and use |
| --- | --- | --- |
| `asset_id` | Stable class-prefixed ID such as `equity:SPY` | Required; feature and model boundary |
| `symbol` | Human-readable instrument | Required by adapter |
| `asset_class` | `crypto`, `equity`, `forex`, or `oil` | Used for comparisons |
| `timestamp` | Source observation date | Required, unique within asset |
| `frequency` | `daily` | Means one observation on source-published days |
| `open`, `high`, `low` | Source daily bar values | Null for ECB and EIA point series; no synthetic bars |
| `close` | Source daily close or point price | Required; BTC, ECB and EIA target |
| `adjusted_close` | Vendor adjusted close or same as point `close` | SPY target; Yahoo corporate-action adjustment |
| `volume` | Source volume | SPY only; null where source does not provide comparable volume |
| `currency` | Price quote currency | USD for all four series; WTI is USD per barrel |
| `provider` | Normalization adapter name | Source provenance |
| `retrieved_at` | Local acquisition/cache timestamp | BTC uses file modification time as a proxy; original acquisition unknown |

Run prediction rows contain `asset_id`, `model`, `fold`, `partition`, `horizon`, `strategy`, `step`, `timestamp`, `actual` and `prediction`. Validation folds are 1–3; fold 0 denotes the locked final test. Horizon 5 means the first five *observed* target dates after the training boundary. The recursive strategy uses prior model predictions for later steps. Metrics are recomputed from these rows by `audit-four-markets`.
