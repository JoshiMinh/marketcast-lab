# Preservation validation

Validation date: 2026-10-03 (Asia/Saigon). Source revision tested: `2baf1732aaab3f7d5882928946c20fea69351836`. Preparation changes affect documentation, dependency/environment snapshots and ignore rules only.

| Check | Result |
| --- | --- |
| Existing environment `pip check` | Passed; no broken requirements |
| Existing suite | 39 passed in 37.83 seconds; no failures or skips |
| Historical `audit-four-markets` | Four runs; all 672 metric rows reproduced from saved predictions |
| Dashboard coverage | Existing tests exercise Overview, Forecasts, Data & Runs, saved scenarios, missing runs and missing scenario handling |
| Indexed run completeness | All four runs have configuration, manifest, metrics, environment and referenced predictions |
| Fresh environment dependency installation | All 73 exact pins installed; frozen versions match the source environment |
| Fresh project installation | Editable installation with `--no-deps --no-build-isolation` passed |
| Fresh environment `pip check` | Passed; no broken requirements |
| Fresh environment suite | 39 passed in 36.08 seconds; no failures or skips |
| Fresh environment offline smoke | Seven models, three horizons, three validation folds and holdout; completed with empty failure and warning lists |
| Restored PyTorch | 2.14.0+cpu; CUDA unavailable, matching the source environment |
| Research integrity after validation | All 1,692 inventoried files / 176,158,285 bytes match initial size and SHA-256; none missing or changed |
| GitHub automation | Zero workflows and deployments; Pages endpoint returned 404 |
| Local automation | No Windows scheduled tasks referenced MarketCast in executable or arguments |

The smoke run used the bundled BTC CSV, a copied smoke configuration and an absolute scratch output outside the repository. No provider downloads, historical retraining, report regeneration or changes to research artifacts were performed. Test-generated outputs also used separate temporary directories. The private checksum inventory and verifier remain in `.preservation/`; they are not part of the GitHub checkpoint.

The environment was restored on the same Windows x64 machine using Python 3.13.15 and available package-index/cache artifacts. This verifies current recovery, not future package availability, bit-identical model retraining or cross-platform portability. No wheel archive or separate research backup was created. Existing data provenance and redistribution limitations remain; no new data-rights claims are made. Services outside the repository and matching Windows scheduled tasks were not comprehensively inventoried.

The preparation commit and annotated `dormancy-2026-10-03` tag preserve these results. Final remote revision/tag equality and unarchived status are checked after push; the source revision above identifies the code actually validated, rather than claiming this document's own commit hash.
