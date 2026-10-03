# Dormancy and recovery

Prepared on 2026-10-03 (Asia/Saigon). This educational project is not actively maintained. GitHub was unarchived for preparation and will remain writable. Archiving it again is a separate future action.

## Preserved state

The historical study ends on 2025-10-14. Code supports 14 model families, 1/5/20 observed-session horizons, three validation folds and a locked holdout across BTC, SPY, EUR/USD and WTI spot. The dashboard reads saved artifacts rather than training on page load.

A GitHub clone contains source, tests, configurations, the bundled cryptocurrency CSV, curated phase3/phase4 reports, figures, slides, future scenarios and legacy tracked outputs. It does **not** contain ignored full runs, raw provider responses, generated fixtures or the virtual environment.

The full local research snapshot remains in this project folder: `data/`, `assets/`, `images/` and `configs/`. Nothing was deleted and no separate backup was created. The private `.preservation/` directory contains a dated SHA-256 inventory (relative paths, byte sizes and hashes) and indexed study manifest details. These metadata files are ignored by Git. Credentials, environments and disposable Python/test caches are excluded from the inventory. Local historical recovery depends on retaining this folder; a clone alone cannot restore its missing runs.

All four entries in `assets/phase3/run_index.json` have their configuration, data manifest, metrics, environment and prediction file locally. Raw source paths and rights statements are in their manifests. BTC upstream provenance/license is unknown and Yahoo raw redistribution rights are unverified. ECB citation/transformation requirements and EIA attribution are recorded by the original study. These are preserved provenance statements, not newly reviewed licenses. No additional raw feeds or run files were uploaded.

Three folds and a short holdout limit ranking confidence; validation improvements can reverse on holdout. Predictive uncertainty is not calibrated, trading costs are not modeled, and future scenarios start at the historical cutoff. Provider revisions and different numerical environments can change reruns. Saved predictions and metrics are the historical evidence.

## Environment and startup

`preservation-environment.json` records the validation source revision, Windows build, Python 3.13.15 and CPU PyTorch 2.14.0+cpu (CUDA unavailable). `requirements-dormant.txt` freezes all 73 installed dependencies without editable-install paths. The project itself is installed separately. The original `.venv` is retained but is not portable.

For recovery, use Windows x64 and Python 3.13.15 where possible. From the project root in PowerShell:

```powershell
python -m venv .venv-restore
.venv-restore\Scripts\python.exe -m pip install -r requirements-dormant.txt
.venv-restore\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
.venv-restore\Scripts\python.exe -m pip check
.venv-restore\Scripts\python.exe -m streamlit run src/streamlit.py
```

Installing dependencies needs an available package index or preexisting wheel cache. No wheel archive was created. Exact version pins do not guarantee future package availability or binary identity. Other operating systems/Python versions are unverified. When reusing this folder, use the existing `.venv` command paths instead.

## Validation without replacing historical evidence

The historical audit is read-only and requires the local full runs:

```powershell
.venv\Scripts\python.exe main.py audit-four-markets
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

For an offline smoke check, copy `configs/smoke.json` into a temporary directory and change `artifacts_dir` to a separate absolute scratch directory. Keep `data_path` pointed at the bundled CSV. Run `main.py experiment --config <scratch-config>` with the restored Python. This checks software, not historical replication; do not replace saved reports with its results.

For a synthetic four-market check, run from a separate scratch working directory to isolate the fixture generator's relative `data/fixtures/generated` path. Set `PYTHONPATH` to the repository's `src`, and invoke the absolute `main.py` path with `offline-fixture`, absolute `--config` and `--assets` paths, and a separate `--output`. Synthetic data is not market evidence. Avoid the default study, deliverable and scenario rebuild commands during preservation because they write saved artifact paths.

Before moving to another machine, retain the whole research directory structure and private inventory, transfer through a private channel, and compare each recorded file size/hash after transfer by running `.venv-restore\Scripts\python.exe .preservation/verify.py` from the transferred project. Replace backslash paths in study indexes if moving to a different OS; that platform remains unverified. Rebuild the environment rather than copying `.venv`. Losing the local files prevents historical auditing even though the clone's comparison dashboard and slides remain available.

## Automation and checkpoint

As inspected on 2026-10-03, `.github/workflows/` is empty, GitHub reports zero Actions workflows and deployments, and no Pages site is configured (API 404). No deployment or scheduled execution references were found in the inspected source/configuration. External services outside this repository were not inventoried. No Windows scheduled tasks referenced MarketCast in their executable or arguments. No project automation needed disabling.

The dated annotated tag `dormancy-2026-10-03` identifies the preparation checkpoint. Its commit adds preservation documentation and dependency/environment snapshots; forecasting, CLI and dashboard behavior are unchanged. See `PRESERVATION_VALIDATION.md` for measured results and any recovery limitations. GitHub remains unarchived and public; ignored research files remain local.
