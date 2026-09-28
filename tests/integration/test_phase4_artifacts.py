from __future__ import annotations

from pathlib import Path

import pytest

from market_forecast.dashboard.artifacts import load_catalog
from market_forecast.publication import build_scientific_report
from market_forecast.reports import audit_cross_market_report


ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "artifacts" / "phase3" / "run_index.json"


def test_missing_artifact_index_is_explicit(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_catalog(tmp_path)


@pytest.mark.skipif(not INDEX.exists(), reason="Local Phase 3 runs are generated and Git-ignored")
def test_saved_metrics_and_report_trace_to_same_runs(tmp_path: Path) -> None:
    catalog = load_catalog(INDEX.parent)
    audit = audit_cross_market_report(catalog.root)
    assert audit == {"runs": 4, "metric_rows_reproduced": 672}
    report = build_scientific_report(catalog, tmp_path / "report.md").read_text(encoding="utf-8")
    for run in catalog.runs:
        assert run.run_id in report
        assert run.asset_id in report
        assert run.predictions().asset_id.unique().tolist() == [run.asset_id]
