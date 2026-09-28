"""Read-only access to immutable experiment runs for the UI and report."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import pandas as pd

from market_forecast.data.providers import load_provider
from market_forecast.data import select_asset

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    asset_id: str
    asset_class: str
    directory: Path
    wall_seconds: float

    def json(self, name: str) -> dict | list:
        return _json(self.directory / name)

    def metrics(self) -> pd.DataFrame:
        return pd.read_csv(self.directory / "metrics.csv")

    def predictions(self) -> pd.DataFrame:
        file_name = self.json("data_manifest.json")["prediction_file"]
        path = self.directory / file_name
        frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        return frame

    def source_data(self) -> pd.DataFrame:
        config = self.json("config.json")
        source = _resolve(config["data_path"])
        if not source.exists():
            raise FileNotFoundError(f"Cached source missing: {source}. Run the four-market acquisition workflow.")
        provider = load_provider(config["provider"], source, asset_id=self.asset_id,
                                 **(config.get("provider_options") or {}))
        if provider.manifest["sha256"] != self.json("data_manifest.json")["sha256"]:
            raise ValueError(f"Source snapshot changed since run {self.run_id}; use its saved figures or regenerate the run.")
        frame = select_asset(provider.frame, self.asset_id)
        if config.get("end_date"):
            frame = frame[frame.timestamp <= pd.Timestamp(config["end_date"], tz="UTC")]
        return frame.reset_index(drop=True)


@dataclass(frozen=True)
class ArtifactCatalog:
    root: Path
    runs: tuple[RunRecord, ...]

    def table(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.root / name)

    def run(self, run_id: str) -> RunRecord:
        return next(record for record in self.runs if record.run_id == run_id)

    def for_asset(self, asset_id: str) -> RunRecord:
        return next(record for record in self.runs if record.asset_id == asset_id)


def load_catalog(root: str | Path = "assets/phase3") -> ArtifactCatalog:
    directory = _resolve(root)
    index = _json(directory / "run_index.json")
    records = tuple(RunRecord(entry["run_id"], entry["asset_id"], entry["asset_class"],
                              _resolve(entry["run_dir"]), float(entry["wall_seconds"]))
                    for entry in index)
    if len({run.asset_id for run in records}) != len(records):
        raise ValueError("The run index must contain one run per asset")
    for run in records:
        manifest = run.json("data_manifest.json")
        if manifest["asset_id"] != run.asset_id or not (run.directory / manifest["prediction_file"]).exists():
            raise ValueError(f"Incomplete or inconsistent run: {run.run_id}")
    return ArtifactCatalog(directory, records)
