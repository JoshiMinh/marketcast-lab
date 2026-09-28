from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json


@dataclass(frozen=True)
class BaselineExperimentConfig:
    data_path: Path = Path("data/crypto_statistics_data.csv")
    asset_id: str = "crypto:BTC-USD"
    output_dir: Path = Path("assets/runs/btc-baseline-smoke")
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    seasonal_period: int | None = 7
    seed: int = 42

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["data_path"] = str(self.data_path)
        values["output_dir"] = str(self.output_dir)
        return values


@dataclass(frozen=True)
class ExperimentConfig:
    data_path: Path = Path("data/crypto_statistics_data.csv")
    provider: str = "crypto_csv"
    end_date: str | None = None
    artifacts_dir: Path = Path("assets/runs")
    asset_id: str = "crypto:BTC-USD"
    models: tuple[str, ...] = ("last_value", "drift", "seasonal_naive")
    horizons: tuple[int, ...] = (1, 5, 20)
    folds: int = 3
    final_test_size: int = 20
    min_train_size: int = 365
    seasonal_period: int = 7
    strategy: str = "recursive"
    seed: int = 42
    max_train_rows: int | None = 730
    model_params: dict[str, dict[str, object]] | None = None
    tuning_grid: dict[str, list[dict[str, object]]] | None = None
    provider_options: dict[str, str] | None = None

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["data_path"] = str(self.data_path)
        values["artifacts_dir"] = str(self.artifacts_dir)
        return values

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in ("data_path", "artifacts_dir"):
            if key in payload:
                payload[key] = Path(payload[key])
        for key in ("models", "horizons"):
            if key in payload:
                payload[key] = tuple(payload[key])
        return cls(**payload)


@dataclass(frozen=True)
class AssetSpec:
    asset_id: str
    provider: str
    data_path: Path
    seasonal_period: int = 5
    provider_options: dict[str, str] = field(default_factory=dict)


def load_assets(path: str | Path) -> tuple[AssetSpec, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assets = tuple(AssetSpec(
        asset_id=item["asset_id"], provider=item["provider"], data_path=Path(item["data_path"]),
        seasonal_period=int(item.get("seasonal_period", 5)),
        provider_options=item.get("provider_options", {}),
    ) for item in payload["assets"])
    ids = [asset.asset_id for asset in assets]
    if not assets or len(ids) != len(set(ids)):
        raise ValueError("Asset catalog must contain unique asset IDs")
    if any(":" not in asset.asset_id or asset.seasonal_period < 2 for asset in assets):
        raise ValueError("Asset IDs need a class prefix and seasonal periods must be at least 2")
    return assets

