from pathlib import Path

import pandas as pd

from market_forecast.data import load_bundled_crypto, select_asset
from market_forecast.features import add_past_features


def load_asset_for_dashboard(
    asset_id: str,
    *,
    path: str | Path = "data/crypto_statistics_data.csv",
    max_points: int | None = None,
) -> pd.DataFrame:
    selected = select_asset(load_bundled_crypto(path), asset_id)
    featured = add_past_features(selected)
    if max_points is not None:
        featured = featured.iloc[-max_points:]
    return featured.reset_index(drop=True)

__all__ = ["load_asset_for_dashboard"]

