"""Repository launcher for the MarketCast Lab study CLI."""
from __future__ import annotations

from pathlib import Path
import sys

SOURCE_ROOT = Path(__file__).resolve().parent / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from market_forecast.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
