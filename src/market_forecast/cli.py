from __future__ import annotations

import argparse
from pathlib import Path

from market_forecast.config import BaselineExperimentConfig, ExperimentConfig
from market_forecast.experiments import run_btc_baseline_experiment, run_experiment
from market_forecast.reports import build_comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MarketCast Lab shared CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    baseline = subparsers.add_parser("baseline", help="Run the Phase 1 BTC baseline experiment")
    baseline.add_argument("--data", type=Path, default=Path("data/crypto_statistics_data.csv"))
    baseline.add_argument("--output", type=Path, default=Path("artifacts/runs/btc-baseline-smoke"))
    experiment = subparsers.add_parser("experiment", help="Run a Phase 2 experiment config")
    experiment.add_argument("--config", type=Path, required=True)
    compare = subparsers.add_parser("compare", help="Regenerate comparison outputs for a run")
    compare.add_argument("--run", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "baseline":
        output = run_btc_baseline_experiment(
            BaselineExperimentConfig(data_path=args.data, output_dir=args.output)
        )
        print(f"Baseline artifacts written to {output}")
    elif args.command == "experiment":
        output = run_experiment(ExperimentConfig.from_json(args.config))
        print(f"Experiment artifacts written to {output}")
    elif args.command == "compare":
        print(build_comparison(args.run).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

