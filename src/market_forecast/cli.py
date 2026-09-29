from __future__ import annotations

import argparse
from pathlib import Path
import sys

from market_forecast.config import BaselineExperimentConfig, ExperimentConfig
from market_forecast.experiments import run_btc_baseline_experiment, run_experiment
from market_forecast.reports import build_comparison, audit_cross_market_report
from market_forecast.experiments.four_markets import run_four_markets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MarketCast Lab — run experiments and inspect saved results.",
        epilog=(
            "Quick start:\n"
            "  python main.py --help\n"
            "  python main.py experiment --config configs/smoke.json\n"
            "  python main.py four-markets\n"
            "  python main.py audit-four-markets"
        ),
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog, max_help_position=30, width=100
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    baseline = subparsers.add_parser("baseline", help="Run a BTC baseline")
    baseline.add_argument("--data", type=Path, default=Path("data/crypto_statistics_data.csv"))
    baseline.add_argument("--output", type=Path, default=Path("assets/runs/btc-baseline-smoke"))
    experiment = subparsers.add_parser("experiment", help="Run an experiment from a config")
    experiment.add_argument("--config", type=Path, required=True)
    compare = subparsers.add_parser("compare", help="Rebuild comparisons for one run")
    compare.add_argument("--run", type=Path, required=True)
    four = subparsers.add_parser("four-markets", help="Run the four-market study")
    four.add_argument("--config", type=Path, default=Path("configs/four_asset_full.json"))
    four.add_argument("--output", type=Path, default=Path("assets/phase3"))
    four.add_argument("--assets", type=Path, default=Path("configs/assets_four.json"))
    audit = subparsers.add_parser("audit-four-markets", help="Audit saved four-market predictions")
    audit.add_argument("--output", type=Path, default=Path("assets/phase3"))
    publication = subparsers.add_parser("build-deliverables", help="Build the report and slides")
    publication.add_argument("--index", type=Path, default=Path("assets/phase3"))
    publication.add_argument("--output", type=Path, default=Path("assets/phase4"))
    future = subparsers.add_parser("build-future-scenarios", help="Refit and save future scenarios")
    future.add_argument("--index", type=Path, default=Path("assets/phase3"))
    future.add_argument("--output", type=Path, default=Path("assets/phase4"))
    fixture = subparsers.add_parser("offline-fixture", help="Build a synthetic offline demo")
    fixture.add_argument("--config", type=Path, default=Path("configs/four_asset_full.json"))
    fixture.add_argument("--output", type=Path, default=Path("assets/offline-fixture"))
    fixture.add_argument("--assets", type=Path, default=Path("configs/assets_four.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    # Treat an empty invocation as a request for help, while preserving normal
    # argument parsing for both direct calls and the command-line launcher.
    if argv is None and len(sys.argv) == 1:
        argv = ["--help"]
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
    elif args.command == "four-markets":
        print(f"Cross-market artifacts written to {run_four_markets(args.config, args.output, args.assets)}")
    elif args.command == "audit-four-markets":
        print(audit_cross_market_report(args.output))
    elif args.command == "build-deliverables":
        from market_forecast.publication import build_deliverables

        print(build_deliverables(args.index, args.output))
    elif args.command == "build-future-scenarios":
        from market_forecast.publication import build_future_scenarios

        print(build_future_scenarios(args.index, args.output))
    elif args.command == "offline-fixture":
        from market_forecast.experiments.offline_fixture import build_offline_fixture

        print(build_offline_fixture(config_path=args.config, output=args.output, assets_path=args.assets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

