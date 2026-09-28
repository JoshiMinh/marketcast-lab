import argparse
import os
import subprocess
import sys
import warnings
import socket
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

warnings.filterwarnings("ignore", message="TensorFlow GPU support is not available.*")


def _find_available_port(start_port=8501, max_port=8510):
    for port in range(start_port, max_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No free port found in range {start_port}-{max_port}.")


def terminate_program(message="Terminating..."):
    print(message)
    raise SystemExit(0)


def _launch_streamlit_app():
    project_root = Path(__file__).resolve().parent
    streamlit_script = project_root / "src" / "streamlit.py"

    if not streamlit_script.exists():
        print(f"Could not find Streamlit entrypoint: {streamlit_script}")
        return

    streamlit_port = _find_available_port()
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(streamlit_script),
        "--server.port",
        str(streamlit_port),
    ]

    print(f"Starting Streamlit UI at http://localhost:{streamlit_port}", flush=True)
    print("Streamlit will run in this terminal. Stop it with Ctrl+C.", flush=True)
    try:
        subprocess.run(command, cwd=str(project_root), check=True)
    except KeyboardInterrupt:
        print("\nStreamlit UI stopped.")
    except subprocess.CalledProcessError as exc:
        print(f"Streamlit exited with status code {exc.returncode}.")
    except Exception as exc:
        print(f"Failed to launch Streamlit UI: {exc}")


def _prompt_model_selection():
    from src.models import MODEL_ORDER, MODEL_LABELS, normalize_model_selection

    while True:
        print("\nSelect model scope")
        print("1) All models")
        for index, model_name in enumerate(MODEL_ORDER, start=2):
            print(f"{index}) {MODEL_LABELS[model_name]}")

        choice = input("Choice: ").strip().lower()
        if choice in {"1", "all"}:
            return normalize_model_selection("all")

        if choice.isdigit():
            numeric_choice = int(choice)
            mapped_index = numeric_choice - 2
            if 0 <= mapped_index < len(MODEL_ORDER):
                return normalize_model_selection(MODEL_ORDER[mapped_index])

        normalized = normalize_model_selection(choice)
        if normalized and len(normalized) == 1:
            return normalized

        print("Invalid selection. Pick one model or 'all'.")


def _prompt_optimizer_selection() -> str:
    from src.models import SUPPORTED_OPTIMIZERS, normalize_optimizer

    while True:
        print("\nSelect optimizer")
        for index, optimizer_name in enumerate(SUPPORTED_OPTIMIZERS, start=1):
            print(f"{index}) {optimizer_name}")

        choice = input("Choice: ").strip().lower()
        if choice.isdigit():
            numeric_choice = int(choice) - 1
            if 0 <= numeric_choice < len(SUPPORTED_OPTIMIZERS):
                return SUPPORTED_OPTIMIZERS[numeric_choice]

        if choice in SUPPORTED_OPTIMIZERS:
            return choice

        normalized = normalize_optimizer(choice, default="")
        if normalized:
            return normalized

        print("Invalid optimizer. Pick one listed option.")


def interactive_menu():
    try:
        while True:
            print("\nMarketCast Lab")
            print("1) Train models")
            print("2) Run Streamlit")
            print("3) Exit")
            choice = input("Choice [1-3]: ").strip().lower()

            if choice == '1':
                selected_models = _prompt_model_selection()
                selected_optimizer = _prompt_optimizer_selection()
                print("Loading training pipeline...", flush=True)
                from src.train import run_pipeline

                run_pipeline(selected_models, optimizer=selected_optimizer)
            elif choice == '2':
                _launch_streamlit_app()
            elif choice in {'3', 'q', 'quit', 'exit'}:
                terminate_program()
            else:
                print("Invalid choice.")
    except KeyboardInterrupt:
        terminate_program("\nInterrupted. Terminating...")


def main():
    parser = argparse.ArgumentParser(description="MarketCast Lab CLI")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Run training pipeline")
    train_parser.add_argument("--models", help="Comma-separated models: all,lstm,gru,arima,prophet,ensemble")
    train_parser.add_argument("--optimizer", help="Sequence model optimizer: adam,rmsprop")

    subparsers.add_parser("ui", help="Launch Streamlit UI")
    baseline_parser = subparsers.add_parser("baseline", help="Run the leakage-safe Phase 1 BTC baseline")
    baseline_parser.add_argument("--output", default="artifacts/runs/btc-baseline-smoke")
    experiment_parser = subparsers.add_parser("experiment", help="Run a Phase 2 experiment configuration")
    experiment_parser.add_argument("--config", required=True)
    compare_parser = subparsers.add_parser("compare", help="Regenerate a Phase 2 run comparison")
    compare_parser.add_argument("--run", required=True)
    four_parser = subparsers.add_parser("four-markets", help="Run the Phase 3 four-asset matrix")
    four_parser.add_argument("--config", default="configs/four_asset_full.json")
    four_parser.add_argument("--output", default="artifacts/phase3")
    audit_parser = subparsers.add_parser("audit-four-markets", help="Recompute Phase 3 metrics from predictions")
    audit_parser.add_argument("--output", default="artifacts/phase3")
    report_parser = subparsers.add_parser("build-deliverables", help="Build Phase 4 report and presentation from runs")
    report_parser.add_argument("--index", default="artifacts/phase3")
    report_parser.add_argument("--output", default="artifacts/phase4")
    future_parser = subparsers.add_parser("build-future-scenarios", help="Explicitly fit scenario forecasts for the UI")
    future_parser.add_argument("--index", default="artifacts/phase3")
    future_parser.add_argument("--output", default="artifacts/phase4")
    fixture_parser = subparsers.add_parser("offline-fixture", help="Build deterministic synthetic four-market demo runs")
    fixture_parser.add_argument("--output", default="artifacts/offline-fixture")

    # Backward-compatible flags from older CLI versions.
    parser.add_argument("--run-pipeline", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--models", dest="legacy_models", help=argparse.SUPPRESS)
    parser.add_argument("--optimizer", dest="legacy_optimizer", help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.command == "ui":
        _launch_streamlit_app()
        return

    if args.command == "baseline":
        from pathlib import Path
        from market_forecast.config import BaselineExperimentConfig
        from market_forecast.experiments import run_btc_baseline_experiment

        output = run_btc_baseline_experiment(
            BaselineExperimentConfig(output_dir=Path(args.output))
        )
        print(f"Baseline artifacts written to {output}")
        return

    if args.command == "experiment":
        from market_forecast.config import ExperimentConfig
        from market_forecast.experiments import run_experiment

        output = run_experiment(ExperimentConfig.from_json(args.config))
        print(f"Experiment artifacts written to {output}")
        return

    if args.command == "compare":
        from market_forecast.reports import build_comparison

        print(build_comparison(args.run).to_string(index=False))
        return

    if args.command == "four-markets":
        from market_forecast.experiments.four_markets import run_four_markets

        print(f"Cross-market artifacts written to {run_four_markets(args.config, args.output)}")
        return

    if args.command == "audit-four-markets":
        from market_forecast.reports import audit_cross_market_report

        print(audit_cross_market_report(args.output))
        return

    if args.command == "build-deliverables":
        from market_forecast.publication import build_deliverables

        print(f"Report and presentation written to {build_deliverables(args.index, args.output)}")
        return

    if args.command == "build-future-scenarios":
        from market_forecast.publication import build_future_scenarios

        print(f"Scenario forecasts written to {build_future_scenarios(args.index, args.output)}")
        return

    if args.command == "offline-fixture":
        from market_forecast.experiments.offline_fixture import build_offline_fixture

        print(f"Synthetic fixture artifacts written to {build_offline_fixture(output=args.output)}")
        return

    should_train = args.command == "train" or args.run_pipeline
    if not should_train:
        interactive_menu()
        return

    from src.models import normalize_model_selection, normalize_optimizer, SUPPORTED_OPTIMIZERS
    from src.train import run_pipeline

    raw_models = getattr(args, "models", None) or args.legacy_models
    raw_optimizer = getattr(args, "optimizer", None) or args.legacy_optimizer

    if raw_models:
        selected_models = normalize_model_selection(raw_models)
    else:
        selected_models = _prompt_model_selection()

    if raw_optimizer:
        selected_optimizer = normalize_optimizer(raw_optimizer)
    elif raw_models:
        selected_optimizer = "adam"
    else:
        selected_optimizer = _prompt_optimizer_selection()

    if raw_models and not selected_models:
        print("Invalid models value.")
        raise SystemExit(1)
    if raw_optimizer and raw_optimizer.strip().lower() not in SUPPORTED_OPTIMIZERS:
        print("Invalid optimizer value.")
        raise SystemExit(1)

    run_pipeline(selected_models, optimizer=selected_optimizer)


if __name__ == "__main__":
    main()
