"""Compatibility launcher for the study CLI and the older interactive trainer."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys
import warnings

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings("ignore", message="TensorFlow GPU support is not available.*")

STUDY_COMMANDS = {"baseline", "experiment", "compare", "four-markets", "audit-four-markets",
                  "build-deliverables", "build-future-scenarios", "offline-fixture"}


def _launch_streamlit_app() -> None:
    script = SOURCE_ROOT / "streamlit.py"
    if not script.exists():
        print(f"Could not find Streamlit entrypoint: {script}")
        return
    for port in range(8501, 8511):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
        print(f"Starting Streamlit UI at http://localhost:{port}", flush=True)
        print("Streamlit will run in this terminal. Stop it with Ctrl+C.", flush=True)
        try:
            subprocess.run([sys.executable, "-m", "streamlit", "run", str(script),
                            "--server.port", str(port)], cwd=PROJECT_ROOT, check=True)
        except KeyboardInterrupt:
            print("\nStreamlit UI stopped.")
        except subprocess.CalledProcessError as exc:
            print(f"Streamlit exited with status code {exc.returncode}.")
        except Exception as exc:
            print(f"Failed to launch Streamlit UI: {exc}")
        return
    raise RuntimeError("No free port found in range 8501-8510.")


def _prompt_model_selection() -> list[str]:
    from src.models import MODEL_ORDER, MODEL_LABELS, normalize_model_selection

    while True:
        print("\nSelect model scope\n1) All models")
        for index, name in enumerate(MODEL_ORDER, start=2):
            print(f"{index}) {MODEL_LABELS[name]}")
        choice = input("Choice: ").strip().lower()
        if choice in {"1", "all"}:
            return normalize_model_selection("all")
        if choice.isdigit() and 0 <= int(choice) - 2 < len(MODEL_ORDER):
            return normalize_model_selection(MODEL_ORDER[int(choice) - 2])
        selected = normalize_model_selection(choice)
        if selected and len(selected) == 1:
            return selected
        print("Invalid selection. Pick one model or 'all'.")


def _prompt_optimizer_selection() -> str:
    from src.models import SUPPORTED_OPTIMIZERS

    while True:
        print("\nSelect optimizer")
        for index, name in enumerate(SUPPORTED_OPTIMIZERS, start=1):
            print(f"{index}) {name}")
        choice = input("Choice: ").strip().lower()
        if choice.isdigit() and 0 <= int(choice) - 1 < len(SUPPORTED_OPTIMIZERS):
            return SUPPORTED_OPTIMIZERS[int(choice) - 1]
        if choice in SUPPORTED_OPTIMIZERS:
            return choice
        print("Invalid optimizer. Pick one listed option.")


def _train(models: str | None, optimizer: str | None) -> None:
    from src.models import normalize_model_selection, SUPPORTED_OPTIMIZERS
    from src.train import run_pipeline

    selected = normalize_model_selection(models) if models else _prompt_model_selection()
    if not selected:
        raise SystemExit("Invalid models value.")
    if optimizer and optimizer.lower() not in SUPPORTED_OPTIMIZERS:
        raise SystemExit("Invalid optimizer value.")
    run_pipeline(selected, optimizer=optimizer.lower() if optimizer else "adam" if models else _prompt_optimizer_selection())


def _interactive_menu() -> None:
    try:
        while True:
            print("\nMarketCast Lab\n1) Train models\n2) Run Streamlit\n3) Exit")
            choice = input("Choice [1-3]: ").strip().lower()
            if choice == "1":
                _train(None, None)
            elif choice == "2":
                _launch_streamlit_app()
            elif choice in {"3", "q", "quit", "exit"}:
                return
            else:
                print("Invalid choice.")
    except KeyboardInterrupt:
        print("\nInterrupted. Terminating...")


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if arguments and arguments[0] in STUDY_COMMANDS:
        from market_forecast.cli import main as study_main

        return study_main(arguments)
    parser = argparse.ArgumentParser(
        description="MarketCast Lab launcher",
        epilog="Study commands: " + ", ".join(sorted(STUDY_COMMANDS)) + ". Run a command with --help for its options.",
    )
    parser.add_argument("command", nargs="?", choices=("train", "ui"))
    parser.add_argument("--run-pipeline", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--models")
    parser.add_argument("--optimizer")
    args = parser.parse_args(arguments)
    if args.command == "ui":
        _launch_streamlit_app()
    elif args.command == "train" or args.run_pipeline:
        _train(args.models, args.optimizer)
    else:
        _interactive_menu()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
