"""Command-line entry points for data acquisition, validation, and training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import RAW_DATA_PATH, VALIDATION_REPORT_PATH, ensure_project_directories
from src.data_loader import download_ames_housing, load_housing_data
from src.training import train_final_model
from src.validation import validate_housing_data


def _write_validation_report(data_path: Path) -> None:
    """Validate one dataset and persist its complete machine-readable report."""
    report = validate_housing_data(load_housing_data(data_path))
    ensure_project_directories()
    VALIDATION_REPORT_PATH.write_text(
        json.dumps(report.as_dict(), indent=2), encoding="utf-8"
    )
    print(json.dumps(report.as_dict(), indent=2))


def _download_command(_: argparse.Namespace) -> None:
    """Implement the download subcommand."""
    data = download_ames_housing()
    print(f"Saved {len(data):,} rows and {len(data.columns):,} columns to {RAW_DATA_PATH}")


def _validate_command(args: argparse.Namespace) -> None:
    """Implement the validate subcommand."""
    _write_validation_report(args.data_path)


def _train_command(args: argparse.Namespace) -> None:
    """Implement the train subcommand and its optional acquisition step."""
    if args.download and not args.data_path.exists():
        download_ames_housing(args.data_path)
    data = load_housing_data(args.data_path)
    _write_validation_report(args.data_path)
    metadata = train_final_model(data)
    print(json.dumps(metadata, indent=2))


def build_parser() -> argparse.ArgumentParser:
    """Build the project CLI parser."""
    parser = argparse.ArgumentParser(description="Ames house-price ML workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="Fetch the OpenML dataset")
    download_parser.set_defaults(handler=_download_command)

    validate_parser = subparsers.add_parser("validate", help="Write a data-quality report")
    validate_parser.add_argument("--data-path", type=Path, default=RAW_DATA_PATH)
    validate_parser.set_defaults(handler=_validate_command)

    train_parser = subparsers.add_parser("train", help="Train and serialize the final pipeline")
    train_parser.add_argument("--data-path", type=Path, default=RAW_DATA_PATH)
    train_parser.add_argument(
        "--download",
        action="store_true",
        help="Download the OpenML data when the requested CSV is absent.",
    )
    train_parser.set_defaults(handler=_train_command)
    return parser


def main() -> None:
    """Dispatch a parsed command-line request."""
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
