"""Lightweight installation and configuration check; never fits a model."""

from __future__ import annotations

import argparse
import json
import platform
from importlib import metadata
from pathlib import Path

from ufc_predictor.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Configuration file to load.",
    )
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    packages = (
        "joblib",
        "numpy",
        "pandas",
        "pydantic",
        "PyYAML",
        "scikit-learn",
        "typer",
        "xgboost",
    )
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    print(
        json.dumps(
            {
                "status": "ready",
                "python": platform.python_version(),
                "config": str(arguments.config.resolve()),
                "project_root": str(config.project_root),
                "packages": versions,
                "training_started": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
