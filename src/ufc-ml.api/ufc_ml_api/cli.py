"""Explicit command-line entry points for UFC predictor workflows."""

from __future__ import annotations

import json
from dataclasses import asdict
from enum import StrEnum
from pathlib import Path
from typing import Any

import typer

from ufc_ml_core import __version__
from ufc_ml_core.config import load_config
from ufc_ml_core.data import (
    assign_configured_splits,
    load_dataset_bundle,
    validate_feature_dictionary,
    validate_model_dataset,
    validate_snapshot_frame,
)
from ufc_ml_core.exceptions import DataValidationError, UFCPredictorError

app = typer.Typer(
    name="ufc-predictor",
    help="Leakage-safe UFC probability training and matchup inference.",
    no_args_is_help=True,
    invoke_without_command=True,
    pretty_exceptions_show_locals=False,
)
data_app = typer.Typer(help="Load and validate configured datasets.", no_args_is_help=True)
app.add_typer(data_app, name="data")


class ModelChoice(StrEnum):
    LOGISTIC = "logistic"
    XGBOOST = "xgboost"
    ALL = "all"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _emit(payload: dict[str, Any]) -> None:
    typer.echo(json.dumps(_json_ready(payload), indent=2, sort_keys=True))


def _abort(exc: Exception) -> None:
    typer.echo(f"Error: {exc}", err=True)
    raise typer.Exit(code=2) from exc


@app.callback()
def root(
    version: bool = typer.Option(
        False,
        "--version",
        help="Show the installed package version and exit.",
        is_eager=True,
    ),
) -> None:
    """UFC model development commands; all fitting requires ``train``."""

    if version:
        typer.echo(__version__)
        raise typer.Exit()


@data_app.command("validate")
def validate_data_command(
    config_path: Path = typer.Option(
        Path("configs/default.yaml"),
        "--config",
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="YAML project configuration.",
    ),
    enforce_expected_counts: bool = typer.Option(
        True,
        "--enforce-counts/--allow-count-drift",
        help="Require configured fight and feature counts.",
    ),
) -> None:
    """Validate data, splits, point-in-time snapshots, and fingerprints only."""

    try:
        config = load_config(config_path)
        bundle = load_dataset_bundle(config.data)
        model_frame = bundle.model.frame
        if config.data.split_column not in model_frame.columns:
            model_frame = assign_configured_splits(model_frame, config.data)

        registry = validate_feature_dictionary(
            bundle.feature_dictionary.frame,
            target_column=config.data.target_column,
            expected_feature_count=(
                config.data.expected_feature_count if enforce_expected_counts else None
            ),
        )
        dataset_summary = validate_model_dataset(
            model_frame,
            config.data,
            registry=registry,
            enforce_expected_counts=enforce_expected_counts,
        )
        snapshot_summary = validate_snapshot_frame(
            bundle.snapshots.frame,
            config.data,
            expected_cutoff=config.data.dataset_cutoff,
        )

        profile_ids = set(bundle.profiles.frame[config.data.fighter_id_column].astype(str))
        snapshot_ids = set(bundle.snapshots.frame[config.data.fighter_id_column].astype(str))
        if len(profile_ids) != len(bundle.profiles.frame):
            raise DataValidationError("Fighter profiles contain duplicate fighter IDs")
        if profile_ids != snapshot_ids:
            missing_snapshots = sorted(profile_ids - snapshot_ids)
            missing_profiles = sorted(snapshot_ids - profile_ids)
            raise DataValidationError(
                "Profile/snapshot fighter IDs disagree: "
                f"missing_snapshots={missing_snapshots[:10]}, "
                f"missing_profiles={missing_profiles[:10]}"
            )

        _emit(
            {
                "status": "valid",
                "config": str(config_path),
                "dataset": asdict(dataset_summary),
                "snapshots": asdict(snapshot_summary),
                "profiles": {"fighter_count": len(profile_ids)},
                "feature_registry": {
                    "count": len(registry),
                    "sha256": registry.fingerprint,
                },
                "file_sha256": bundle.fingerprints,
                "training_started": False,
            }
        )
    except (UFCPredictorError, ValueError, OSError) as exc:
        _abort(exc)


@app.command("train")
def train_command(
    model: ModelChoice = typer.Option(
        ...,
        "--model",
        case_sensitive=False,
        help="Model family to fit; 'all' compares both on validation only.",
    ),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"),
        "--config",
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="YAML project configuration.",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Artifact version directory name; defaults to a UTC timestamp.",
    ),
    tune_xgboost: bool = typer.Option(
        False,
        "--tune-xgboost/--no-tune-xgboost",
        help="Run the bounded optional Optuna search before the final XGBoost fit.",
    ),
    ablation: bool = typer.Option(
        False,
        "--ablation/--no-ablation",
        help="Run validation-only feature-group ablations after model selection.",
    ),
) -> None:
    """Explicitly fit on train, select/calibrate on validation, and save a run."""

    try:
        from ufc_ml_core.workflows import train_model

        config = load_config(config_path)
        result = train_model(
            config,
            model=model.value,
            run_id=run_id,
            tune_xgboost=tune_xgboost,
            ablation=ablation,
        )
        _emit(
            {
                "status": "trained",
                "selected_model": result.selected_model,
                "validation_log_loss": result.validation_log_loss,
                "calibration_fit_log_loss": result.calibration_fit_log_loss,
                "selection_rationale": result.selection_rationale,
                "run_dir": str(result.run_dir),
                "validation_report_json": str(result.report_json),
                "validation_report_markdown": str(result.report_markdown),
                "calibration_fit_report_json": str(result.calibration_report_json),
                "calibration_fit_report_markdown": str(result.calibration_report_markdown),
                "final_test_evaluated": False,
            }
        )
    except (UFCPredictorError, ImportError, OSError, RuntimeError, ValueError) as exc:
        _abort(exc)


@app.command("evaluate-final")
def evaluate_final_command(
    run_dir: Path = typer.Option(
        ...,
        "--run-dir",
        exists=True,
        file_okay=False,
        readable=True,
        resolve_path=True,
        help="Trusted versioned artifact directory.",
    ),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"),
        "--config",
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="YAML project configuration.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite/--no-overwrite",
        help="Replace a prior final-test result only when explicitly requested.",
    ),
) -> None:
    """Explicitly score the untouched chronological test split once."""

    try:
        from ufc_ml_core.workflows import evaluate_final

        config = load_config(config_path)
        result = evaluate_final(config, run_dir=run_dir, overwrite=overwrite)
        _emit(
            {
                "status": "final_test_evaluated",
                "run_dir": str(result.run_dir),
                "test_rows": result.test_rows,
                "test_log_loss": result.test_log_loss,
                "test_report_json": str(result.report_json),
                "test_report_markdown": str(result.report_markdown),
            }
        )
    except (UFCPredictorError, OSError, RuntimeError, ValueError) as exc:
        _abort(exc)


@app.command("serve")
def serve_command(
    run_dir: Path = typer.Option(
        ...,
        "--run-dir",
        file_okay=False,
        readable=True,
        help=(
            "Trusted versioned artifact directory. Keep this relative to the private asset bundle "
            "when UFC_ML_ASSETS_REPOSITORY is configured."
        ),
    ),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"),
        "--config",
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="YAML project configuration.",
    ),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Interface to bind; use the loopback default for local use.",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        min=1,
        max=65535,
        help="Local HTTP port for the React UI API.",
    ),
) -> None:
    """Serve the selected trusted artifact to the local React prediction UI."""

    try:
        from ufc_ml_api.api import run_server

        run_server(
            config_path=config_path,
            run_dir=run_dir,
            host=host,
            port=port,
        )
    except ImportError:
        _abort(
            RuntimeError(
                "Web UI support requires optional dependencies. "
                'Install them with: python -m pip install -e ".[web]"'
            )
        )
    except (UFCPredictorError, OSError, RuntimeError, ValueError) as exc:
        _abort(exc)


@app.command("predict")
def predict_command(
    fighter_a: str = typer.Option(..., "--fighter-a", help="First fighter name."),
    fighter_b: str = typer.Option(..., "--fighter-b", help="Second fighter name."),
    run_dir: Path = typer.Option(
        ...,
        "--run-dir",
        exists=True,
        file_okay=False,
        readable=True,
        resolve_path=True,
        help="Trusted versioned artifact directory.",
    ),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"),
        "--config",
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="YAML project configuration.",
    ),
    fighter_a_id: str | None = typer.Option(
        None,
        "--fighter-a-id",
        help="Stable ID used to resolve an ambiguous first fighter name.",
    ),
    fighter_b_id: str | None = typer.Option(
        None,
        "--fighter-b-id",
        help="Stable ID used to resolve an ambiguous second fighter name.",
    ),
    division: str | None = typer.Option(
        None,
        "--division",
        help="Division code or label when it cannot be inferred.",
    ),
    division_lbs: float | None = typer.Option(
        None,
        "--division-lbs",
        min=1.0,
        help="Explicit weight for catchweight or unsupported division context.",
    ),
    womens: bool | None = typer.Option(
        None,
        "--womens/--mens",
        help="Gender context for an explicit nonstandard division.",
    ),
    catch_weight: bool | None = typer.Option(
        None,
        "--catch-weight/--not-catch-weight",
        help="Mark explicitly supplied division context as catchweight.",
    ),
    include_features: bool = typer.Option(
        False,
        "--include-features/--no-features",
        help="Include both constructed feature rows in the JSON response.",
    ),
) -> None:
    """Predict a matchup using the current UTC server timestamp."""

    try:
        from ufc_ml_core.workflows import predict_fight

        config = load_config(config_path)
        result = predict_fight(
            config,
            run_dir=run_dir,
            fighter_a=fighter_a,
            fighter_b=fighter_b,
            fighter_a_id=fighter_a_id,
            fighter_b_id=fighter_b_id,
            division=division,
            division_lbs=division_lbs,
            is_womens=womens,
            is_catch_weight=catch_weight,
            include_features=include_features,
        )
        _emit(result)
    except (UFCPredictorError, OSError, RuntimeError, ValueError) as exc:
        _abort(exc)


if __name__ == "__main__":
    app()
