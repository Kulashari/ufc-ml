"""Small local HTTP API for the React matchup-prediction interface.

The API intentionally accepts only matchup inputs.  The artifact directory and
configuration are supplied when the server starts, so a browser client cannot
read arbitrary files from the host machine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ufc_predictor.config import AppConfig, load_config
from ufc_predictor.data import load_fighter_snapshots, validate_snapshot_frame
from ufc_predictor.exceptions import UFCPredictorError
from ufc_predictor.inference.fighter_lookup import (
    AmbiguousFighterError,
    FighterLookup,
    FighterLookupError,
    SnapshotUnavailableError,
)
from ufc_predictor.inference.predictor import SameFighterError
from ufc_predictor.workflows import predict_fight

_FIGHTER_ID_PATTERN = re.compile(r"\b[0-9a-f]{12,}\b", re.IGNORECASE)


class PredictionRequest(BaseModel):
    """Validated inputs accepted from the matchup form."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    fighter_a: str = Field(min_length=1, max_length=160)
    fighter_b: str = Field(min_length=1, max_length=160)
    division: str | None = Field(default=None, max_length=80)

    @field_validator("division")
    @classmethod
    def empty_division_is_none(cls, value: str | None) -> str | None:
        """Treat an empty optional form field as an inferred division."""

        return value or None


class HealthResponse(BaseModel):
    """Non-sensitive server state exposed for the UI's startup check."""

    status: str
    artifact_version: str
    dataset_cutoff: date
    snapshot_start_date: date
    snapshot_end_date: date


@dataclass(frozen=True)
class SnapshotAvailability:
    """Snapshot bounds and identity lookup shared by local API requests."""

    snapshot_start_date: date
    snapshot_end_date: date
    lookup: FighterLookup


def _fighter_name(lookup: FighterLookup, fighter_id: str) -> str:
    """Resolve an internal ID for a user-facing message without exposing it."""

    try:
        candidate = lookup.candidate_for_id(fighter_id)
    except FighterLookupError:
        return "Unknown Fighter"
    return candidate.display_name or candidate.fighter_name or "Unknown Fighter"


def _replace_internal_ids(message: str, lookup: FighterLookup) -> str:
    """Ensure a browser never receives a raw stable fighter identifier."""

    return _FIGHTER_ID_PATTERN.sub(lambda match: _fighter_name(lookup, match.group()), message)


def _friendly_prediction_error(error: Exception, lookup: FighterLookup) -> HTTPException:
    """Turn expected inference failures into user-safe HTTP responses."""

    if isinstance(error, SnapshotUnavailableError):
        fighter_name = _fighter_name(lookup, error.fighter_id)
        if error.available_dates:
            first = error.available_dates[0].isoformat()
            last = error.available_dates[-1].isoformat()
            detail = f" Available snapshot dates run from {first} through {last}."
        else:
            detail = " No usable snapshot is available for this fighter."
        return HTTPException(
            status_code=422,
            detail=(
                f"{fighter_name} has no snapshot available before the current server date "
                f"({error.reference_date.isoformat()}).{detail}"
            ),
        )
    if isinstance(error, AmbiguousFighterError):
        candidates = ", ".join(
            candidate.display_name or candidate.fighter_name or "Unknown Fighter"
            for candidate in error.candidates
        )
        detail = "More than one fighter matches that name. Use a more specific full name."
        if candidates:
            detail = f"{detail} Matches: {candidates}."
        return HTTPException(status_code=422, detail=detail)
    if isinstance(error, SameFighterError):
        return HTTPException(status_code=422, detail="Choose two different fighters.")

    return HTTPException(status_code=422, detail=_replace_internal_ids(str(error), lookup))


def _load_snapshot_availability(config: AppConfig) -> SnapshotAvailability:
    """Validate the configured snapshot table and expose its usable date range."""

    snapshots = load_fighter_snapshots(config.data)
    summary = validate_snapshot_frame(
        snapshots,
        config.data,
        expected_cutoff=config.data.dataset_cutoff,
    )
    return SnapshotAvailability(
        snapshot_start_date=summary.as_of_date,
        snapshot_end_date=summary.as_of_date,
        lookup=FighterLookup(snapshots, aliases=config.inference.aliases),
    )


def _resolve_server_inputs(
    *,
    config_path: str | Path,
    run_dir: str | Path,
) -> tuple[Path, AppConfig]:
    """Resolve and validate fixed server inputs before accepting requests."""

    resolved_config = Path(config_path).expanduser().resolve()
    resolved_run_dir = Path(run_dir).expanduser().resolve()
    if not resolved_config.is_file():
        raise FileNotFoundError(f"configuration file does not exist: {resolved_config}")
    if not resolved_run_dir.is_dir():
        raise FileNotFoundError(f"artifact directory does not exist: {resolved_run_dir}")
    return resolved_run_dir, load_config(resolved_config)


def create_app(
    *,
    config_path: str | Path,
    run_dir: str | Path,
) -> FastAPI:
    """Create an API bound to one trusted model artifact and configuration."""

    resolved_run_dir, config = _resolve_server_inputs(
        config_path=config_path,
        run_dir=run_dir,
    )
    availability = _load_snapshot_availability(config)
    app = FastAPI(
        title="UFC Predictor API",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=("http://127.0.0.1:5173", "http://localhost:5173"),
        allow_credentials=False,
        allow_methods=("GET", "POST"),
        allow_headers=("Content-Type",),
    )

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Confirm that the UI can reach the local prediction service."""

        return HealthResponse(
            status="ok",
            artifact_version=resolved_run_dir.name,
            dataset_cutoff=config.data.dataset_cutoff,
            snapshot_start_date=availability.snapshot_start_date,
            snapshot_end_date=availability.snapshot_end_date,
        )

    @app.post("/api/predict")
    def predict(request: PredictionRequest) -> dict[str, Any]:
        """Run the existing leakage-safe inference workflow for one matchup."""

        try:
            return predict_fight(
                config,
                run_dir=resolved_run_dir,
                fighter_a=request.fighter_a,
                fighter_b=request.fighter_b,
                division=request.division,
            )
        except (UFCPredictorError, OSError, RuntimeError, ValueError) as error:
            raise _friendly_prediction_error(error, availability.lookup) from error

    return app


def run_server(
    *,
    config_path: str | Path,
    run_dir: str | Path,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Run the local API without importing Uvicorn during ordinary CLI use."""

    try:
        import uvicorn
    except ImportError as error:
        raise RuntimeError(
            "Web UI support requires the optional dependencies. "
            'Install them with: python -m pip install -e ".[web]"'
        ) from error

    uvicorn.run(create_app(config_path=config_path, run_dir=run_dir), host=host, port=port)


__all__ = ["PredictionRequest", "create_app", "run_server"]
