"""Small local HTTP API for the React matchup-prediction interface.

The API intentionally accepts only matchup inputs.  The artifact directory and
configuration are supplied when the server starts, so a browser client cannot
read arbitrary files from the host machine.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from logging import getLogger
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ufc_ml_core.config import AppConfig, load_config
from ufc_ml_core.data import load_fighter_snapshots, validate_snapshot_frame
from ufc_ml_core.exceptions import UFCPredictorError
from ufc_ml_core.inference.fighter_lookup import (
    AmbiguousFighterError,
    FighterCandidate,
    FighterLookup,
    FighterLookupError,
    SnapshotUnavailableError,
)
from ufc_ml_core.inference.predictor import SameFighterError, UnsupportedMatchupError
from ufc_ml_core.workflows import predict_fight

from .assets import checkout_configured_assets

_FIGHTER_ID_PATTERN = re.compile(r"\b[0-9a-f]{12,}\b", re.IGNORECASE)
_LOCAL_CORS_ORIGINS = ("http://127.0.0.1:5173", "http://localhost:5173")
_LOGGER = getLogger(__name__)
_PUBLIC_PREDICTION_FIELDS = (
    "predicted_at",
    "model_cutoff",
    "dataset_cutoff",
    "division",
    "probability_a",
    "probability_b",
    "prior_ufc_fights_a",
    "prior_ufc_fights_b",
    "snapshot_date_a",
    "snapshot_date_b",
    "predicted_winner_name",
    "is_even_probability",
    "orientation_disagreement",
    "confidence_tier",
)


class PredictionRequest(BaseModel):
    """Validated inputs accepted from the matchup form."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    fighter_a: str = Field(min_length=1, max_length=160)
    fighter_b: str = Field(min_length=1, max_length=160)
    fighter_a_id: str | None = Field(default=None, max_length=256)
    fighter_b_id: str | None = Field(default=None, max_length=256)
    division: str | None = Field(default=None, max_length=80)

    @field_validator("fighter_a_id", "fighter_b_id", "division")
    @classmethod
    def empty_optional_value_is_none(cls, value: str | None) -> str | None:
        """Treat an empty optional form field as absent."""

        return value or None


class FighterOptionResponse(BaseModel):
    """Minimal fighter identity returned to an autocomplete control."""

    id: str
    name: str


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


def _cors_origins_from_environment() -> tuple[str, ...]:
    """Return exact browser origins without allowing wildcard CORS access."""

    configured = os.environ.get("UFC_ML_CORS_ORIGINS")
    if configured is None:
        return _LOCAL_CORS_ORIGINS
    raw_origins = [value.strip() for value in configured.split(",") if value.strip()]
    if not raw_origins:
        raise ValueError("UFC_ML_CORS_ORIGINS must contain at least one exact origin.")

    validated: list[str] = []
    for raw_origin in raw_origins:
        if raw_origin == "*":
            raise ValueError("UFC_ML_CORS_ORIGINS cannot contain a wildcard origin.")
        origin = raw_origin.rstrip("/")
        parsed = urlparse(origin)
        try:
            _port = parsed.port
        except ValueError as error:
            raise ValueError(
                "UFC_ML_CORS_ORIGINS values must use valid http(s) origins."
            ) from error
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.path
            or parsed.params
            or parsed.query
            or parsed.fragment
            or parsed.username
            or parsed.password
        ):
            raise ValueError(
                "UFC_ML_CORS_ORIGINS values must be exact http(s) origins without a path."
            )
        if origin not in validated:
            validated.append(origin)
    return tuple(validated)


def _public_fighter(value: object) -> dict[str, str]:
    """Return only names needed by the browser; stable IDs stay server-side."""

    source = value if isinstance(value, Mapping) else {}
    fighter_name = str(source.get("fighter_name") or "").strip()
    display_name = str(source.get("display_name") or fighter_name).strip()
    safe_name = fighter_name or display_name or "Unknown Fighter"
    return {"fighter_name": safe_name, "display_name": display_name or safe_name}


def _public_warnings(value: object) -> list[dict[str, str]]:
    """Expose only user-facing warning fields, never internal diagnostic details."""

    if not isinstance(value, list):
        return []
    warnings: list[dict[str, str]] = []
    for candidate in value:
        if not isinstance(candidate, Mapping):
            continue
        code = str(candidate.get("code") or "").strip()
        severity = str(candidate.get("severity") or "").strip()
        message = str(candidate.get("message") or "").strip()
        if code and severity and message:
            warnings.append({"code": code, "severity": severity, "message": message})
    return warnings


def _public_prediction(result: Mapping[str, Any]) -> dict[str, Any]:
    """Create the browser contract without paths, IDs, model internals, or features."""

    public = {
        field_name: result[field_name]
        for field_name in _PUBLIC_PREDICTION_FIELDS
        if field_name in result
    }
    warnings = _public_warnings(result.get("warnings"))
    confidence_source = result.get("confidence")
    confidence = confidence_source if isinstance(confidence_source, Mapping) else {}
    public["fighter_a"] = _public_fighter(result.get("fighter_a"))
    public["fighter_b"] = _public_fighter(result.get("fighter_b"))
    public["warnings"] = warnings
    public["confidence"] = {
        "tier": str(confidence.get("tier") or result.get("confidence_tier") or "reduced"),
        "score": confidence.get("score"),
        "orientation_disagreement": confidence.get(
            "orientation_disagreement",
            result.get("orientation_disagreement"),
        ),
        "warnings": _public_warnings(confidence.get("warnings")) or warnings,
    }
    return public


def _fighter_name(lookup: FighterLookup, fighter_id: str) -> str:
    """Resolve an internal ID for a user-facing message without exposing it."""

    try:
        candidate = lookup.candidate_for_id(fighter_id)
    except FighterLookupError:
        return "Unknown Fighter"
    return candidate.display_name or candidate.fighter_name or "Unknown Fighter"


def _fighter_option(candidate: FighterCandidate) -> FighterOptionResponse:
    """Return a stable lookup ID and safe canonical display name."""

    name = candidate.display_name or candidate.fighter_name or "Unknown Fighter"
    return FighterOptionResponse(id=candidate.fighter_id, name=name)


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
    if isinstance(error, (FighterLookupError, UnsupportedMatchupError)):
        return HTTPException(
            status_code=422,
            detail=_replace_internal_ids(str(error), lookup),
        )

    return HTTPException(
        status_code=500,
        detail="The prediction service could not complete this request. Please try again later.",
    )


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
    if not resolved_config.is_file():
        raise FileNotFoundError(f"configuration file does not exist: {resolved_config}")
    checkout = checkout_configured_assets()
    requested_run_dir = Path(run_dir).expanduser()
    if checkout is None:
        resolved_run_dir = requested_run_dir.resolve()
        config = load_config(resolved_config)
    else:
        if requested_run_dir.is_absolute():
            raise ValueError(
                "When private assets are configured, --run-dir must be relative to "
                "the asset bundle."
            )
        resolved_run_dir = (checkout.root / requested_run_dir).resolve()
        try:
            resolved_run_dir.relative_to(checkout.root)
        except ValueError as error:
            raise ValueError("--run-dir must stay inside the private asset bundle.") from error
        config = load_config(resolved_config, project_root=checkout.root)
    if not resolved_run_dir.is_dir():
        raise FileNotFoundError(f"artifact directory does not exist: {resolved_run_dir}")
    return resolved_run_dir, config


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
        allow_origins=_cors_origins_from_environment(),
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

    @app.get("/api/fighters", response_model=list[FighterOptionResponse])
    def search_fighters(
        query: Annotated[str, Query(min_length=1, max_length=160)],
        limit: Annotated[int, Query(ge=1, le=20)] = 8,
    ) -> list[FighterOptionResponse]:
        """Return canonical fighter names for the matchup autocomplete controls."""

        return [
            _fighter_option(candidate)
            for candidate in availability.lookup.search(query, limit=limit)
        ]

    @app.post("/api/predict")
    def predict(request: PredictionRequest) -> dict[str, Any]:
        """Run the existing leakage-safe inference workflow for one matchup."""

        try:
            result = predict_fight(
                config,
                run_dir=resolved_run_dir,
                fighter_a=request.fighter_a,
                fighter_b=request.fighter_b,
                fighter_a_id=request.fighter_a_id,
                fighter_b_id=request.fighter_b_id,
                division=request.division,
            )
            return _public_prediction(result)
        except (UFCPredictorError, OSError, RuntimeError, ValueError) as error:
            _LOGGER.exception(
                "Prediction failed for configured artifact %s.",
                resolved_run_dir.name,
            )
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


__all__ = ["FighterOptionResponse", "PredictionRequest", "create_app", "run_server"]
