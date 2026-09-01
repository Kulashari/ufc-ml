"""Strict configuration loading for the standalone latest-data CLI."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ufc_ml_latestdatafetcher.errors import FetcherConfigurationError


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceConfig(_StrictModel):
    base_url: str = "http://ufcstats.com"
    completed_events_path: str = "/statistics/events/completed?page=all"
    fighter_directory_path: str = "/statistics/fighters?char={letter}&page=all"
    user_agent: str = "Mozilla/5.0 (compatible; UFCMLLatestDataFetcher/0.1)"
    timeout_seconds: int = Field(default=60, ge=10, le=180)
    retry_attempts: int = Field(default=3, ge=1, le=6)
    delay_seconds: float = Field(default=2.0, ge=0.0, le=30.0)
    jitter_seconds: float = Field(default=0.5, ge=0.0, le=10.0)
    headless: bool = True
    minimum_completed_event_count: int = Field(default=700, ge=1)
    earliest_completed_event_date: date = date(1994, 3, 11)
    minimum_fighter_directory_count: int = Field(default=4000, ge=1)

    @field_validator("base_url")
    @classmethod
    def base_url_is_supported(cls, value: str) -> str:
        normalized = value.strip().rstrip(",/")
        if normalized != "http://ufcstats.com":
            raise ValueError("base_url must be http://ufcstats.com")
        return normalized

    @field_validator("completed_events_path", "fighter_directory_path", "user_agent")
    @classmethod
    def strings_are_not_empty(cls, value: str) -> str:
        normalized = value.strip().rstrip(",")
        if not normalized:
            raise ValueError("source strings cannot be empty")
        return normalized


class StorageConfig(_StrictModel):
    raw_html_dir: Path = Path("data/raw/ufcstats/html")
    manifest_dir: Path = Path("data/raw/ufcstats/manifests")
    normalized_dir: Path = Path("data/interim/ufcstats")
    candidate_dir: Path = Path("data/candidates/latestdatafetcher")
    baseline_fights_path: Path = Path("data/ufc_gold_dataset_final.csv")
    baseline_fighters_path: Path = Path("data/ufc_fighters_final.csv")

    def resolve_paths(self, root: Path) -> Self:
        def resolve(path: Path) -> Path:
            expanded = path.expanduser()
            return expanded.resolve() if expanded.is_absolute() else (root / expanded).resolve()

        return self.model_copy(
            update={name: resolve(getattr(self, name)) for name in type(self).model_fields}
        )


class FetcherConfig(_StrictModel):
    source: SourceConfig = Field(default_factory=SourceConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    model_config_path: Path = Path("configs/production-rolling-2026.yaml")
    project_root: Path | None = Field(default=None, exclude=True)

    def resolve_paths(self, root: Path) -> Self:
        resolved_root = root.expanduser().resolve()
        model_path = self.model_config_path.expanduser()
        if not model_path.is_absolute():
            model_path = resolved_root / model_path
        return self.model_copy(
            update={
                "storage": self.storage.resolve_paths(resolved_root),
                "model_config_path": model_path.resolve(),
                "project_root": resolved_root,
            }
        )


def _discover_project_root(path: Path) -> Path:
    for candidate in (path.parent, *path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


def load_fetcher_config(
    path: str | Path = "configs/latestdatafetcher.yaml",
    *,
    project_root: str | Path | None = None,
) -> FetcherConfig:
    """Load a fetcher config and resolve every local path against the project root."""

    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    if not config_path.is_file():
        raise FetcherConfigurationError(f"Configuration file does not exist: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw: Any = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise FetcherConfigurationError(f"Could not read {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise FetcherConfigurationError(f"Configuration root must be a mapping: {config_path}")

    try:
        config = FetcherConfig.model_validate(raw)
    except ValidationError as exc:
        raise FetcherConfigurationError(f"Invalid configuration {config_path}:\n{exc}") from exc

    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else _discover_project_root(config_path).resolve()
    )
    return config.resolve_paths(root)


__all__ = ["FetcherConfig", "SourceConfig", "StorageConfig", "load_fetcher_config"]
