"""Typed project configuration and YAML loading.

Paths in the checked-in configuration are project-root relative. ``load_config``
resolves them once so downstream code does not depend on the process working
directory.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from ufc_ml_core.exceptions import ConfigurationError


class StrictModel(BaseModel):
    """Common settings for immutable, typo-resistant configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectMetadataConfig(StrictModel):
    """Project-wide metadata and reproducibility settings."""

    name: str = "ufc-predictor"
    random_seed: int = 42


class SplitLabels(StrictModel):
    """Configured names for the three chronological splits."""

    train: str = "train"
    validation: str = "validation"
    test: str = "test"

    @model_validator(mode="after")
    def labels_are_distinct(self) -> Self:
        values = (self.train, self.validation, self.test)
        if len(set(values)) != len(values):
            raise ValueError("train, validation, and test labels must be distinct")
        if any(not value.strip() for value in values):
            raise ValueError("split labels cannot be empty")
        return self

    @property
    def ordered(self) -> tuple[str, str, str]:
        """Return labels in chronological order."""

        return (self.train, self.validation, self.test)


class DataConfig(StrictModel):
    """Paths, schema names, and point-in-time boundaries for data assets."""

    model_dataset_path: Path
    fighter_snapshots_path: Path
    fighter_profiles_path: Path
    feature_dictionary_path: Path
    expected_fight_count: int = Field(gt=0)
    expected_feature_count: int = Field(gt=0)
    dataset_cutoff: date
    train_end: date
    validation_start: date
    validation_end: date
    test_start: date
    date_column: str = "event_date"
    split_column: str = "split"
    target_column: str = "target_a_win"
    fight_id_column: str = "fight_id"
    feature_prefix: str = "feature_"
    fighter_id_column: str = "fighter_id"
    fighter_name_column: str = "fighter_name"
    snapshot_date_column: str = "as_of_date"
    last_fight_date_column: str = "last_fight_date"
    dob_column: str = "dob"
    split_labels: SplitLabels = Field(default_factory=SplitLabels)

    @field_validator(
        "date_column",
        "split_column",
        "target_column",
        "fight_id_column",
        "feature_prefix",
        "fighter_id_column",
        "fighter_name_column",
        "snapshot_date_column",
        "last_fight_date_column",
        "dob_column",
    )
    @classmethod
    def schema_names_are_not_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("schema column names and feature_prefix cannot be empty")
        return value

    @model_validator(mode="after")
    def chronological_boundaries_are_valid(self) -> Self:
        if not (
            self.train_end
            < self.validation_start
            <= self.validation_end
            < self.test_start
            <= self.dataset_cutoff
        ):
            raise ValueError(
                "expected train_end < validation_start <= validation_end "
                "< test_start <= dataset_cutoff"
            )
        return self

    def resolve_paths(self, root: Path) -> Self:
        """Return a copy with all data paths resolved against ``root``."""

        root = root.expanduser().resolve()
        updates: dict[str, Path] = {}
        for field_name in (
            "model_dataset_path",
            "fighter_snapshots_path",
            "fighter_profiles_path",
            "feature_dictionary_path",
        ):
            configured = getattr(self, field_name).expanduser()
            updates[field_name] = (
                configured.resolve() if configured.is_absolute() else (root / configured).resolve()
            )
        return self.model_copy(update=updates)


class LogisticConfig(StrictModel):
    """Hyperparameter space for the calibrated linear baseline."""

    penalty: Literal["l2", "elasticnet", "both"] = "both"
    c_values: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
    l1_ratios: tuple[float, ...] = (0.1, 0.5, 0.9)
    max_iter: int = Field(default=5000, ge=100, le=20_000)
    tolerance: float = Field(default=0.0001, ge=1e-8, le=0.1)
    class_weight: Literal["balanced"] | None = None

    @field_validator("c_values")
    @classmethod
    def c_values_are_positive(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if not values or any(not 1e-4 <= value <= 1e3 for value in values):
            raise ValueError("c_values must be within [1e-4, 1e3]")
        return values

    @field_validator("l1_ratios")
    @classmethod
    def l1_ratios_are_probabilities(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("l1_ratios must be between 0 and 1")
        return values

    @model_validator(mode="after")
    def candidate_grid_is_bounded(self) -> Self:
        elastic_count = (
            len(self.c_values) * len(self.l1_ratios)
            if self.penalty in {"elasticnet", "both"}
            else 0
        )
        l2_count = len(self.c_values) if self.penalty in {"l2", "both"} else 0
        if self.penalty in {"elasticnet", "both"} and not self.l1_ratios:
            raise ValueError("elastic-net search requires at least one l1_ratio")
        if l2_count + elastic_count > 24:
            raise ValueError("logistic search cannot exceed 24 candidates")
        return self


class XGBoostSearchSpace(StrictModel):
    """Bounds used when tuning the boosted-tree candidate."""

    learning_rate: tuple[float, float]
    max_depth: tuple[int, int]
    min_child_weight: tuple[float, float]
    subsample: tuple[float, float]
    colsample_bytree: tuple[float, float]
    gamma: tuple[float, float]
    reg_alpha: tuple[float, float]
    reg_lambda: tuple[float, float]
    max_bin: tuple[int, int]
    n_estimators: tuple[int, int]

    @model_validator(mode="after")
    def ranges_are_ordered(self) -> Self:
        limits: dict[str, tuple[float, float]] = {
            "learning_rate": (0.005, 0.3),
            "max_depth": (2.0, 10.0),
            "min_child_weight": (0.0, 50.0),
            "subsample": (0.5, 1.0),
            "colsample_bytree": (0.5, 1.0),
            "gamma": (0.0, 20.0),
            "reg_alpha": (0.0, 100.0),
            "reg_lambda": (0.0, 100.0),
            "max_bin": (32.0, 2048.0),
            "n_estimators": (50.0, 5000.0),
        }
        for name in type(self).model_fields:
            low, high = getattr(self, name)
            if low > high:
                raise ValueError(f"{name} lower bound cannot exceed upper bound")
            minimum, maximum = limits[name]
            if low < minimum or high > maximum:
                raise ValueError(f"{name} bounds must stay within [{minimum:g}, {maximum:g}]")
        for name in ("learning_rate", "min_child_weight", "reg_lambda"):
            low, high = getattr(self, name)
            if low <= 0 or low >= high:
                raise ValueError(f"{name} tuning bounds require 0 < lower < upper")
        return self


class XGBoostConfig(StrictModel):
    """Boosted-tree defaults and optional tuning controls."""

    device: Literal["auto", "cpu", "cuda"] = "auto"
    learning_rate: float = Field(default=0.04, ge=0.005, le=0.3)
    max_depth: int = Field(default=4, ge=2, le=10)
    min_child_weight: float = Field(default=3.0, ge=0.0, le=50.0)
    subsample: float = Field(default=0.85, ge=0.5, le=1.0)
    colsample_bytree: float = Field(default=0.85, ge=0.5, le=1.0)
    gamma: float = Field(default=0.0, ge=0.0, le=20.0)
    reg_alpha: float = Field(default=0.0, ge=0.0, le=100.0)
    reg_lambda: float = Field(default=2.0, ge=0.0, le=100.0)
    max_bin: int = Field(default=256, ge=32, le=2048)
    n_estimators: int = Field(default=1000, ge=50, le=5000)
    early_stopping_rounds: int = Field(default=50, ge=5, le=500)
    tuning_trials: int = Field(default=30, ge=0, le=100)
    search_space: XGBoostSearchSpace


class CalibrationConfig(StrictModel):
    """Probability calibration settings."""

    method: Literal["none", "sigmoid", "isotonic"] = "sigmoid"
    n_bins: int = Field(default=10, ge=2, le=100)


class SelectionConfig(StrictModel):
    """Guardrails used when selecting a deployable model."""

    log_loss_tie_tolerance: float = Field(default=0.005, ge=0.0, le=0.02)
    max_calibration_error_regression: float = Field(default=0.01, ge=0.0, le=0.1)
    max_subgroup_log_loss_regression: float = Field(default=0.02, ge=0.0, le=0.2)


class InferenceConfig(StrictModel):
    """Prediction-time identity, freshness, and symmetry settings."""

    aliases: dict[str, str] = Field(default_factory=dict)
    stale_after_days: int = Field(default=180, ge=0)
    limited_history_threshold: int = Field(default=3, ge=0)
    orientation_disagreement_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    allow_post_cutoff_prediction: bool = True


class ArtifactConfig(StrictModel):
    """Locations and format version for persisted model artifacts."""

    root_dir: Path = Path("artifacts")
    reports_dir: Path = Path("reports")
    format_version: Literal[1] = 1

    def resolve_paths(self, root: Path) -> Self:
        """Return a copy with output directories resolved against ``root``."""

        root = root.expanduser().resolve()

        def resolve(path: Path) -> Path:
            path = path.expanduser()
            return path.resolve() if path.is_absolute() else (root / path).resolve()

        return self.model_copy(
            update={
                "root_dir": resolve(self.root_dir),
                "reports_dir": resolve(self.reports_dir),
            }
        )


class AppConfig(StrictModel):
    """Complete application configuration."""

    project: ProjectMetadataConfig
    data: DataConfig
    logistic: LogisticConfig
    xgboost: XGBoostConfig
    calibration: CalibrationConfig
    selection: SelectionConfig
    inference: InferenceConfig
    artifacts: ArtifactConfig
    project_root: Path | None = Field(default=None, exclude=True)

    def resolve_paths(self, root: Path) -> Self:
        """Return a copy whose configured paths are absolute."""

        resolved_root = root.expanduser().resolve()
        return self.model_copy(
            update={
                "data": self.data.resolve_paths(resolved_root),
                "artifacts": self.artifacts.resolve_paths(resolved_root),
                "project_root": resolved_root,
            }
        )


def _discover_project_root(config_path: Path) -> Path:
    """Find the nearest parent containing ``pyproject.toml``."""

    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


def load_config(
    path: str | Path = "configs/default.yaml",
    *,
    project_root: str | Path | None = None,
) -> AppConfig:
    """Load, validate, and resolve a YAML configuration file.

    Raises:
        ConfigurationError: If the file cannot be read or fails schema validation.
    """

    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw: Any = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Could not read configuration {config_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigurationError(f"Configuration root must be a mapping: {config_path}")

    try:
        config = AppConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid configuration {config_path}:\n{exc}") from exc

    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else _discover_project_root(config_path).resolve()
    )
    return config.resolve_paths(root)


__all__ = [
    "AppConfig",
    "ArtifactConfig",
    "CalibrationConfig",
    "DataConfig",
    "InferenceConfig",
    "LogisticConfig",
    "ProjectMetadataConfig",
    "SelectionConfig",
    "SplitLabels",
    "XGBoostConfig",
    "XGBoostSearchSpace",
    "load_config",
]
