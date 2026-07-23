"""Strict validation for the processed, pre-fight training dataset."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from ufc_predictor.config import DataConfig
from ufc_predictor.data.loader import dataframe_sha256
from ufc_predictor.data.splits import SplitSummary, validate_configured_splits
from ufc_predictor.exceptions import (
    DataValidationError,
    LeakageValidationError,
    SchemaValidationError,
)
from ufc_predictor.features.registry import FeatureRegistry

FORBIDDEN_POST_FIGHT_COLUMNS = frozenset(
    {
        "winner",
        "method",
        "end_round",
        "end_time",
        "total_fight_time_sec",
        "fighter_1",
        "fighter_2",
        "wins",
        "losses",
        "draws",
        "slpm",
        "sapm",
        "str_acc",
        "str_def",
        "td_avg",
        "td_acc",
        "td_def",
        "sub_avg",
    }
)
FORBIDDEN_POST_FIGHT_PATTERNS = (
    re.compile(r"^(?:f1|f2)_", flags=re.IGNORECASE),
    re.compile(r"(?:^|_)post_?fight(?:_|$)", flags=re.IGNORECASE),
    re.compile(r"(?:^|_)fight_result(?:_|$)", flags=re.IGNORECASE),
)


@dataclass(frozen=True, slots=True)
class DatasetValidationSummary:
    """Evidence returned after all strict dataset checks pass."""

    row_count: int
    feature_count: int
    positive_count: int
    negative_count: int
    first_event_date: date
    last_event_date: date
    frame_sha256: str
    splits: tuple[SplitSummary, ...]


def validate_model_dataset(
    frame: pd.DataFrame,
    config: DataConfig,
    *,
    registry: FeatureRegistry | None = None,
    enforce_expected_counts: bool = True,
) -> DatasetValidationSummary:
    """Validate the complete training-ready dataset or raise a clear error."""

    required_metadata = (
        config.fight_id_column,
        config.date_column,
        config.split_column,
        config.target_column,
        "fighter_a_id",
        "fighter_a_name",
        "fighter_b_id",
        "fighter_b_name",
        "division",
        "is_title_bout",
    )
    require_columns(frame, required_metadata, asset_name="model dataset")
    assert_no_post_fight_columns(frame.columns, target_column=config.target_column)

    feature_registry = registry or FeatureRegistry.discover(
        frame,
        prefix=config.feature_prefix,
        exclude=(config.target_column,),
    )
    feature_registry.assert_matches(frame, prefix=config.feature_prefix)
    features = feature_registry.names
    if enforce_expected_counts and len(features) != config.expected_feature_count:
        raise SchemaValidationError(
            f"Expected {config.expected_feature_count} model features, found {len(features)}"
        )
    if enforce_expected_counts and len(frame) != config.expected_fight_count:
        raise DataValidationError(
            f"Expected {config.expected_fight_count} fights, found {len(frame)}"
        )

    _validate_non_null_metadata(frame, required_metadata)
    _validate_unique_ids(frame, config.fight_id_column, "fight")
    _validate_fighter_pair(frame)
    _validate_binary_target(frame[config.target_column], config.target_column)
    _validate_feature_matrix(frame, features)
    splits = validate_configured_splits(frame, config)

    dates = pd.to_datetime(frame[config.date_column], errors="coerce")
    if dates.max().date() > config.dataset_cutoff:
        raise DataValidationError(
            f"Dataset extends past configured cutoff {config.dataset_cutoff}: {dates.max().date()}"
        )
    target = pd.to_numeric(frame[config.target_column], errors="raise")
    return DatasetValidationSummary(
        row_count=len(frame),
        feature_count=len(features),
        positive_count=int(target.eq(1).sum()),
        negative_count=int(target.eq(0).sum()),
        first_event_date=dates.min().date(),
        last_event_date=dates.max().date(),
        frame_sha256=dataframe_sha256(frame),
        splits=splits,
    )


def validate_feature_dictionary(
    dictionary: pd.DataFrame,
    *,
    target_column: str,
    expected_feature_count: int | None = None,
) -> FeatureRegistry:
    """Validate roles and pre-fight availability, then return the registry."""

    registry = FeatureRegistry.from_dictionary(dictionary)
    target_rows = dictionary.loc[dictionary["column"].astype(str).eq(target_column)]
    if len(target_rows) != 1:
        raise SchemaValidationError(
            f"Feature dictionary must define target {target_column!r} exactly once"
        )
    target_row = target_rows.iloc[0]
    if str(target_row["role"]) != "target":
        raise SchemaValidationError(f"Dictionary row {target_column!r} must have role='target'")
    if _coerce_bool(target_row["available_pre_fight"], target_column):
        raise LeakageValidationError(
            f"Target {target_column!r} cannot be marked available pre-fight"
        )
    if expected_feature_count is not None and len(registry) != expected_feature_count:
        raise SchemaValidationError(
            f"Expected {expected_feature_count} dictionary features, found {len(registry)}"
        )
    assert_no_post_fight_columns(registry.names, target_column=target_column)
    return registry


def assert_no_post_fight_columns(
    columns: Iterable[str],
    *,
    target_column: str,
) -> None:
    """Reject known outcome, in-fight, and current-career leakage columns."""

    leaked: list[str] = []
    for original in columns:
        normalized = str(original).strip().casefold()
        if normalized == target_column.casefold():
            continue
        if normalized.startswith("target_"):
            leaked.append(str(original))
            continue
        if normalized in FORBIDDEN_POST_FIGHT_COLUMNS or any(
            pattern.search(normalized) for pattern in FORBIDDEN_POST_FIGHT_PATTERNS
        ):
            leaked.append(str(original))
    if leaked:
        raise LeakageValidationError(
            f"Post-fight/current-career columns are forbidden in model data: {sorted(leaked)}"
        )


def require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    asset_name: str,
) -> None:
    """Require a set of named columns."""

    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise SchemaValidationError(f"{asset_name} is missing required columns: {missing}")


def _validate_non_null_metadata(
    frame: pd.DataFrame,
    columns: Iterable[str],
) -> None:
    nulls = {
        column: int(frame[column].isna().sum()) for column in columns if frame[column].isna().any()
    }
    if nulls:
        raise DataValidationError(f"Required columns contain nulls: {nulls}")


def _validate_unique_ids(frame: pd.DataFrame, column: str, entity: str) -> None:
    duplicate_mask = frame[column].duplicated(keep=False)
    if duplicate_mask.any():
        examples = frame.loc[duplicate_mask, column].astype(str).drop_duplicates().head(10).tolist()
        raise DataValidationError(
            f"{entity.capitalize()} IDs must be unique; duplicate examples={examples}"
        )


def _validate_fighter_pair(frame: pd.DataFrame) -> None:
    same = frame["fighter_a_id"].astype(str).eq(frame["fighter_b_id"].astype(str))
    if same.any():
        examples = frame.loc[same, "fighter_a_id"].astype(str).head(10).tolist()
        raise DataValidationError(f"A fighter cannot face itself; invalid fighter IDs={examples}")


def _validate_binary_target(target: pd.Series, column: str) -> None:
    if target.isna().any():
        raise DataValidationError(f"Target {column!r} contains {int(target.isna().sum())} nulls")
    numeric = pd.to_numeric(target, errors="coerce")
    invalid_numeric = numeric.isna()
    if invalid_numeric.any():
        examples = target.loc[invalid_numeric].astype(str).head(10).tolist()
        raise DataValidationError(f"Target {column!r} contains non-numeric values: {examples}")
    values = set(numeric.unique())
    if not values.issubset({0, 1}):
        raise DataValidationError(
            f"Target {column!r} must be binary {{0, 1}}, found {sorted(values)}"
        )
    if values != {0, 1}:
        raise DataValidationError(
            f"Target {column!r} must contain both classes, found {sorted(values)}"
        )


def _validate_feature_matrix(
    frame: pd.DataFrame,
    features: tuple[str, ...],
) -> None:
    non_numeric = [
        column
        for column in features
        if not (is_numeric_dtype(frame[column]) or is_bool_dtype(frame[column]))
    ]
    if non_numeric:
        raise SchemaValidationError(
            f"All model features must be numeric; invalid columns={non_numeric}"
        )

    null_counts = frame.loc[:, list(features)].isna().sum()
    with_nulls = {column: int(count) for column, count in null_counts.items() if int(count) > 0}
    if with_nulls:
        raise DataValidationError(f"Model features contain NaNs: {with_nulls}")

    values = frame.loc[:, list(features)].to_numpy(dtype=np.float64, copy=False)
    finite = np.isfinite(values)
    if not finite.all():
        bad_counts = (~finite).sum(axis=0)
        invalid = {
            feature: int(count)
            for feature, count in zip(features, bad_counts, strict=True)
            if int(count) > 0
        }
        raise DataValidationError(f"Model features contain infinite/non-finite values: {invalid}")


def _coerce_bool(value: object, column: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float) and not pd.isna(value) and value in (0, 1):
        return bool(value)
    normalized = str(value).strip().casefold()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise SchemaValidationError(f"Invalid boolean value for {column!r}: {value!r}")


__all__ = [
    "FORBIDDEN_POST_FIGHT_COLUMNS",
    "DatasetValidationSummary",
    "assert_no_post_fight_columns",
    "require_columns",
    "validate_feature_dictionary",
    "validate_model_dataset",
]
