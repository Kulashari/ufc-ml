"""Point-in-time validation and lookup for latest fighter snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Self

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from ufc_predictor.config import DataConfig
from ufc_predictor.data.loader import dataframe_sha256
from ufc_predictor.data.validation import assert_no_post_fight_columns, require_columns
from ufc_predictor.exceptions import (
    AmbiguousFighterError,
    FighterNotFoundError,
    SnapshotValidationError,
)
from ufc_predictor.features.registry import discover_feature_columns


@dataclass(frozen=True, slots=True)
class SnapshotValidationSummary:
    """Evidence returned after point-in-time snapshot validation."""

    fighter_count: int
    feature_count: int
    as_of_date: date
    last_recorded_fight_date: date | None
    frame_sha256: str


def validate_snapshot_frame(
    frame: pd.DataFrame,
    config: DataConfig,
    *,
    expected_cutoff: date | None = None,
) -> SnapshotValidationSummary:
    """Validate identities, numeric features, and all snapshot date cutoffs."""

    required = (
        config.fighter_id_column,
        config.fighter_name_column,
        config.snapshot_date_column,
        config.last_fight_date_column,
    )
    require_columns(frame, required, asset_name="fighter snapshots")
    if frame.empty:
        raise SnapshotValidationError("Fighter snapshot dataset is empty")

    for column in (
        config.fighter_id_column,
        config.fighter_name_column,
        config.snapshot_date_column,
    ):
        if frame[column].isna().any():
            raise SnapshotValidationError(
                f"Snapshot column {column!r} contains {int(frame[column].isna().sum())} nulls"
            )

    duplicated = frame[config.fighter_id_column].duplicated(keep=False)
    if duplicated.any():
        duplicate_examples = (
            frame.loc[duplicated, config.fighter_id_column]
            .astype(str)
            .drop_duplicates()
            .head(10)
            .tolist()
        )
        raise SnapshotValidationError(
            f"Snapshot fighter IDs must be unique; examples={duplicate_examples}"
        )

    as_of = _strict_dates(
        frame[config.snapshot_date_column],
        config.snapshot_date_column,
        allow_null=False,
    )
    unique_cutoffs = as_of.dt.normalize().unique()
    if len(unique_cutoffs) != 1:
        rendered = sorted(pd.Timestamp(value).date().isoformat() for value in unique_cutoffs)
        raise SnapshotValidationError(
            f"All snapshots must share one as-of date; found {rendered[:10]}"
        )
    actual_cutoff = pd.Timestamp(unique_cutoffs[0]).date()
    required_cutoff = expected_cutoff or config.dataset_cutoff
    if actual_cutoff != required_cutoff:
        raise SnapshotValidationError(
            f"Snapshot as-of date {actual_cutoff} does not match expected cutoff {required_cutoff}"
        )

    last_fight = _strict_dates(
        frame[config.last_fight_date_column],
        config.last_fight_date_column,
        allow_null=True,
    )
    future_fights = last_fight.notna() & last_fight.gt(as_of)
    if future_fights.any():
        future_examples = (
            frame.loc[
                future_fights,
                [config.fighter_id_column, config.last_fight_date_column],
            ]
            .head(10)
            .to_dict("records")
        )
        raise SnapshotValidationError(
            f"Snapshots contain last_fight_date after as_of_date; examples={future_examples}"
        )

    if config.dob_column in frame.columns:
        dob = _strict_dates(frame[config.dob_column], config.dob_column, allow_null=True)
        future_births = dob.notna() & dob.gt(as_of)
        if future_births.any():
            raise SnapshotValidationError(
                f"{int(future_births.sum())} fighter DOBs occur after the snapshot cutoff"
            )

    assert_no_post_fight_columns(frame.columns, target_column=config.target_column)
    features = discover_feature_columns(frame, prefix=config.feature_prefix)
    _validate_snapshot_features(frame, features)
    last_known = last_fight.max()
    return SnapshotValidationSummary(
        fighter_count=len(frame),
        feature_count=len(features),
        as_of_date=actual_cutoff,
        last_recorded_fight_date=(None if pd.isna(last_known) else pd.Timestamp(last_known).date()),
        frame_sha256=dataframe_sha256(frame),
    )


class SnapshotStore:
    """Validated lookup over one common point-in-time fighter snapshot."""

    def __init__(
        self,
        frame: pd.DataFrame,
        config: DataConfig,
        *,
        aliases: dict[str, str] | None = None,
    ) -> None:
        self._summary = validate_snapshot_frame(frame, config)
        self._frame = frame.copy()
        self._config = config
        self._feature_columns = discover_feature_columns(self._frame, prefix=config.feature_prefix)
        self._frame[config.fighter_id_column] = self._frame[config.fighter_id_column].astype(str)
        self._by_id = self._frame.set_index(config.fighter_id_column, drop=False)

        names: dict[str, list[str]] = {}
        name_columns = [
            column
            for column in (config.fighter_name_column, "display_name")
            if column in self._frame.columns
        ]
        lookup_columns = [config.fighter_id_column, *name_columns]
        for _, row in self._frame.loc[:, lookup_columns].iterrows():
            fighter_id = str(row[config.fighter_id_column])
            for column in name_columns:
                value = row[column]
                if pd.notna(value) and str(value).strip():
                    names.setdefault(_normalize_name(str(value)), []).append(fighter_id)
        self._name_to_ids = {name: tuple(dict.fromkeys(ids)) for name, ids in names.items()}
        self._aliases = {
            _normalize_name(alias): destination for alias, destination in (aliases or {}).items()
        }

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        config: DataConfig,
        *,
        aliases: dict[str, str] | None = None,
    ) -> Self:
        """Named constructor for symmetry with future persistence APIs."""

        return cls(frame, config, aliases=aliases)

    @property
    def summary(self) -> SnapshotValidationSummary:
        return self._summary

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self._feature_columns

    @property
    def as_of_date(self) -> date:
        return self._summary.as_of_date

    def get_by_id(self, fighter_id: str) -> pd.Series:
        """Return a defensive copy of one fighter snapshot."""

        key = str(fighter_id).strip()
        if key not in self._by_id.index:
            raise FighterNotFoundError(f"Unknown fighter ID: {fighter_id!r}")
        row = self._by_id.loc[key]
        if isinstance(row, pd.DataFrame):
            raise SnapshotValidationError(
                f"Fighter ID unexpectedly resolved to multiple rows: {key}"
            )
        return row.copy()

    def resolve(self, fighter: str) -> pd.Series:
        """Resolve an exact ID, configured alias, fighter name, or display name."""

        query = str(fighter).strip()
        if not query:
            raise FighterNotFoundError("Fighter identifier cannot be empty")
        if query in self._by_id.index:
            return self.get_by_id(query)

        normalized = _normalize_name(query)
        destination = self._aliases.get(normalized)
        if destination is not None:
            if destination in self._by_id.index:
                return self.get_by_id(destination)
            normalized = _normalize_name(destination)

        matches = self._name_to_ids.get(normalized, ())
        if not matches:
            raise FighterNotFoundError(f"No fighter matches {fighter!r}")
        if len(matches) > 1:
            raise AmbiguousFighterError(
                f"Fighter name {fighter!r} matches multiple IDs: {list(matches)}; "
                "use a fighter ID or configured alias"
            )
        return self.get_by_id(matches[0])

    def pair(
        self,
        fighter_a: str,
        fighter_b: str,
    ) -> tuple[pd.Series, pd.Series]:
        """Resolve two distinct fighters from the configured snapshot table."""
        left = self.resolve(fighter_a)
        right = self.resolve(fighter_b)
        id_column = self._config.fighter_id_column
        if str(left[id_column]) == str(right[id_column]):
            raise SnapshotValidationError("A fighter cannot be matched against itself")
        return left, right


def _strict_dates(
    values: pd.Series,
    column: str,
    *,
    allow_null: bool,
) -> pd.Series:
    converted = pd.to_datetime(values, errors="coerce")
    invalid = converted.isna() & values.notna()
    if invalid.any():
        examples = values.loc[invalid].astype(str).head(10).tolist()
        raise SnapshotValidationError(
            f"Snapshot column {column!r} contains invalid dates: {examples}"
        )
    if not allow_null and converted.isna().any():
        raise SnapshotValidationError(
            f"Snapshot column {column!r} contains {int(converted.isna().sum())} null dates"
        )
    return converted


def _validate_snapshot_features(
    frame: pd.DataFrame,
    features: tuple[str, ...],
) -> None:
    non_numeric = [
        column
        for column in features
        if not (is_numeric_dtype(frame[column]) or is_bool_dtype(frame[column]))
    ]
    if non_numeric:
        raise SnapshotValidationError(f"Snapshot features must be numeric: {non_numeric}")
    nulls = frame.loc[:, list(features)].isna().sum()
    invalid_nulls = {column: int(count) for column, count in nulls.items() if int(count)}
    if invalid_nulls:
        raise SnapshotValidationError(f"Snapshot features contain NaNs: {invalid_nulls}")
    matrix = frame.loc[:, list(features)].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(matrix).all():
        raise SnapshotValidationError("Snapshot features contain infinite/non-finite values")


def _normalize_name(value: str) -> str:
    return " ".join(value.casefold().split())


__all__ = [
    "SnapshotStore",
    "SnapshotValidationSummary",
    "validate_snapshot_frame",
]
