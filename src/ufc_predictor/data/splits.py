"""Construction and validation of leakage-safe chronological splits."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from ufc_predictor.config import DataConfig, SplitLabels
from ufc_predictor.exceptions import SplitValidationError


@dataclass(frozen=True, slots=True)
class SplitSummary:
    """Compact audit information for one split."""

    label: str
    row_count: int
    date_count: int
    start_date: date
    end_date: date


@dataclass(frozen=True, slots=True)
class SplitFrames:
    """Chronologically ordered train, validation, and test frames."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def construct_chronological_splits(
    frame: pd.DataFrame,
    *,
    date_column: str = "event_date",
    split_column: str = "split",
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    labels: SplitLabels | None = None,
) -> pd.DataFrame:
    """Assign whole event dates to train/validation/test splits.

    Cut points minimize distance to requested row fractions while never placing
    one event date in multiple splits. Input row order is preserved.
    """

    labels = labels or SplitLabels()
    if not 0.0 < train_fraction < 1.0:
        raise SplitValidationError("train_fraction must be between 0 and 1")
    if not 0.0 < validation_fraction < 1.0:
        raise SplitValidationError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1.0:
        raise SplitValidationError("train_fraction + validation_fraction must be less than 1")
    dates = _coerce_dates(frame, date_column)
    counts = dates.value_counts(sort=False).sort_index()
    if len(counts) < 3:
        raise SplitValidationError(
            "At least three distinct event dates are required to construct splits"
        )

    cumulative = counts.cumsum().to_numpy()
    total_rows = int(cumulative[-1])
    train_cut = _closest_valid_cut(
        cumulative,
        target=total_rows * train_fraction,
        minimum_index=0,
        maximum_index=len(counts) - 3,
    )
    validation_cut = _closest_valid_cut(
        cumulative,
        target=total_rows * (train_fraction + validation_fraction),
        minimum_index=train_cut + 1,
        maximum_index=len(counts) - 2,
    )

    unique_dates = counts.index
    train_end = unique_dates[train_cut]
    validation_end = unique_dates[validation_cut]
    assigned = np.select(
        [dates <= train_end, dates <= validation_end],
        [labels.train, labels.validation],
        default=labels.test,
    )
    result = frame.copy()
    result[date_column] = dates
    result[split_column] = assigned
    validate_split_column(
        result,
        date_column=date_column,
        split_column=split_column,
        labels=labels,
    )
    return result


def assign_configured_splits(
    frame: pd.DataFrame,
    config: DataConfig,
) -> pd.DataFrame:
    """Assign splits using the explicit point-in-time boundaries in config."""

    dates = _coerce_dates(frame, config.date_column)
    labels = config.split_labels
    train_mask = dates <= pd.Timestamp(config.train_end)
    validation_mask = dates.between(
        pd.Timestamp(config.validation_start),
        pd.Timestamp(config.validation_end),
        inclusive="both",
    )
    test_mask = dates.between(
        pd.Timestamp(config.test_start),
        pd.Timestamp(config.dataset_cutoff),
        inclusive="both",
    )
    covered = train_mask | validation_mask | test_mask
    if not covered.all():
        uncovered = sorted(dates.loc[~covered].dt.strftime("%Y-%m-%d").unique().tolist())
        raise SplitValidationError(
            f"Configured split boundaries leave event dates unassigned: {uncovered[:10]}"
        )

    result = frame.copy()
    result[config.date_column] = dates
    result[config.split_column] = np.select(
        [train_mask, validation_mask, test_mask],
        [labels.train, labels.validation, labels.test],
        default="",
    )
    validate_configured_splits(result, config)
    return result


def validate_configured_splits(
    frame: pd.DataFrame,
    config: DataConfig,
) -> tuple[SplitSummary, ...]:
    """Validate labels, whole-date isolation, chronology, and configured bounds."""

    summaries = validate_split_column(
        frame,
        date_column=config.date_column,
        split_column=config.split_column,
        labels=config.split_labels,
    )
    dates = _coerce_dates(frame, config.date_column)
    splits = frame[config.split_column].astype(str)
    labels = config.split_labels

    bounds = {
        labels.train: (
            None,
            pd.Timestamp(config.train_end),
        ),
        labels.validation: (
            pd.Timestamp(config.validation_start),
            pd.Timestamp(config.validation_end),
        ),
        labels.test: (
            pd.Timestamp(config.test_start),
            pd.Timestamp(config.dataset_cutoff),
        ),
    }
    violations: list[str] = []
    for label, (minimum, maximum) in bounds.items():
        split_dates = dates.loc[splits.eq(label)]
        if minimum is not None and (split_dates < minimum).any():
            violations.append(f"{label} contains dates before {minimum.date()}")
        if (split_dates > maximum).any():
            violations.append(f"{label} contains dates after {maximum.date()}")
    if violations:
        raise SplitValidationError("; ".join(violations))
    return summaries


def validate_split_column(
    frame: pd.DataFrame,
    *,
    date_column: str = "event_date",
    split_column: str = "split",
    labels: SplitLabels | None = None,
) -> tuple[SplitSummary, ...]:
    """Validate a generic three-way chronological split column."""

    labels = labels or SplitLabels()
    missing_columns = [
        column for column in (date_column, split_column) if column not in frame.columns
    ]
    if missing_columns:
        raise SplitValidationError(f"Cannot validate splits; missing columns: {missing_columns}")
    dates = _coerce_dates(frame, date_column)
    if frame[split_column].isna().any():
        raise SplitValidationError(f"{split_column!r} contains null labels")
    split_values = frame[split_column].astype(str)
    allowed = set(labels.ordered)
    unexpected = sorted(set(split_values.unique()) - allowed)
    if unexpected:
        raise SplitValidationError(f"Unexpected split labels: {unexpected}")

    missing_labels = [label for label in labels.ordered if not split_values.eq(label).any()]
    if missing_labels:
        raise SplitValidationError(f"Empty required splits: {missing_labels}")

    date_split_counts = (
        pd.DataFrame({"date": dates, "split": split_values})
        .drop_duplicates()
        .groupby("date", sort=False)["split"]
        .nunique()
    )
    shared_dates = date_split_counts.loc[date_split_counts.gt(1)].index
    if len(shared_dates):
        examples = [timestamp.date().isoformat() for timestamp in shared_dates[:10]]
        raise SplitValidationError(f"Event dates occur in multiple splits: {examples}")

    summaries: list[SplitSummary] = []
    previous_end: pd.Timestamp | None = None
    for label in labels.ordered:
        split_dates = dates.loc[split_values.eq(label)]
        start = split_dates.min()
        end = split_dates.max()
        if previous_end is not None and start <= previous_end:
            raise SplitValidationError(
                f"Split {label!r} starts on {start.date()}, which is not after "
                f"the previous split end {previous_end.date()}"
            )
        summaries.append(
            SplitSummary(
                label=label,
                row_count=len(split_dates),
                date_count=int(split_dates.nunique()),
                start_date=start.date(),
                end_date=end.date(),
            )
        )
        previous_end = end
    return tuple(summaries)


def split_frames(
    frame: pd.DataFrame,
    config: DataConfig,
    *,
    copy: bool = True,
) -> SplitFrames:
    """Validate then materialize train/validation/test dataframes."""

    validate_configured_splits(frame, config)
    labels = config.split_labels

    def select(label: str) -> pd.DataFrame:
        selected = frame.loc[frame[config.split_column].astype(str).eq(label)]
        return selected.copy() if copy else selected

    return SplitFrames(
        train=select(labels.train),
        validation=select(labels.validation),
        test=select(labels.test),
    )


def _coerce_dates(frame: pd.DataFrame, date_column: str) -> pd.Series:
    if date_column not in frame.columns:
        raise SplitValidationError(f"Missing split date column: {date_column!r}")
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    if dates.isna().any():
        count = int(dates.isna().sum())
        raise SplitValidationError(f"{date_column!r} contains {count} null or invalid dates")
    return dates


def _closest_valid_cut(
    cumulative: np.ndarray,
    *,
    target: float,
    minimum_index: int,
    maximum_index: int,
) -> int:
    candidates = np.arange(minimum_index, maximum_index + 1)
    distances = np.abs(cumulative[candidates] - target)
    return int(candidates[int(np.argmin(distances))])


__all__ = [
    "SplitFrames",
    "SplitSummary",
    "assign_configured_splits",
    "construct_chronological_splits",
    "split_frames",
    "validate_configured_splits",
    "validate_split_column",
]
