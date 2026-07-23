"""CSV loading and reproducible SHA256 fingerprints for processed assets."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from ufc_predictor.config import DataConfig
from ufc_predictor.exceptions import (
    DataLoadError,
    FingerprintMismatchError,
    SchemaValidationError,
)

DEFAULT_HASH_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class LoadedFrame:
    """A dataframe together with the exact source-file fingerprint."""

    frame: pd.DataFrame
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    """All processed data assets needed for training and inference."""

    model: LoadedFrame
    snapshots: LoadedFrame
    profiles: LoadedFrame
    feature_dictionary: LoadedFrame

    @property
    def fingerprints(self) -> dict[str, str]:
        """Return stable artifact names mapped to their file digests."""

        return {
            "model_dataset": self.model.sha256,
            "fighter_snapshots": self.snapshots.sha256,
            "fighter_profiles": self.profiles.sha256,
            "feature_dictionary": self.feature_dictionary.sha256,
        }


def file_sha256(
    path: str | Path,
    *,
    chunk_size: int = DEFAULT_HASH_CHUNK_SIZE,
) -> str:
    """Compute a streaming SHA256 digest for one file."""

    resolved = Path(path).expanduser().resolve()
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not resolved.is_file():
        raise DataLoadError(f"Data file does not exist: {resolved}")

    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
    except OSError as exc:
        raise DataLoadError(f"Could not fingerprint {resolved}: {exc}") from exc
    return digest.hexdigest()


def dataframe_sha256(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    include_index: bool = True,
) -> str:
    """Hash dataframe values, column order, and dtypes deterministically.

    File fingerprints are the primary provenance check. This semantic digest is
    useful after an in-memory transformation where no corresponding file exists.
    """

    selected = frame if columns is None else frame.loc[:, list(columns)]
    digest = hashlib.sha256()
    digest.update("\x1f".join(map(str, selected.columns)).encode("utf-8"))
    digest.update(b"\x1e")
    digest.update("\x1f".join(str(dtype) for dtype in selected.dtypes).encode("utf-8"))
    digest.update(b"\x1e")
    row_hashes = pd.util.hash_pandas_object(
        selected, index=include_index, categorize=True
    ).to_numpy(dtype=np.uint64, copy=False)
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def verify_file_sha256(path: str | Path, expected: str) -> str:
    """Verify a configured SHA256 and return the normalized actual digest."""

    normalized = expected.strip().casefold()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("expected SHA256 must be a 64-character hexadecimal string")
    actual = file_sha256(path)
    if actual != normalized:
        raise FingerprintMismatchError(
            f"SHA256 mismatch for {Path(path)}: expected {normalized}, got {actual}"
        )
    return actual


def load_csv(
    path: str | Path,
    *,
    parse_dates: Iterable[str] = (),
    required_columns: Iterable[str] = (),
    dtype: Any = None,
) -> pd.DataFrame:
    """Load one CSV with clear path, parse, and minimum-schema errors."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise DataLoadError(f"Configured data file does not exist: {resolved}")

    try:
        frame = pd.read_csv(resolved, dtype=dtype, low_memory=False)
    except (OSError, UnicodeDecodeError, EmptyDataError, ParserError, ValueError) as exc:
        raise DataLoadError(f"Could not load CSV {resolved}: {exc}") from exc

    required = tuple(required_columns)
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise SchemaValidationError(f"{resolved.name} is missing required columns: {missing}")

    for column in parse_dates:
        if column not in frame.columns:
            raise SchemaValidationError(
                f"Cannot parse absent date column {column!r} in {resolved.name}"
            )
        converted = pd.to_datetime(frame[column], errors="coerce")
        invalid = converted.isna() & frame[column].notna()
        if invalid.any():
            examples = frame.loc[invalid, column].astype(str).head(5).tolist()
            raise SchemaValidationError(
                f"{resolved.name}.{column} contains {int(invalid.sum())} invalid "
                f"dates; examples={examples}"
            )
        frame[column] = converted
    return frame


def load_csv_asset(
    path: str | Path,
    *,
    parse_dates: Iterable[str] = (),
    required_columns: Iterable[str] = (),
    dtype: Any = None,
) -> LoadedFrame:
    """Load a CSV and attach the SHA256 of its exact bytes."""

    resolved = Path(path).expanduser().resolve()
    digest = file_sha256(resolved)
    frame = load_csv(
        resolved,
        parse_dates=parse_dates,
        required_columns=required_columns,
        dtype=dtype,
    )
    return LoadedFrame(frame=frame, path=resolved, sha256=digest)


def load_model_dataset(config: DataConfig) -> pd.DataFrame:
    """Load the training-ready fight dataset."""

    required = (
        config.fight_id_column,
        config.date_column,
        config.target_column,
        "fighter_a_id",
        "fighter_b_id",
    )
    return load_csv(
        config.model_dataset_path,
        parse_dates=(config.date_column,),
        required_columns=required,
    )


def load_fighter_snapshots(config: DataConfig) -> pd.DataFrame:
    """Load latest point-in-time fighter features."""

    required = (
        config.fighter_id_column,
        config.fighter_name_column,
        config.snapshot_date_column,
        config.last_fight_date_column,
    )
    return load_csv(
        config.fighter_snapshots_path,
        parse_dates=(config.snapshot_date_column, config.last_fight_date_column),
        required_columns=required,
    )


def load_fighter_profiles(config: DataConfig) -> pd.DataFrame:
    """Load cleaned fighter identity and static profile data."""

    return load_csv(
        config.fighter_profiles_path,
        parse_dates=(config.dob_column,),
        required_columns=(
            config.fighter_id_column,
            config.fighter_name_column,
            config.dob_column,
        ),
    )


def load_feature_dictionary(config: DataConfig) -> pd.DataFrame:
    """Load the feature role and availability dictionary."""

    return load_csv(
        config.feature_dictionary_path,
        required_columns=(
            "column",
            "role",
            "available_pre_fight",
            "description",
        ),
    )


def load_dataset_bundle(config: DataConfig) -> DatasetBundle:
    """Load and fingerprint every configured processed data asset."""

    model = load_csv_asset(
        config.model_dataset_path,
        parse_dates=(config.date_column,),
        required_columns=(
            config.fight_id_column,
            config.date_column,
            config.target_column,
            "fighter_a_id",
            "fighter_b_id",
        ),
    )
    snapshots = load_csv_asset(
        config.fighter_snapshots_path,
        parse_dates=(config.snapshot_date_column, config.last_fight_date_column),
        required_columns=(
            config.fighter_id_column,
            config.fighter_name_column,
            config.snapshot_date_column,
            config.last_fight_date_column,
        ),
    )
    profiles = load_csv_asset(
        config.fighter_profiles_path,
        parse_dates=(config.dob_column,),
        required_columns=(
            config.fighter_id_column,
            config.fighter_name_column,
            config.dob_column,
        ),
    )
    dictionary = load_csv_asset(
        config.feature_dictionary_path,
        required_columns=(
            "column",
            "role",
            "available_pre_fight",
            "description",
        ),
    )
    return DatasetBundle(
        model=model,
        snapshots=snapshots,
        profiles=profiles,
        feature_dictionary=dictionary,
    )


__all__ = [
    "DatasetBundle",
    "LoadedFrame",
    "dataframe_sha256",
    "file_sha256",
    "load_csv",
    "load_csv_asset",
    "load_dataset_bundle",
    "load_feature_dictionary",
    "load_fighter_profiles",
    "load_fighter_snapshots",
    "load_model_dataset",
    "verify_file_sha256",
]
