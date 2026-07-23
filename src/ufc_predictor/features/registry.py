"""Canonical feature registry backed by the processed feature dictionary."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator, Sequence
from typing import Self, overload

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator

from ufc_predictor.exceptions import FeatureRegistryError
from ufc_predictor.features.groups import FeatureGroup, classify_feature, group_features

DICTIONARY_REQUIRED_COLUMNS = frozenset({"column", "role", "available_pre_fight", "description"})
MODEL_FEATURE_ROLE = "model_feature"


class FeatureSpec(BaseModel):
    """One ordered feature definition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    column: str
    description: str = ""
    group: FeatureGroup
    available_pre_fight: bool = True

    @field_validator("column")
    @classmethod
    def column_is_not_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("feature column cannot be empty")
        return value


class FeatureRegistry(Sequence[FeatureSpec]):
    """Immutable, ordered feature contract.

    Dictionary row order is the canonical model-matrix order. Fallback
    discovery from a dataframe sorts names lexicographically so it remains
    deterministic even when upstream column order changes.
    """

    def __init__(self, specs: Iterable[FeatureSpec]) -> None:
        self._specs = tuple(specs)
        if not self._specs:
            raise FeatureRegistryError("Feature registry cannot be empty")
        names = [spec.column for spec in self._specs]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise FeatureRegistryError(f"Feature registry contains duplicate columns: {duplicates}")
        unavailable = [spec.column for spec in self._specs if not spec.available_pre_fight]
        if unavailable:
            raise FeatureRegistryError(
                f"Model features must be available pre-fight; invalid columns: {unavailable}"
            )

    @classmethod
    def from_dictionary(cls, dictionary: pd.DataFrame) -> Self:
        """Build a registry from ``ufc_model_feature_dictionary.csv`` data."""

        missing = sorted(DICTIONARY_REQUIRED_COLUMNS.difference(dictionary.columns))
        if missing:
            raise FeatureRegistryError(f"Feature dictionary is missing required columns: {missing}")
        if dictionary["column"].isna().any():
            raise FeatureRegistryError("Feature dictionary contains a null column name")
        duplicate_mask = dictionary["column"].astype(str).duplicated(keep=False)
        if duplicate_mask.any():
            duplicates = sorted(
                dictionary.loc[duplicate_mask, "column"].astype(str).unique().tolist()
            )
            raise FeatureRegistryError(
                f"Feature dictionary contains duplicate columns: {duplicates}"
            )

        model_rows = dictionary.loc[dictionary["role"].astype(str).eq(MODEL_FEATURE_ROLE)]
        if model_rows.empty:
            raise FeatureRegistryError("Feature dictionary contains no model_feature rows")

        specs: list[FeatureSpec] = []
        for row in model_rows.itertuples(index=False):
            name = str(row.column).strip()
            availability = _coerce_dictionary_bool(row.available_pre_fight, name)
            specs.append(
                FeatureSpec(
                    column=name,
                    description="" if pd.isna(row.description) else str(row.description),
                    group=classify_feature(name),
                    available_pre_fight=availability,
                )
            )
        return cls(specs)

    @classmethod
    def discover(
        cls,
        frame: pd.DataFrame,
        *,
        prefix: str = "feature_",
        exclude: Iterable[str] = (),
    ) -> Self:
        """Discover numeric-prefixed features in lexicographic order."""

        excluded = frozenset(exclude)
        names = sorted(
            column
            for column in frame.columns
            if column.startswith(prefix) and column not in excluded
        )
        if not names:
            raise FeatureRegistryError(f"No feature columns found with prefix {prefix!r}")
        return cls(
            FeatureSpec(
                column=name,
                description="Discovered model feature",
                group=classify_feature(name),
            )
            for name in names
        )

    @overload
    def __getitem__(self, index: int) -> FeatureSpec: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[FeatureSpec, ...]: ...

    def __getitem__(self, index: int | slice) -> FeatureSpec | tuple[FeatureSpec, ...]:
        return self._specs[index]

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self) -> Iterator[FeatureSpec]:
        return iter(self._specs)

    @property
    def names(self) -> tuple[str, ...]:
        """Canonical model-matrix column order."""

        return tuple(spec.column for spec in self._specs)

    @property
    def groups(self) -> dict[FeatureGroup, tuple[str, ...]]:
        """Features grouped in canonical order."""

        return group_features(self.names)

    @property
    def fingerprint(self) -> str:
        """SHA256 of the ordered feature contract."""

        digest = hashlib.sha256()
        for spec in self._specs:
            digest.update(spec.column.encode("utf-8"))
            digest.update(b"\x1f")
            digest.update(spec.group.value.encode("utf-8"))
            digest.update(b"\x1f")
            digest.update(str(int(spec.available_pre_fight)).encode("ascii"))
            digest.update(b"\x1e")
        return digest.hexdigest()

    def for_group(self, group: FeatureGroup | str) -> tuple[str, ...]:
        """Return canonical columns in one semantic group."""

        try:
            parsed_group = FeatureGroup(group)
        except ValueError as exc:
            raise FeatureRegistryError(f"Unknown feature group: {group!r}") from exc
        return self.groups.get(parsed_group, ())

    def assert_matches(self, frame: pd.DataFrame, *, prefix: str = "feature_") -> None:
        """Require exact agreement with prefixed features in ``frame``."""

        expected = set(self.names)
        actual = {column for column in frame.columns if column.startswith(prefix)}
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing or unexpected:
            parts: list[str] = []
            if missing:
                parts.append(f"missing={missing}")
            if unexpected:
                parts.append(f"unexpected={unexpected}")
            raise FeatureRegistryError(
                "Dataset does not match feature registry: " + "; ".join(parts)
            )

    def select(self, frame: pd.DataFrame, *, copy: bool = True) -> pd.DataFrame:
        """Select the model matrix in canonical order."""

        missing = [name for name in self.names if name not in frame.columns]
        if missing:
            raise FeatureRegistryError(f"Cannot select model matrix; missing features: {missing}")
        selected = frame.loc[:, list(self.names)]
        return selected.copy() if copy else selected


def discover_feature_columns(
    frame: pd.DataFrame,
    *,
    prefix: str = "feature_",
    exclude: Iterable[str] = (),
) -> tuple[str, ...]:
    """Convenience wrapper returning deterministic feature names."""

    return FeatureRegistry.discover(frame, prefix=prefix, exclude=exclude).names


def _coerce_dictionary_bool(value: object, column: str) -> bool:
    """Parse the compact boolean encodings accepted in the dictionary."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int | float) and not pd.isna(value) and value in (0, 1):
        return bool(value)
    normalized = str(value).strip().casefold()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise FeatureRegistryError(f"Invalid available_pre_fight value for {column!r}: {value!r}")


__all__ = [
    "DICTIONARY_REQUIRED_COLUMNS",
    "MODEL_FEATURE_ROLE",
    "FeatureRegistry",
    "FeatureSpec",
    "discover_feature_columns",
]
