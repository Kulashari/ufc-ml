"""Fighter identity resolution and point-in-time snapshot selection.

Names are presentation data, not identifiers.  This module deliberately keeps
identity resolution separate from feature construction so every downstream
prediction can be tied to the stable identifier extracted from UFCStats.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from difflib import get_close_matches
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pandas as pd

from ..exceptions import UFCPredictorError

_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")
_UFCSTATS_ID = re.compile(r"^[0-9a-f]{16}$", re.IGNORECASE)


def normalize_name(value: str) -> str:
    """Return a deterministic, accent-insensitive fighter-name key."""

    if not isinstance(value, str):
        raise TypeError("fighter name must be a string")
    decomposed = unicodedata.normalize("NFKD", value)
    ascii_value = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return _NON_ALPHANUMERIC.sub(" ", ascii_value.casefold()).strip()


def normalize_fighter_id(value: str) -> str:
    """Normalize an explicit stable ID or a UFCStats fighter URL.

    UFCStats identifiers are lower-cased.  Non-UFCStats identifiers are
    retained verbatim (after whitespace and trailing-slash removal), allowing
    this lookup to support a future internal identity system as well.
    """

    if not isinstance(value, str):
        raise TypeError("fighter_id must be a string")
    cleaned = value.strip().rstrip("/")
    if not cleaned:
        raise ValueError("fighter_id cannot be empty")
    final_segment = cleaned.rsplit("/", maxsplit=1)[-1]
    return final_segment.lower() if _UFCSTATS_ID.fullmatch(final_segment) else final_segment


def coerce_date(value: date | datetime | str, *, field_name: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"{field_name} cannot be empty")
        try:
            return date.fromisoformat(cleaned[:10])
        except ValueError as exc:
            raise ValueError(
                f"{field_name} must be an ISO date (YYYY-MM-DD), got {value!r}"
            ) from exc
    raise TypeError(f"{field_name} must be a date, datetime, or ISO date string")


class LookupStatus(StrEnum):
    """Outcome of a non-raising lookup."""

    NOT_FOUND = "not_found"
    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class FighterCandidate:
    """Compact identity record returned to callers and user interfaces."""

    fighter_id: str
    fighter_name: str
    display_name: str
    division: str | None = None
    dob: date | None = None
    as_of_date: date | None = None
    aliases: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "fighter_id": self.fighter_id,
            "fighter_name": self.fighter_name,
            "display_name": self.display_name,
            "division": self.division,
            "dob": self.dob.isoformat() if self.dob else None,
            "as_of_date": self.as_of_date.isoformat() if self.as_of_date else None,
            "aliases": list(self.aliases),
        }


@dataclass(frozen=True)
class FighterLookupResult:
    """Non-raising lookup result, including ambiguity candidates."""

    query: str
    normalized_query: str
    status: LookupStatus
    candidates: tuple[FighterCandidate, ...] = ()
    suggestions: tuple[FighterCandidate, ...] = ()

    @property
    def resolved(self) -> FighterCandidate | None:
        return self.candidates[0] if self.status is LookupStatus.RESOLVED else None


@dataclass(frozen=True)
class FighterSnapshot:
    """A fighter's feature state as known at a particular point in time."""

    fighter_id: str
    as_of_date: date
    values: Mapping[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    @property
    def fighter_name(self) -> str:
        return str(self.values.get("fighter_name", ""))

    @property
    def display_name(self) -> str:
        return str(self.values.get("display_name") or self.fighter_name)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.values)


class FighterLookupError(UFCPredictorError, ValueError):
    """Base class for fighter lookup failures."""


class FighterNotFoundError(FighterLookupError):
    def __init__(self, query: str, suggestions: Sequence[FighterCandidate] = ()) -> None:
        self.query = query
        self.suggestions = tuple(suggestions)
        suffix = ""
        if self.suggestions:
            labels = ", ".join(candidate.display_name for candidate in self.suggestions)
            suffix = f". Close matches: {labels}"
        super().__init__(f"No fighter matched {query!r}{suffix}")


class AmbiguousFighterError(FighterLookupError):
    def __init__(self, query: str, candidates: Sequence[FighterCandidate]) -> None:
        self.query = query
        self.candidates = tuple(candidates)
        labels = ", ".join(
            f"{candidate.display_name} [{candidate.fighter_id}]" for candidate in self.candidates
        )
        super().__init__(
            f"Fighter name {query!r} is ambiguous. Select a stable fighter_id: {labels}"
        )


class FighterIdentityMismatchError(FighterLookupError):
    """Raised when an explicit ID conflicts with the accompanying name."""


class SnapshotUnavailableError(FighterLookupError):
    def __init__(
        self,
        fighter_id: str,
        reference_date: date,
        available_dates: Sequence[date] = (),
    ) -> None:
        self.fighter_id = fighter_id
        self.reference_date = reference_date
        self.available_dates = tuple(sorted(available_dates))
        if self.available_dates:
            first = self.available_dates[0].isoformat()
            last = self.available_dates[-1].isoformat()
            detail = f" Available snapshot range: {first} through {last}."
        else:
            detail = " No dated snapshots are available."
        super().__init__(
            "No snapshot exists strictly before "
            f"{reference_date.isoformat()} for fighter {fighter_id}.{detail}"
        )


class DuplicateSnapshotError(FighterLookupError):
    """Raised when a fighter has more than one row for the selected as-of date."""


class FighterLookup:
    """Resolve names and aliases to stable IDs and select leakage-safe snapshots."""

    def __init__(
        self,
        snapshots: pd.DataFrame | Iterable[Mapping[str, Any]] | str | Path,
        *,
        aliases: Mapping[str, str | Sequence[str]] | None = None,
        id_column: str = "fighter_id",
        name_column: str = "fighter_name",
        display_name_column: str = "display_name",
        as_of_column: str = "as_of_date",
    ) -> None:
        records: list[dict[str, Any]]
        if isinstance(snapshots, (str, Path)):
            frame = pd.read_csv(snapshots)
            records = [
                {str(key): value for key, value in record.items()}
                for record in frame.to_dict(orient="records")
            ]
        elif isinstance(snapshots, pd.DataFrame):
            records = [
                {str(key): value for key, value in record.items()}
                for record in snapshots.to_dict(orient="records")
            ]
        else:
            records = [dict(record) for record in snapshots]

        if not records:
            raise ValueError("fighter snapshot data cannot be empty")

        self.id_column = id_column
        self.name_column = name_column
        self.display_name_column = display_name_column
        self.as_of_column = as_of_column

        self._records_by_id: dict[str, list[dict[str, Any]]] = {}
        self._search_index: dict[str, set[str]] = {}
        self._explicit_aliases_by_id: dict[str, set[str]] = {}

        for position, raw_record in enumerate(records):
            if id_column not in raw_record:
                raise ValueError(f"snapshot row {position} is missing {id_column!r}")
            fighter_id = normalize_fighter_id(str(raw_record[id_column]))
            fighter_name = str(raw_record.get(name_column, "")).strip()
            if not fighter_name:
                raise ValueError(f"snapshot row {position} has no fighter name")

            record = dict(raw_record)
            record[id_column] = fighter_id
            self._records_by_id.setdefault(fighter_id, []).append(record)

            display_name = str(record.get(display_name_column) or fighter_name).strip()
            for label in (fighter_name, display_name):
                key = normalize_name(label)
                if key:
                    self._search_index.setdefault(key, set()).add(fighter_id)

        if aliases:
            self.add_aliases(aliases)

    @property
    def fighter_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._records_by_id))

    def add_aliases(self, aliases: Mapping[str, str | Sequence[str]]) -> None:
        """Add aliases mapped to one or more stable fighter IDs."""

        for alias, targets in aliases.items():
            alias_key = normalize_name(alias)
            if not alias_key:
                raise ValueError("fighter alias cannot normalize to an empty value")
            target_values = (targets,) if isinstance(targets, str) else tuple(targets)
            if not target_values:
                raise ValueError(f"alias {alias!r} has no target fighter IDs")
            for target in target_values:
                fighter_id = normalize_fighter_id(target)
                if fighter_id not in self._records_by_id:
                    raise ValueError(
                        f"alias {alias!r} references unknown fighter_id {fighter_id!r}"
                    )
                self._search_index.setdefault(alias_key, set()).add(fighter_id)
                self._explicit_aliases_by_id.setdefault(fighter_id, set()).add(alias)

    def candidate_for_id(self, fighter_id: str) -> FighterCandidate:
        canonical_id = normalize_fighter_id(fighter_id)
        try:
            records = self._records_by_id[canonical_id]
        except KeyError as exc:
            raise FighterNotFoundError(fighter_id) from exc
        record = self._latest_record(records)
        return self._candidate_from_record(record)

    def lookup(self, query: str, *, suggestion_limit: int = 5) -> FighterLookupResult:
        """Resolve without raising, returning all candidates when ambiguous."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("fighter query must be a non-empty string")

        stripped_query = query.strip()
        possible_id = normalize_fighter_id(stripped_query)
        if possible_id in self._records_by_id:
            candidate = self.candidate_for_id(possible_id)
            return FighterLookupResult(
                query=query,
                normalized_query=possible_id,
                status=LookupStatus.RESOLVED,
                candidates=(candidate,),
            )

        key = normalize_name(stripped_query)
        fighter_ids = sorted(self._search_index.get(key, ()))
        candidates = tuple(self.candidate_for_id(value) for value in fighter_ids)
        if len(candidates) == 1:
            status = LookupStatus.RESOLVED
        elif len(candidates) > 1:
            status = LookupStatus.AMBIGUOUS
        else:
            status = LookupStatus.NOT_FOUND

        suggestions: tuple[FighterCandidate, ...] = ()
        if status is LookupStatus.NOT_FOUND and suggestion_limit > 0:
            close_keys = get_close_matches(
                key,
                self._search_index.keys(),
                n=max(suggestion_limit * 2, suggestion_limit),
                cutoff=0.65,
            )
            suggested_ids: list[str] = []
            for close_key in close_keys:
                for fighter_id in sorted(self._search_index[close_key]):
                    if fighter_id not in suggested_ids:
                        suggested_ids.append(fighter_id)
                    if len(suggested_ids) >= suggestion_limit:
                        break
                if len(suggested_ids) >= suggestion_limit:
                    break
            suggestions = tuple(self.candidate_for_id(fighter_id) for fighter_id in suggested_ids)

        return FighterLookupResult(
            query=query,
            normalized_query=key,
            status=status,
            candidates=candidates,
            suggestions=suggestions,
        )

    def resolve(self, query: str, *, fighter_id: str | None = None) -> FighterCandidate:
        """Resolve a fighter, requiring an explicit stable ID when ambiguous."""

        result = self.lookup(query)
        if fighter_id is not None:
            candidate = self.candidate_for_id(fighter_id)
            if result.candidates and candidate.fighter_id not in {
                value.fighter_id for value in result.candidates
            }:
                raise FighterIdentityMismatchError(
                    f"{query!r} does not refer to fighter_id {candidate.fighter_id!r}"
                )
            return candidate

        if result.status is LookupStatus.NOT_FOUND:
            raise FighterNotFoundError(query, result.suggestions)
        if result.status is LookupStatus.AMBIGUOUS:
            raise AmbiguousFighterError(query, result.candidates)
        assert result.resolved is not None
        return result.resolved

    def select_snapshot(
        self,
        fighter_id: str,
        reference_date: date,
    ) -> FighterSnapshot:
        """Select the latest snapshot with ``as_of_date < reference_date``.

        Strict inequality is intentional: a snapshot stamped with the prediction
        day may already contain data that was not available at prediction time.
        """

        canonical_id = normalize_fighter_id(fighter_id)
        if canonical_id not in self._records_by_id:
            raise FighterNotFoundError(fighter_id)

        dated_records: list[tuple[date, dict[str, Any]]] = []
        available_dates: list[date] = []
        for record in self._records_by_id[canonical_id]:
            parsed = self._record_date(record)
            if parsed is None:
                continue
            available_dates.append(parsed)
            if parsed < reference_date:
                dated_records.append((parsed, record))

        if not dated_records:
            raise SnapshotUnavailableError(
                canonical_id, reference_date, available_dates=available_dates
            )

        latest_date = max(value[0] for value in dated_records)
        selected = [record for as_of, record in dated_records if as_of == latest_date]
        if len(selected) != 1:
            raise DuplicateSnapshotError(
                f"fighter {canonical_id} has {len(selected)} snapshots on {latest_date.isoformat()}"
            )

        values = dict(selected[0])
        values[self.id_column] = canonical_id
        values[self.as_of_column] = latest_date.isoformat()
        return FighterSnapshot(
            fighter_id=canonical_id,
            as_of_date=latest_date,
            values=MappingProxyType(values),
        )

    def _candidate_from_record(self, record: Mapping[str, Any]) -> FighterCandidate:
        fighter_id = normalize_fighter_id(str(record[self.id_column]))
        fighter_name = str(record[self.name_column]).strip()
        display_name = str(record.get(self.display_name_column) or fighter_name).strip()
        division_value = record.get("last_division") or record.get("division")
        division = (
            str(division_value).strip()
            if division_value is not None and str(division_value).strip()
            else None
        )
        dob = self._optional_date(record.get("dob"))
        as_of_date = self._record_date(record)
        aliases = tuple(sorted(self._explicit_aliases_by_id.get(fighter_id, ())))
        return FighterCandidate(
            fighter_id=fighter_id,
            fighter_name=fighter_name,
            display_name=display_name,
            division=division,
            dob=dob,
            as_of_date=as_of_date,
            aliases=aliases,
        )

    def _record_date(self, record: Mapping[str, Any]) -> date | None:
        return self._optional_date(record.get(self.as_of_column))

    @staticmethod
    def _optional_date(value: Any) -> date | None:
        if value is None or pd.isna(value) or str(value).strip() == "":
            return None
        return coerce_date(value, field_name="snapshot date")

    def _latest_record(self, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return max(
            records,
            key=lambda record: self._record_date(record) or date.min,
        )


__all__ = [
    "AmbiguousFighterError",
    "DuplicateSnapshotError",
    "FighterCandidate",
    "FighterIdentityMismatchError",
    "FighterLookup",
    "FighterLookupError",
    "FighterLookupResult",
    "FighterNotFoundError",
    "FighterSnapshot",
    "LookupStatus",
    "SnapshotUnavailableError",
    "coerce_date",
    "normalize_fighter_id",
    "normalize_name",
]
