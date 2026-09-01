"""Legacy CSV and normalized-SQLite adapters for feature construction.

No parser from the live-data fetcher is imported here.  The core feature
builder consumes its published, ID-rich SQLite schema through a narrow read
adapter, which keeps acquisition and modeling independent.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping
from datetime import date
from pathlib import Path
from typing import Any, cast

import pandas as pd

from ufc_ml_core.data.loader import load_csv
from ufc_ml_core.exceptions import DataLoadError, DataValidationError
from ufc_ml_core.feature_building.domain import (
    Bout,
    FighterFightStats,
    FighterProfile,
    Result,
    normalize_division,
    raw_bout_flags,
    scheduled_round_context,
)


def _is_missing(value: object) -> bool:
    """Return whether one scalar source value is absent.

    Pandas accepts a wider collection of scalar values than its type stubs can
    express as ``object``.  Source adapters only pass scalar cells here, and
    the explicit ``bool`` keeps a pandas scalar from leaking into control flow.
    """

    return bool(pd.isna(cast(Any, value)))


def _id_from_url(value: object) -> str:
    raw = str(value or "").strip().rstrip("/")
    identifier = raw.rsplit("/", maxsplit=1)[-1]
    if len(identifier) != 16 or any(
        character not in "0123456789abcdef" for character in identifier
    ):
        raise DataValidationError(f"Could not derive a UFCStats ID from URL {value!r}")
    return identifier


def _optional_float(value: object) -> float | None:
    if value is None or _is_missing(value):
        return None
    text = str(value).strip().replace('"', "")
    if not text or text.casefold() in {"nan", "none", "n/a", "--"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _height_inches(value: object) -> float | None:
    if value is None or _is_missing(value):
        return None
    text = str(value).strip().replace('"', "")
    if not text or text.casefold() in {"nan", "none", "n/a", "--"}:
        return None
    if "'" not in text:
        return _optional_float(text)
    feet, inches = text.split("'", maxsplit=1)
    try:
        return float(int(feet.strip()) * 12 + int(inches.strip() or "0"))
    except ValueError:
        return None


def _optional_date(value: object) -> date | None:
    if value is None or _is_missing(value) or not str(value).strip():
        return None
    parsed = pd.to_datetime(str(value), errors="coerce")
    if not isinstance(parsed, pd.Timestamp) or _is_missing(parsed):
        return None
    return parsed.date()


def _clean_stance(value: object) -> str | None:
    if value is None or _is_missing(value):
        return None
    text = str(value).strip()
    return None if not text or text.casefold() in {"nan", "unknown", "none", "n/a"} else text


def _as_int(value: object, *, field: str, default: int | None = None) -> int:
    if value is None or _is_missing(value) or str(value).strip() == "":
        if default is not None:
            return default
        raise DataValidationError(f"{field} is missing")
    try:
        return int(float(cast(Any, value)))
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{field} must be an integer, got {value!r}") from exc


def _stats_from_legacy(row: Mapping[Hashable, object], corner: int) -> FighterFightStats:
    prefix = f"F{corner}_"
    return FighterFightStats(
        kd=_as_int(row[f"{prefix}KD"], field=f"{prefix}KD"),
        sig_landed=_as_int(row[f"{prefix}Sig_Landed"], field=f"{prefix}Sig_Landed"),
        sig_attempted=_as_int(row[f"{prefix}Sig_Att"], field=f"{prefix}Sig_Att"),
        td_landed=_as_int(row[f"{prefix}TD_Landed"], field=f"{prefix}TD_Landed"),
        td_attempted=_as_int(row[f"{prefix}TD_Att"], field=f"{prefix}TD_Att"),
        sub_attempts=_as_int(row[f"{prefix}Sub_Att"], field=f"{prefix}Sub_Att"),
        control_seconds=_as_int(row[f"{prefix}Ctrl_Sec"], field=f"{prefix}Ctrl_Sec"),
        head_landed=_as_int(row[f"{prefix}Head"], field=f"{prefix}Head"),
        body_landed=_as_int(row[f"{prefix}Body"], field=f"{prefix}Body"),
        leg_landed=_as_int(row[f"{prefix}Leg"], field=f"{prefix}Leg"),
        distance_landed=_as_int(row[f"{prefix}Distance"], field=f"{prefix}Distance"),
        clinch_landed=_as_int(row[f"{prefix}Clinch"], field=f"{prefix}Clinch"),
        ground_landed=_as_int(row[f"{prefix}Ground"], field=f"{prefix}Ground"),
    )


def _stats_from_sqlite(row: sqlite3.Row) -> FighterFightStats:
    return FighterFightStats(
        kd=_as_int(row["kd"], field="bout_fighter_totals.kd"),
        sig_landed=_as_int(row["sig_str_landed"], field="bout_fighter_totals.sig_str_landed"),
        sig_attempted=_as_int(
            row["sig_str_attempted"], field="bout_fighter_totals.sig_str_attempted"
        ),
        td_landed=_as_int(row["td_landed"], field="bout_fighter_totals.td_landed"),
        td_attempted=_as_int(row["td_attempted"], field="bout_fighter_totals.td_attempted"),
        sub_attempts=_as_int(row["sub_attempts"], field="bout_fighter_totals.sub_attempts"),
        control_seconds=_as_int(
            row["control_seconds"], field="bout_fighter_totals.control_seconds"
        ),
        head_landed=_as_int(row["head_landed"], field="bout_fighter_totals.head_landed"),
        body_landed=_as_int(row["body_landed"], field="bout_fighter_totals.body_landed"),
        leg_landed=_as_int(row["leg_landed"], field="bout_fighter_totals.leg_landed"),
        distance_landed=_as_int(
            row["distance_landed"], field="bout_fighter_totals.distance_landed"
        ),
        clinch_landed=_as_int(row["clinch_landed"], field="bout_fighter_totals.clinch_landed"),
        ground_landed=_as_int(row["ground_landed"], field="bout_fighter_totals.ground_landed"),
    )


class LegacyIdentityResolver:
    """Resolve legacy name-only corners to stable UFCStats fighter IDs.

    Most names map directly to one profile URL.  The legacy corpus has seven
    duplicate names; label rows supply an ID bridge in the checked-in model
    data, and any remaining ambiguity must be provided as a compact, auditable
    fight override rather than guessed.
    """

    def __init__(
        self,
        profiles: Iterable[FighterProfile],
        *,
        reference_model_path: Path | None = None,
        overrides: Mapping[tuple[str, str], str] | None = None,
    ) -> None:
        self._by_name: dict[str, tuple[str, ...]] = {}
        grouped: dict[str, list[str]] = defaultdict(list)
        for profile in profiles:
            grouped[profile.fighter_name].append(profile.fighter_id)
        self._by_name = {name: tuple(sorted(set(ids))) for name, ids in grouped.items()}
        self._corner_ids: dict[tuple[str, str], str] = {}
        self._overrides = dict(overrides or {})
        if reference_model_path is not None and reference_model_path.is_file():
            self._load_reference_model(reference_model_path)

    def _load_reference_model(self, path: Path) -> None:
        frame = load_csv(
            path,
            required_columns=(
                "fight_id",
                "fighter_a_id",
                "fighter_a_name",
                "fighter_b_id",
                "fighter_b_name",
            ),
        )
        for row in frame.to_dict(orient="records"):
            fight_id = str(row["fight_id"])
            is_even = int(fight_id[-1], 16) % 2 == 0
            if is_even:
                pairs = (
                    (str(row["fighter_a_name"]), str(row["fighter_a_id"])),
                    (str(row["fighter_b_name"]), str(row["fighter_b_id"])),
                )
            else:
                pairs = (
                    (str(row["fighter_b_name"]), str(row["fighter_b_id"])),
                    (str(row["fighter_a_name"]), str(row["fighter_a_id"])),
                )
            for name, fighter_id in pairs:
                self._corner_ids[(fight_id, name)] = fighter_id

    def resolve(self, fight_id: str, fighter_name: str) -> str:
        key = (fight_id, fighter_name)
        if key in self._corner_ids:
            return self._corner_ids[key]
        if key in self._overrides:
            return self._overrides[key]
        candidates = self._by_name.get(fighter_name, ())
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise DataValidationError(
                f"Legacy fighter {fighter_name!r} has no matching profile URL for {fight_id}"
            )
        raise DataValidationError(
            f"Legacy fighter {fighter_name!r} is ambiguous for {fight_id}; "
            f"candidate IDs={list(candidates)}. Add an identity override."
        )


def load_legacy_profiles(path: Path) -> list[FighterProfile]:
    """Read and normalize the legacy UFCStats fighter-profile CSV."""

    frame = load_csv(
        path,
        required_columns=("Fighter_Name", "Height", "Reach", "Stance", "DOB", "Fighter_URL"),
    )
    profiles: list[FighterProfile] = []
    seen: set[str] = set()
    for row in frame.to_dict(orient="records"):
        fighter_id = _id_from_url(row["Fighter_URL"])
        if fighter_id in seen:
            raise DataValidationError(f"Duplicate legacy fighter profile ID: {fighter_id}")
        seen.add(fighter_id)
        profiles.append(
            FighterProfile(
                fighter_id=fighter_id,
                fighter_name=str(row["Fighter_Name"]).strip(),
                height_inches=_height_inches(row["Height"]),
                reach_inches=_optional_float(row["Reach"]),
                stance=_clean_stance(row["Stance"]),
                dob=_optional_date(row["DOB"]),
                fighter_url=str(row["Fighter_URL"]).strip(),
            )
        )
    return profiles


def load_sqlite_profiles(path: Path) -> list[FighterProfile]:
    """Read ID-rich current profiles materialized by the latest-data fetcher."""

    if not path.is_file():
        return []
    try:
        with sqlite3.connect(path) as connection:
            rows = connection.execute(
                "SELECT fighter_id, fighter_name, height_inches, reach_inches, "
                "stance, dob, fighter_url FROM fighters"
            ).fetchall()
    except sqlite3.Error as exc:
        raise DataLoadError(f"Could not read normalized SQLite profiles {path}: {exc}") from exc
    return [
        FighterProfile(
            fighter_id=str(row[0]),
            fighter_name=str(row[1]),
            height_inches=_optional_float(row[2]),
            reach_inches=_optional_float(row[3]),
            stance=_clean_stance(row[4]),
            dob=_optional_date(row[5]),
            fighter_url=str(row[6]),
        )
        for row in rows
    ]


def load_legacy_bouts(path: Path, resolver: LegacyIdentityResolver) -> list[Bout]:
    """Load raw legacy fights in causal date/card order.

    The old scraped CSV is date-sorted, but UFCStats displays each card from
    main event down.  Reversing each same-date card block makes a tournament's
    earlier bouts update state before its final while retaining chronological
    event dates.
    """

    frame = load_csv(
        path,
        required_columns=(
            "Fight_URL",
            "Fighter_1",
            "Fighter_2",
            "Winner",
            "Weight_Class",
            "Method",
            "End_Round",
            "End_Time",
            "Total_Fight_Time_Sec",
            "Time_Format",
            "Event_Date",
        ),
    )
    bouts: list[Bout] = []
    previous_date: date | None = None
    for order, row in enumerate(frame.to_dict(orient="records")):
        fight_id = _id_from_url(row["Fight_URL"])
        event_date = _optional_date(row["Event_Date"])
        if event_date is None:
            raise DataValidationError(f"Legacy fight {fight_id} has no valid Event_Date")
        if previous_date is not None and event_date < previous_date:
            raise DataValidationError("Legacy raw fight CSV is not chronological by Event_Date")
        previous_date = event_date
        fighter_1_name = str(row["Fighter_1"]).strip()
        fighter_2_name = str(row["Fighter_2"]).strip()
        winner = str(row["Winner"]).strip()
        result_1: Result
        result_2: Result
        if winner == fighter_1_name:
            result_1, result_2 = "W", "L"
        elif winner == fighter_2_name:
            result_1, result_2 = "L", "W"
        elif str(row["Method"]).strip().casefold().startswith("decision"):
            # The legacy CSV collapses Draw and No Contest into ``Draw/NC``.
            # A recorded decision is the recoverable draw signal; overturned
            # and other non-decision rows remain no contests for Elo/streak.
            result_1 = result_2 = "D"
        else:
            result_1 = result_2 = "NC"
        raw_weight_class = str(row["Weight_Class"]).strip()
        scheduled_rounds, scheduled_duration_sec = scheduled_round_context(str(row["Time_Format"]))
        title, interim, tournament, superfight = raw_bout_flags(raw_weight_class)
        bouts.append(
            Bout(
                fight_id=fight_id,
                event_date=event_date,
                source_order=order,
                fighter_1_id=resolver.resolve(fight_id, fighter_1_name),
                fighter_1_name=fighter_1_name,
                fighter_2_id=resolver.resolve(fight_id, fighter_2_name),
                fighter_2_name=fighter_2_name,
                result_1=result_1,
                result_2=result_2,
                raw_weight_class=raw_weight_class,
                division=normalize_division(raw_weight_class),
                method=str(row["Method"]).strip(),
                end_round=_as_int(row["End_Round"], field="End_Round"),
                end_time=str(row["End_Time"]).strip(),
                total_fight_time_sec=_as_int(
                    row["Total_Fight_Time_Sec"], field="Total_Fight_Time_Sec"
                ),
                time_format=str(row["Time_Format"]).strip(),
                scheduled_rounds=scheduled_rounds,
                scheduled_duration_sec=scheduled_duration_sec,
                is_title_bout=title,
                is_interim_title=interim,
                is_tournament_final=tournament,
                is_superfight=superfight,
                stats_1=_stats_from_legacy(row, 1),
                stats_2=_stats_from_legacy(row, 2),
            )
        )
    ordered: list[Bout] = []
    cursor = 0
    while cursor < len(bouts):
        end = cursor + 1
        while end < len(bouts) and bouts[end].event_date == bouts[cursor].event_date:
            end += 1
        ordered.extend(reversed(bouts[cursor:end]))
        cursor = end
    # ``source_order`` is the deterministic secondary chronology used when
    # legacy and SQLite records are merged later.
    return [
        Bout(
            fight_id=bout.fight_id,
            event_date=bout.event_date,
            source_order=index,
            fighter_1_id=bout.fighter_1_id,
            fighter_1_name=bout.fighter_1_name,
            fighter_2_id=bout.fighter_2_id,
            fighter_2_name=bout.fighter_2_name,
            result_1=bout.result_1,
            result_2=bout.result_2,
            raw_weight_class=bout.raw_weight_class,
            division=bout.division,
            method=bout.method,
            end_round=bout.end_round,
            end_time=bout.end_time,
            total_fight_time_sec=bout.total_fight_time_sec,
            time_format=bout.time_format,
            scheduled_rounds=bout.scheduled_rounds,
            scheduled_duration_sec=bout.scheduled_duration_sec,
            is_title_bout=bout.is_title_bout,
            is_interim_title=bout.is_interim_title,
            is_tournament_final=bout.is_tournament_final,
            is_superfight=bout.is_superfight,
            stats_1=bout.stats_1,
            stats_2=bout.stats_2,
        )
        for index, bout in enumerate(ordered)
    ]


def load_normalized_bouts(path: Path, *, starting_order: int = 0) -> list[Bout]:
    """Load normalized fights, reversing UFCStats display order within events.

    UFCStats lists the main event as ``bout_order=1``.  Reversing it preserves
    the causal event order used by the legacy historical CSV.
    """

    if not path.is_file():
        return []
    try:
        with sqlite3.connect(path) as connection:
            connection.row_factory = sqlite3.Row
            fights = connection.execute(
                "SELECT * FROM fights "
                "ORDER BY event_date, event_id, CAST(bout_order AS INTEGER) DESC"
            ).fetchall()
            totals = connection.execute("SELECT * FROM bout_fighter_totals").fetchall()
    except sqlite3.Error as exc:
        raise DataLoadError(f"Could not read normalized SQLite fights {path}: {exc}") from exc
    totals_by_fight: dict[str, dict[str, sqlite3.Row]] = defaultdict(dict)
    for total in totals:
        totals_by_fight[str(total["fight_id"])][str(total["fighter_id"])] = total

    bouts: list[Bout] = []
    for index, fight in enumerate(fights):
        fight_id = str(fight["fight_id"])
        first_id = str(fight["fighter_1_id"])
        second_id = str(fight["fighter_2_id"])
        first_total = totals_by_fight.get(fight_id, {}).get(first_id)
        second_total = totals_by_fight.get(fight_id, {}).get(second_id)
        if first_total is None or second_total is None:
            raise DataValidationError(
                f"Normalized fight {fight_id} is missing one corner's total stats"
            )
        raw_weight_class = str(fight["raw_weight_class"])
        scheduled_rounds = _as_int(fight["scheduled_rounds"], field="scheduled_rounds", default=0)
        scheduled_duration_sec = _as_int(
            fight["scheduled_duration_sec"], field="scheduled_duration_sec", default=0
        )
        bouts.append(
            Bout(
                fight_id=fight_id,
                event_date=_optional_date(fight["event_date"]) or date.min,
                source_order=starting_order + index,
                fighter_1_id=first_id,
                fighter_1_name=str(fight["fighter_1_name"]),
                fighter_2_id=second_id,
                fighter_2_name=str(fight["fighter_2_name"]),
                result_1=cast(Result, str(fight["fighter_1_status"]).strip().upper()),
                result_2=cast(Result, str(fight["fighter_2_status"]).strip().upper()),
                raw_weight_class=raw_weight_class,
                division=normalize_division(raw_weight_class),
                method=str(fight["method"]),
                end_round=_as_int(fight["end_round"], field="end_round"),
                end_time=str(fight["end_time"]),
                total_fight_time_sec=_as_int(
                    fight["total_fight_time_sec"], field="total_fight_time_sec"
                ),
                time_format=str(fight["time_format"]),
                scheduled_rounds=scheduled_rounds,
                scheduled_duration_sec=scheduled_duration_sec,
                is_title_bout=_as_int(fight["is_title_bout"], field="is_title_bout"),
                is_interim_title=_as_int(fight["is_interim_title"], field="is_interim_title"),
                is_tournament_final=_as_int(
                    fight["is_tournament_final"], field="is_tournament_final"
                ),
                is_superfight=_as_int(fight["is_superfight"], field="is_superfight"),
                stats_1=_stats_from_sqlite(first_total),
                stats_2=_stats_from_sqlite(second_total),
            )
        )
    return bouts


def merge_chronological_bouts(legacy: Iterable[Bout], normalized: Iterable[Bout]) -> list[Bout]:
    """Merge source adapters, rejecting conflicting duplicate fight IDs."""

    seen: dict[str, Bout] = {}
    for bout in (*legacy, *normalized):
        existing = seen.get(bout.fight_id)
        if existing is not None:
            if existing.event_date != bout.event_date:
                raise DataValidationError(
                    f"Fight {bout.fight_id} conflicts across legacy and normalized sources"
                )
            continue
        seen[bout.fight_id] = bout
    return sorted(seen.values(), key=lambda bout: (bout.event_date, bout.source_order))


__all__ = [
    "LegacyIdentityResolver",
    "load_legacy_bouts",
    "load_legacy_profiles",
    "load_normalized_bouts",
    "load_sqlite_profiles",
    "merge_chronological_bouts",
]
