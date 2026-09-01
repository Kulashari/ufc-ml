"""Domain records and pure normalization used by the raw-to-feature builder.

The model data set intentionally has no post-fight columns.  This module keeps
the raw observations separate from the state calculated *before* each bout so
that the chronological builder has a small, auditable surface area.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

Result = Literal["W", "L", "D", "NC"]


@dataclass(frozen=True, slots=True)
class Division:
    """Normalized bout division used for feature context and division Elo."""

    code: str
    pounds: float
    is_womens: bool
    is_catch_weight: bool = False


UNKNOWN_DIVISION = Division("UNKNOWN", 0.0, False)
OPEN_DIVISION = Division("OPEN", 0.0, False)
CATCH_DIVISION = Division("CATCH", 0.0, False, is_catch_weight=True)

_DIVISIONS: tuple[tuple[str, Division], ...] = (
    ("women's strawweight", Division("W_STRAW", 115.0, True)),
    ("women's flyweight", Division("W_FLY", 125.0, True)),
    ("women's bantamweight", Division("W_BANTAM", 135.0, True)),
    ("women's featherweight", Division("W_FEATHER", 145.0, True)),
    ("light heavyweight", Division("M_LIGHT_HEAVY", 205.0, False)),
    ("heavyweight", Division("M_HEAVY", 265.0, False)),
    ("middleweight", Division("M_MIDDLE", 185.0, False)),
    ("welterweight", Division("M_WELTER", 170.0, False)),
    ("lightweight", Division("M_LIGHT", 155.0, False)),
    ("featherweight", Division("M_FEATHER", 145.0, False)),
    ("bantamweight", Division("M_BANTAM", 135.0, False)),
    ("flyweight", Division("M_FLY", 125.0, False)),
)

_DIVISION_BY_CODE = {division.code: division for _, division in _DIVISIONS} | {
    UNKNOWN_DIVISION.code: UNKNOWN_DIVISION,
    OPEN_DIVISION.code: OPEN_DIVISION,
    CATCH_DIVISION.code: CATCH_DIVISION,
}


def normalize_division(raw_weight_class: str) -> Division:
    """Map UFCStats' raw label to one stable division context.

    UFCStats labels title and tournament finals with their actual division, but
    catch/open-weight bouts intentionally retain separate contexts because the
    checked-in model did the same.
    """

    normalized = " ".join(raw_weight_class.casefold().split())
    if "catch weight" in normalized or "catchweight" in normalized:
        return CATCH_DIVISION
    if "open weight" in normalized or "super heavy" in normalized:
        return OPEN_DIVISION
    for needle, division in _DIVISIONS:
        if needle in normalized:
            return division
    return UNKNOWN_DIVISION


def division_from_code(value: str) -> Division:
    """Return a stored division code's feature context or ``UNKNOWN``."""

    return _DIVISION_BY_CODE.get(value.strip().upper(), UNKNOWN_DIVISION)


def scheduled_round_context(time_format: str) -> tuple[int, int]:
    """Return scheduled rounds/duration for known UFCStats time formats.

    The historic model rows only include standard three- and five-round bouts,
    but retaining sensible metadata for history-only formats makes raw audits
    easier and avoids treating them as model labels.
    """

    normalized = " ".join(time_format.split())
    if normalized == "3 Rnd (5-5-5)":
        return (3, 900)
    if normalized == "5 Rnd (5-5-5-5-5)":
        return (5, 1500)
    if normalized.startswith("2 Rnd (5-5)"):
        return (2, 600)
    if normalized.startswith("1 Rnd"):
        return (1, 0)
    return (0, 0)


def raw_bout_flags(raw_weight_class: str) -> tuple[int, int, int, int]:
    """Derive frozen bout metadata from a UFCStats legacy weight-class label."""

    normalized = " ".join(raw_weight_class.casefold().split())
    tournament = int("tournament" in normalized)
    interim = int("interim" in normalized and "title" in normalized)
    superfight = int("superfight" in normalized)
    # Tournament-finals are not championship-title bouts in the legacy model.
    title = int("title bout" in normalized and not tournament and not superfight)
    return title, interim, tournament, superfight


def is_standard_label_format(time_format: str) -> bool:
    """Return whether a format belongs in the existing binary model contract."""

    return time_format in {"3 Rnd (5-5-5)", "5 Rnd (5-5-5-5-5)"}


@dataclass(frozen=True, slots=True)
class FighterProfile:
    """Time-invariant profile fields as known by the historical corpus."""

    fighter_id: str
    fighter_name: str
    height_inches: float | None
    reach_inches: float | None
    stance: str | None
    dob: date | None
    fighter_url: str


@dataclass(frozen=True, slots=True)
class FighterFightStats:
    """One corner's aggregate UFCStats totals for a completed bout."""

    kd: int
    sig_landed: int
    sig_attempted: int
    td_landed: int
    td_attempted: int
    sub_attempts: int
    control_seconds: int
    head_landed: int
    body_landed: int
    leg_landed: int
    distance_landed: int
    clinch_landed: int
    ground_landed: int


@dataclass(frozen=True, slots=True)
class Bout:
    """One completed fight, in source chronology, before any state mutation."""

    fight_id: str
    event_date: date
    source_order: int
    fighter_1_id: str
    fighter_1_name: str
    fighter_2_id: str
    fighter_2_name: str
    result_1: Result
    result_2: Result
    raw_weight_class: str
    division: Division
    method: str
    end_round: int
    end_time: str
    total_fight_time_sec: int
    time_format: str
    scheduled_rounds: int
    scheduled_duration_sec: int
    is_title_bout: int
    is_interim_title: int
    is_tournament_final: int
    is_superfight: int
    stats_1: FighterFightStats
    stats_2: FighterFightStats

    @property
    def decisive(self) -> bool:
        return {self.result_1, self.result_2} == {"W", "L"}

    @property
    def is_label_eligible(self) -> bool:
        return (
            self.event_date >= date(2001, 2, 23)
            and self.decisive
            and is_standard_label_format(self.time_format)
        )

    @property
    def winner_id(self) -> str | None:
        if self.result_1 == "W":
            return self.fighter_1_id
        if self.result_2 == "W":
            return self.fighter_2_id
        return None


__all__ = [
    "CATCH_DIVISION",
    "OPEN_DIVISION",
    "UNKNOWN_DIVISION",
    "Bout",
    "Division",
    "FighterFightStats",
    "FighterProfile",
    "Result",
    "division_from_code",
    "is_standard_label_format",
    "normalize_division",
    "raw_bout_flags",
    "scheduled_round_context",
]
