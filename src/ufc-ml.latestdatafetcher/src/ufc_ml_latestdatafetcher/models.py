"""Typed records shared by parsers, the crawler, storage, and exporters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class FighterReference:
    fighter_id: str
    fighter_name: str
    fighter_url: str


@dataclass(frozen=True, slots=True)
class EventRecord:
    event_id: str
    event_name: str
    event_date: date
    location: str
    event_url: str


@dataclass(frozen=True, slots=True)
class FightReference:
    fight_id: str
    fight_url: str
    bout_order: int
    fighters: tuple[FighterReference, FighterReference]


@dataclass(frozen=True, slots=True)
class ParsedEvent:
    event: EventRecord
    fights: tuple[FightReference, ...]


@dataclass(frozen=True, slots=True)
class FightRecord:
    fight_id: str
    event_id: str
    event_date: date
    bout_order: int
    fight_url: str
    fighter_1_id: str
    fighter_1_name: str
    fighter_2_id: str
    fighter_2_name: str
    fighter_1_status: str
    fighter_2_status: str
    winner_id: str | None
    winner_name: str | None
    raw_weight_class: str
    method: str
    method_details: str
    end_round: int
    end_time: str
    total_fight_time_sec: int
    time_format: str
    scheduled_rounds: int | None
    scheduled_duration_sec: int | None
    referee: str
    is_title_bout: int
    is_interim_title: int
    is_tournament_final: int
    is_superfight: int
    stats_available: int


@dataclass(frozen=True, slots=True)
class FighterFightStats:
    fight_id: str
    event_id: str
    event_date: date
    fighter_id: str
    opponent_id: str
    fighter_name: str
    corner: int
    result: str
    kd: int | None
    sig_str_landed: int | None
    sig_str_attempted: int | None
    total_str_landed: int | None
    total_str_attempted: int | None
    td_landed: int | None
    td_attempted: int | None
    sub_attempts: int | None
    reversals: int | None
    control_seconds: int | None
    head_landed: int | None
    head_attempted: int | None
    body_landed: int | None
    body_attempted: int | None
    leg_landed: int | None
    leg_attempted: int | None
    distance_landed: int | None
    distance_attempted: int | None
    clinch_landed: int | None
    clinch_attempted: int | None
    ground_landed: int | None
    ground_attempted: int | None


@dataclass(frozen=True, slots=True)
class RoundStats:
    fight_id: str
    event_id: str
    event_date: date
    round_number: int
    fighter_id: str
    opponent_id: str
    fighter_name: str
    corner: int
    kd: int | None
    sig_str_landed: int | None
    sig_str_attempted: int | None
    total_str_landed: int | None
    total_str_attempted: int | None
    td_landed: int | None
    td_attempted: int | None
    sub_attempts: int | None
    reversals: int | None
    control_seconds: int | None
    head_landed: int | None
    head_attempted: int | None
    body_landed: int | None
    body_attempted: int | None
    leg_landed: int | None
    leg_attempted: int | None
    distance_landed: int | None
    distance_attempted: int | None
    clinch_landed: int | None
    clinch_attempted: int | None
    ground_landed: int | None
    ground_attempted: int | None


@dataclass(frozen=True, slots=True)
class FighterProfile:
    fighter_id: str
    fighter_name: str
    nickname: str
    height_inches: float | None
    weight_lbs: float | None
    reach_inches: float | None
    stance: str
    dob: date | None
    wins: int | None
    losses: int | None
    draws: int | None
    no_contests: int | None
    slpm_current: float | None
    striking_accuracy_current: float | None
    sapm_current: float | None
    striking_defense_current: float | None
    takedown_average_current: float | None
    takedown_accuracy_current: float | None
    takedown_defense_current: float | None
    submission_average_current: float | None
    fighter_url: str
    profile_as_of_utc: str | None = None
    profile_source_sha256: str | None = None
    profile_origin: str = "ufcstats"


@dataclass(frozen=True, slots=True)
class ParsedFight:
    fight: FightRecord
    totals: tuple[FighterFightStats, ...]
    rounds: tuple[RoundStats, ...]
    fighters: tuple[FighterReference, FighterReference]


@dataclass(frozen=True, slots=True)
class SourcePageRecord:
    page_kind: str
    source_id: str
    url: str
    fetched_at_utc: str
    sha256: str
    cache_path: str
    from_cache: int
    parser_version: str


__all__ = [
    "EventRecord",
    "FightRecord",
    "FightReference",
    "FighterFightStats",
    "FighterProfile",
    "FighterReference",
    "ParsedEvent",
    "ParsedFight",
    "RoundStats",
    "SourcePageRecord",
]
