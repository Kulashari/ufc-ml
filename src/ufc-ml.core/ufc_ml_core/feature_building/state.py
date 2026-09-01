"""Chronological point-in-time fighter state for the 71-feature contract."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date
from math import log1p

import numpy as np

from ufc_ml_core.exceptions import DataValidationError
from ufc_ml_core.feature_building.domain import Bout, Division, FighterFightStats, FighterProfile

DAYS_PER_YEAR = 365.2425
DEFAULT_LAYOFF_DAYS = 365
MAX_LAYOFF_DAYS = 3650
ELO_INITIAL = 1500.0


@dataclass(frozen=True, slots=True)
class ProfileImputers:
    """Train-era static-profile medians fitted without looking at label targets."""

    age_by_division: Mapping[str, float]
    height_by_division: Mapping[str, float]
    reach_offset_by_division: Mapping[str, float]
    fallback_age: float
    fallback_height: float
    fallback_reach_offset: float

    def age(self, division: Division) -> float:
        return float(self.age_by_division.get(division.code, self.fallback_age))

    def height(self, division: Division) -> float:
        return float(self.height_by_division.get(division.code, self.fallback_height))

    def reach_offset(self, division: Division) -> float:
        return float(self.reach_offset_by_division.get(division.code, self.fallback_reach_offset))


def _median(values: Iterable[float], *, fallback: float, context: str) -> float:
    series = list(values)
    if not series:
        if fallback < 0:
            raise DataValidationError(f"Cannot fit required static imputer for {context}")
        return fallback
    return float(np.median(np.asarray(series, dtype=float)))


def fit_profile_imputers(
    bouts: Iterable[Bout],
    profiles: Mapping[str, FighterProfile],
    *,
    train_end: date,
) -> ProfileImputers:
    """Fit the historic profile imputers from pre-validation training participants.

    Ages use every eligible training appearance because age changes by bout date.
    Height and reach-minus-height use each ``(fighter, division)`` once, which
    reproduces the legacy static-profile fit and avoids frequent fighters
    shifting a median simply by appearing more often.
    """

    ages: dict[str, list[float]] = defaultdict(list)
    heights: dict[str, list[float]] = defaultdict(list)
    reach_offsets: dict[str, list[float]] = defaultdict(list)
    seen_static: set[tuple[str, str]] = set()
    for bout in bouts:
        if not bout.is_label_eligible or bout.event_date > train_end:
            continue
        for fighter_id in (bout.fighter_1_id, bout.fighter_2_id):
            profile = profiles.get(fighter_id)
            if profile is None:
                raise DataValidationError(
                    f"No profile exists for training participant {fighter_id} in {bout.fight_id}"
                )
            if profile.dob is not None and profile.dob < bout.event_date:
                ages[bout.division.code].append(
                    (bout.event_date - profile.dob).days / DAYS_PER_YEAR
                )
            static_key = (fighter_id, bout.division.code)
            if static_key in seen_static:
                continue
            seen_static.add(static_key)
            if profile.height_inches is not None:
                heights[bout.division.code].append(profile.height_inches)
                if profile.reach_inches is not None:
                    reach_offsets[bout.division.code].append(
                        profile.reach_inches - profile.height_inches
                    )

    # The original legacy fallback came from the general training population,
    # not a future/current UFCStats profile summary.
    # Legacy code used the catchweight fit as the generic/unseen-division
    # fallback.  It happens to be well supported in the training corpus and is
    # deliberately not recomputed from post-cutoff/current fighter pages.
    fallback_age = _median(ages.get("CATCH", ()), fallback=-1.0, context="catch age")
    fallback_height = _median(heights.get("CATCH", ()), fallback=-1.0, context="catch height")
    fallback_reach_offset = _median(
        reach_offsets.get("CATCH", ()), fallback=-1.0, context="catch reach offset"
    )
    return ProfileImputers(
        age_by_division={
            division: _median(values, fallback=fallback_age, context=f"{division} age")
            for division, values in ages.items()
        },
        height_by_division={
            division: _median(values, fallback=fallback_height, context=f"{division} height")
            for division, values in heights.items()
        },
        reach_offset_by_division={
            division: _median(values, fallback=fallback_reach_offset, context=f"{division} reach")
            for division, values in reach_offsets.items()
        },
        fallback_age=fallback_age,
        fallback_height=fallback_height,
        fallback_reach_offset=fallback_reach_offset,
    )


@dataclass(frozen=True, slots=True)
class RecentFight:
    """One perspective's completed bout retained for recent-five calculation."""

    decisive: bool
    result: str
    won: bool
    finish_win: bool
    duration_seconds: int
    own: FighterFightStats
    opponent: FighterFightStats


@dataclass(slots=True)
class FighterState:
    """All raw state needed to emit a fighter's pre-fight feature snapshot."""

    prior_fights: int = 0
    career_seconds: int = 0
    decisive_bouts: int = 0
    wins: int = 0
    finish_wins: int = 0
    finish_losses: int = 0
    ko_wins: int = 0
    submission_wins: int = 0
    current_streak: float = 0.0
    elo_global: float = ELO_INITIAL
    elo_by_division: dict[str, float] = field(default_factory=dict)
    division_elo_fights: dict[str, int] = field(default_factory=dict)
    opponent_elo_sum: float = 0.0
    opponent_elo_count: int = 0
    fight_dates: list[date] = field(default_factory=list)
    last_fight_date: date | None = None
    last_division: str = "UNKNOWN"
    sig_landed: int = 0
    sig_attempted: int = 0
    sig_absorbed: int = 0
    sig_absorbed_attempted: int = 0
    kd: int = 0
    kd_absorbed: int = 0
    td_landed: int = 0
    td_attempted: int = 0
    td_absorbed: int = 0
    td_absorbed_attempted: int = 0
    sub_attempts: int = 0
    sub_absorbed: int = 0
    control_seconds: int = 0
    control_allowed_seconds: int = 0
    body_landed: int = 0
    leg_landed: int = 0
    clinch_landed: int = 0
    ground_landed: int = 0
    recent: deque[RecentFight] = field(default_factory=lambda: deque(maxlen=5))

    def division_elo(self, division: Division) -> float:
        return float(self.elo_by_division.get(division.code, ELO_INITIAL))

    def division_fights(self, division: Division) -> int:
        return int(self.division_elo_fights.get(division.code, 0))

    @property
    def career_minutes(self) -> float:
        return self.career_seconds / 60.0


def _stance_features(stance: str | None) -> tuple[str, float, float, float, float]:
    if stance is None:
        return ("Unknown", 0.0, 0.0, 0.0, 1.0)
    normalized = stance.strip().casefold()
    if normalized == "southpaw":
        return ("Southpaw", 1.0, 0.0, 0.0, 0.0)
    if normalized == "switch":
        return ("Switch", 0.0, 1.0, 0.0, 0.0)
    if normalized == "orthodox":
        return ("Orthodox", 0.0, 0.0, 0.0, 0.0)
    return ("Other", 0.0, 0.0, 1.0, 0.0)


def _rate(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        raise DataValidationError("Feature denominator must be positive")
    return numerator / denominator


def _career_performance(state: FighterState) -> dict[str, float]:
    minutes = state.career_minutes
    denominator_minutes = minutes + 15.0
    return {
        "feature_sig_landed_per_min": _rate(state.sig_landed + 52.5, denominator_minutes),
        "feature_sig_absorbed_per_min": _rate(state.sig_absorbed + 52.5, denominator_minutes),
        "feature_sig_accuracy": _rate(state.sig_landed + 45.0, state.sig_attempted + 100.0),
        "feature_sig_defense": _rate(
            (state.sig_absorbed_attempted - state.sig_absorbed) + 55.0,
            state.sig_absorbed_attempted + 100.0,
        ),
        "feature_kd_per15": _rate(15.0 * (state.kd + 0.35), denominator_minutes),
        "feature_kd_absorbed_per15": _rate(15.0 * (state.kd_absorbed + 0.35), denominator_minutes),
        "feature_td_per15": _rate(15.0 * (state.td_landed + 1.5), denominator_minutes),
        "feature_td_absorbed_per15": _rate(15.0 * (state.td_absorbed + 1.5), denominator_minutes),
        "feature_td_accuracy": _rate(state.td_landed + 7.6, state.td_attempted + 20.0),
        "feature_td_defense": _rate(
            (state.td_absorbed_attempted - state.td_absorbed) + 12.4,
            state.td_absorbed_attempted + 20.0,
        ),
        "feature_sub_per15": _rate(15.0 * (state.sub_attempts + 0.5), denominator_minutes),
        "feature_sub_absorbed_per15": _rate(15.0 * (state.sub_absorbed + 0.5), denominator_minutes),
        "feature_control_share": _rate(state.control_seconds + 162.0, state.career_seconds + 900.0),
        "feature_control_allowed_share": _rate(
            state.control_allowed_seconds + 162.0, state.career_seconds + 900.0
        ),
        "feature_body_strike_share": _rate(state.body_landed + 10.0, state.sig_landed + 50.0),
        "feature_leg_strike_share": _rate(state.leg_landed + 7.5, state.sig_landed + 50.0),
        "feature_clinch_strike_share": _rate(state.clinch_landed + 7.5, state.sig_landed + 50.0),
        "feature_ground_strike_share": _rate(state.ground_landed + 7.5, state.sig_landed + 50.0),
    }


def _recent_performance(state: FighterState) -> dict[str, float]:
    """Build the finite-window features from the current pre-fight state.

    The legacy transformer blends the latest five raw bouts with the already
    smoothed *career* feature rather than fitting a second arbitrary prior.
    This makes an early-career fighter shrink smoothly toward their own known
    history, while a debutant retains the career defaults.
    """

    fights = tuple(state.recent)
    count = len(fights)
    duration_seconds = sum(fight.duration_seconds for fight in fights)
    minutes = duration_seconds / 60.0
    own_sig = sum(fight.own.sig_landed for fight in fights)
    opp_sig = sum(fight.opponent.sig_landed for fight in fights)
    own_kd = sum(fight.own.kd for fight in fights)
    opp_kd = sum(fight.opponent.kd for fight in fights)
    own_td = sum(fight.own.td_landed for fight in fights)
    opp_td = sum(fight.opponent.td_landed for fight in fights)
    own_sub = sum(fight.own.sub_attempts for fight in fights)
    opp_sub = sum(fight.opponent.sub_attempts for fight in fights)
    own_control = sum(fight.own.control_seconds for fight in fights)
    opp_control = sum(fight.opponent.control_seconds for fight in fights)
    # The legacy source collapses Draw/NC text.  A recorded decision draw is a
    # half win in the recent form blend, while an NC is absent from this rate's
    # denominator.  ``recent5_count`` still reports raw bouts, separately.
    outcome_count = sum(fight.result != "NC" for fight in fights)
    wins = sum(
        1.0 if fight.result == "W" else 0.5 if fight.result == "D" else 0.0 for fight in fights
    )
    finishes = sum(fight.finish_win for fight in fights)
    career = _career_performance(state)
    decisive_denominator = state.decisive_bouts + 2.0
    career_win_rate = _rate(state.wins + 1.0, decisive_denominator)
    career_finish_win_rate = _rate(state.finish_wins + 1.0, decisive_denominator)
    denominator_minutes = minutes + 15.0
    return {
        "feature_recent5_count": float(count),
        "feature_recent5_win_rate": _rate(wins + 2.0 * career_win_rate, outcome_count + 2.0),
        "feature_recent5_finish_win_rate": _rate(
            finishes + 2.0 * career_finish_win_rate, outcome_count + 2.0
        ),
        "feature_recent5_sig_landed_per_min": _rate(
            own_sig + 15.0 * career["feature_sig_landed_per_min"], denominator_minutes
        ),
        "feature_recent5_sig_absorbed_per_min": _rate(
            opp_sig + 15.0 * career["feature_sig_absorbed_per_min"], denominator_minutes
        ),
        "feature_recent5_kd_per15": _rate(
            15.0 * (own_kd + career["feature_kd_per15"]), denominator_minutes
        ),
        "feature_recent5_kd_absorbed_per15": _rate(
            15.0 * (opp_kd + career["feature_kd_absorbed_per15"]), denominator_minutes
        ),
        "feature_recent5_td_per15": _rate(
            15.0 * (own_td + career["feature_td_per15"]), denominator_minutes
        ),
        "feature_recent5_td_absorbed_per15": _rate(
            15.0 * (opp_td + career["feature_td_absorbed_per15"]), denominator_minutes
        ),
        "feature_recent5_sub_per15": _rate(
            15.0 * (own_sub + career["feature_sub_per15"]), denominator_minutes
        ),
        "feature_recent5_sub_absorbed_per15": _rate(
            15.0 * (opp_sub + career["feature_sub_absorbed_per15"]), denominator_minutes
        ),
        "feature_recent5_control_share": _rate(
            own_control + 900.0 * career["feature_control_share"], duration_seconds + 900.0
        ),
        "feature_recent5_control_allowed_share": _rate(
            opp_control + 900.0 * career["feature_control_allowed_share"],
            duration_seconds + 900.0,
        ),
    }


def fighter_features(
    state: FighterState,
    profile: FighterProfile,
    *,
    event_date: date,
    division: Division,
    imputers: ProfileImputers,
) -> dict[str, float | str | None]:
    """Emit all individual snapshot values immediately before a fight."""

    if profile.dob is not None and profile.dob < event_date:
        age_years = (event_date - profile.dob).days / DAYS_PER_YEAR
        age_missing = 0.0
    else:
        age_years = imputers.age(division)
        age_missing = 1.0
    if profile.height_inches is None:
        height_inches = imputers.height(division)
        height_missing = 1.0
    else:
        height_inches = profile.height_inches
        height_missing = 0.0
    if profile.reach_inches is None:
        reach_inches = height_inches + imputers.reach_offset(division)
        reach_missing = 1.0
    else:
        reach_inches = profile.reach_inches
        reach_missing = 0.0
    stance, southpaw, switch, other_stance, stance_missing = _stance_features(profile.stance)
    if state.last_fight_date is None:
        layoff_days = DEFAULT_LAYOFF_DAYS
    else:
        layoff_days = min((event_date - state.last_fight_date).days, MAX_LAYOFF_DAYS)
    if layoff_days < 0:
        raise DataValidationError(
            f"Fighter {profile.fighter_id} has a future last fight before {event_date}"
        )
    activity_365 = sum((event_date - value).days <= 365 for value in state.fight_dates)
    activity_730 = sum((event_date - value).days <= 730 for value in state.fight_dates)
    decisive_denominator = state.decisive_bouts + 2.0
    values: dict[str, float | str | None] = {
        "stance": stance,
        "dob": profile.dob.isoformat() if profile.dob is not None else None,
        "feature_age_years": age_years,
        "feature_height_inches": height_inches,
        "feature_reach_inches": reach_inches,
        "feature_southpaw": southpaw,
        "feature_switch": switch,
        "feature_other_stance": other_stance,
        "feature_stance_missing": stance_missing,
        "feature_age_missing": age_missing,
        "feature_height_missing": height_missing,
        "feature_reach_missing": reach_missing,
        "feature_debut": float(state.prior_fights == 0),
        "feature_log_prior_fights": log1p(state.prior_fights),
        "feature_log_career_minutes": log1p(state.career_minutes),
        "feature_win_rate": _rate(state.wins + 1.0, decisive_denominator),
        "feature_finish_win_rate": _rate(state.finish_wins + 1.0, decisive_denominator),
        "feature_finish_loss_rate": _rate(state.finish_losses + 1.0, decisive_denominator),
        "feature_ko_win_rate": _rate(state.ko_wins + 0.5, decisive_denominator),
        "feature_submission_win_rate": _rate(state.submission_wins + 0.5, decisive_denominator),
        "feature_current_streak": state.current_streak,
        "feature_elo_global": state.elo_global,
        "feature_elo_division": state.division_elo(division),
        "feature_log_division_elo_fights": log1p(state.division_fights(division)),
        "feature_avg_opponent_elo": (
            # Two 1500-rated pseudo-opponents stabilize debut/small-sample
            # schedules.  This is deliberately independent of whether a bout
            # changed Elo: every completed historical bout is still a known
            # opponent for the activity/experience history.
            (state.opponent_elo_sum + 2.0 * ELO_INITIAL) / (state.opponent_elo_count + 2)
        ),
        "feature_log_layoff_days": log1p(layoff_days),
        "feature_activity_365d": float(activity_365),
        "feature_activity_730d": float(activity_730),
        "feature_prior_fights": float(state.prior_fights),
        "feature_career_minutes": state.career_minutes,
        "feature_division_elo_fights": float(state.division_fights(division)),
        "feature_division_lbs": division.pounds,
        "feature_is_womens": float(division.is_womens),
    }
    values.update(_career_performance(state))
    values.update(_recent_performance(state))
    return values


def _is_ko(method: str) -> bool:
    return method.strip().casefold() in {"ko/tko", "tko - doctor's stoppage"}


def _is_submission(method: str) -> bool:
    return method.strip().casefold() == "submission"


def _is_finish(method: str) -> bool:
    normalized = method.strip().casefold()
    return normalized in {"ko/tko", "tko - doctor's stoppage", "submission", "dq"}


def _elo_expected(rating: float, opponent: float) -> float:
    return float(1.0 / (1.0 + 10.0 ** ((opponent - rating) / 400.0)))


def _elo_k_factor(fights: int) -> float:
    """Return the legacy constant K-factor for global and division ratings."""

    del fights
    return 32.0


def _update_elo_pair(
    first: FighterState,
    second: FighterState,
    *,
    division: Division,
    first_result: str,
    second_result: str,
) -> None:
    """Apply one simultaneous Elo update after the feature row has been emitted."""

    if {first_result, second_result} == {"W", "L"}:
        score_first = 1.0 if first_result == "W" else 0.0
    elif first_result == second_result == "D":
        score_first = 0.5
    else:
        # No contests/overturns remain history but do not alter a rating.
        return
    score_second = 1.0 - score_first
    first_global = first.elo_global
    second_global = second.elo_global
    first_division = first.division_elo(division)
    second_division = second.division_elo(division)
    first.elo_global = first_global + _elo_k_factor(first.prior_fights) * (
        score_first - _elo_expected(first_global, second_global)
    )
    second.elo_global = second_global + _elo_k_factor(second.prior_fights) * (
        score_second - _elo_expected(second_global, first_global)
    )
    first.elo_by_division[division.code] = first_division + _elo_k_factor(
        first.division_fights(division)
    ) * (score_first - _elo_expected(first_division, second_division))
    second.elo_by_division[division.code] = second_division + _elo_k_factor(
        second.division_fights(division)
    ) * (score_second - _elo_expected(second_division, first_division))
    first.division_elo_fights[division.code] = first.division_fights(division) + 1
    second.division_elo_fights[division.code] = second.division_fights(division) + 1


def _update_statistics(
    state: FighterState,
    own: FighterFightStats,
    opponent: FighterFightStats,
) -> None:
    state.sig_landed += own.sig_landed
    state.sig_attempted += own.sig_attempted
    state.sig_absorbed += opponent.sig_landed
    state.sig_absorbed_attempted += opponent.sig_attempted
    state.kd += own.kd
    state.kd_absorbed += opponent.kd
    state.td_landed += own.td_landed
    state.td_attempted += own.td_attempted
    state.td_absorbed += opponent.td_landed
    state.td_absorbed_attempted += opponent.td_attempted
    state.sub_attempts += own.sub_attempts
    state.sub_absorbed += opponent.sub_attempts
    state.control_seconds += own.control_seconds
    state.control_allowed_seconds += opponent.control_seconds
    state.body_landed += own.body_landed
    state.leg_landed += own.leg_landed
    state.clinch_landed += own.clinch_landed
    state.ground_landed += own.ground_landed


def _update_outcome(state: FighterState, *, result: str, method: str) -> tuple[bool, bool]:
    if result == "D":
        # A true recorded draw breaks a run but is not a decisive rate result.
        state.current_streak = 0.0
        return False, False
    if result == "NC":
        # An overturned/no-contest bout remains in fight/stat/activity history,
        # but preserves a fighter's preceding streak and rating.
        return False, False
    if result not in {"W", "L"}:
        raise DataValidationError(f"Unsupported bout result {result!r}")
    state.decisive_bouts += 1
    won = result == "W"
    finish = _is_finish(method)
    if won:
        state.wins += 1
        state.current_streak = state.current_streak + 1.0 if state.current_streak > 0 else 1.0
        if finish:
            state.finish_wins += 1
        if _is_ko(method):
            state.ko_wins += 1
        if _is_submission(method):
            state.submission_wins += 1
    else:
        state.current_streak = state.current_streak - 1.0 if state.current_streak < 0 else -1.0
        if finish:
            state.finish_losses += 1
    return won, bool(won and finish)


def apply_bout(
    states: Mapping[str, FighterState],
    bout: Bout,
) -> None:
    """Mutate both fighter states only after their pre-fight snapshots are read."""

    try:
        first = states[bout.fighter_1_id]
        second = states[bout.fighter_2_id]
    except KeyError as exc:
        raise DataValidationError(f"No state initialized for fight {bout.fight_id}") from exc
    first_pre_elo = first.elo_global
    second_pre_elo = second.elo_global
    first_won, first_finish = _update_outcome(first, result=bout.result_1, method=bout.method)
    second_won, second_finish = _update_outcome(second, result=bout.result_2, method=bout.method)
    _update_statistics(first, bout.stats_1, bout.stats_2)
    _update_statistics(second, bout.stats_2, bout.stats_1)
    first.recent.append(
        RecentFight(
            decisive=bout.decisive,
            result=bout.result_1,
            won=first_won,
            finish_win=first_finish,
            duration_seconds=bout.total_fight_time_sec,
            own=bout.stats_1,
            opponent=bout.stats_2,
        )
    )
    second.recent.append(
        RecentFight(
            decisive=bout.decisive,
            result=bout.result_2,
            won=second_won,
            finish_win=second_finish,
            duration_seconds=bout.total_fight_time_sec,
            own=bout.stats_2,
            opponent=bout.stats_1,
        )
    )
    _update_elo_pair(
        first,
        second,
        division=bout.division,
        first_result=bout.result_1,
        second_result=bout.result_2,
    )
    for state, opponent_elo in ((first, second_pre_elo), (second, first_pre_elo)):
        state.prior_fights += 1
        state.career_seconds += bout.total_fight_time_sec
        state.opponent_elo_sum += opponent_elo
        state.opponent_elo_count += 1
        state.fight_dates.append(bout.event_date)
        state.last_fight_date = bout.event_date
        state.last_division = bout.division.code


__all__ = [
    "DAYS_PER_YEAR",
    "ELO_INITIAL",
    "FighterState",
    "ProfileImputers",
    "RecentFight",
    "apply_bout",
    "fighter_features",
    "fit_profile_imputers",
]
