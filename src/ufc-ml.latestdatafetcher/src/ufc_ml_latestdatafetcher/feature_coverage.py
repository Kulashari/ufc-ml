"""Describe the static source mapping for every configured model feature.

This module validates the feature-to-source contract only.  Whether a particular
local crawl actually contains complete, internally consistent data is the
responsibility of :mod:`ufc_ml_latestdatafetcher.validation`.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ufc_ml_latestdatafetcher.errors import DatasetValidationError

_PROFILE = ("fighter", "fighters.csv", "dob,height_inches,reach_inches,stance")
_OUTCOME = (
    "event + fight",
    "events.csv,fights.csv",
    "event_date,fighter IDs,status,winner,method,total_fight_time_sec,raw_weight_class",
)
_ELO = (
    "event + fight",
    "events.csv,fights.csv",
    "event_date,fighter IDs,status,winner,raw_weight_class",
)
_ACTIVITY = ("event + fight", "events.csv,fights.csv", "event_date,fighter IDs")
_PERFORMANCE = (
    "fight",
    "fights.csv,bout_fighter_totals.csv",
    "duration,KD,SIG L/A,TD L/A,SUB,CTRL,target and position strike L/A",
)

_PROFILE_SUFFIXES = frozenset(
    {
        "age_years",
        "height_inches",
        "reach_inches",
        "southpaw",
        "switch",
        "other_stance",
        "stance_missing",
        "age_missing",
        "height_missing",
        "reach_missing",
    }
)
_OUTCOME_SUFFIXES = frozenset(
    {
        "debut",
        "log_prior_fights",
        "log_career_minutes",
        "win_rate",
        "finish_win_rate",
        "finish_loss_rate",
        "ko_win_rate",
        "submission_win_rate",
        "current_streak",
    }
)
_ELO_SUFFIXES = frozenset(
    {"elo_global", "elo_division", "log_division_elo_fights", "avg_opponent_elo"}
)
_ACTIVITY_SUFFIXES = frozenset({"log_layoff_days", "activity_365d", "activity_730d"})
_PERFORMANCE_SUFFIXES = frozenset(
    {
        "sig_landed_per_min",
        "sig_absorbed_per_min",
        "sig_accuracy",
        "sig_defense",
        "kd_per15",
        "kd_absorbed_per15",
        "td_per15",
        "td_absorbed_per15",
        "td_accuracy",
        "td_defense",
        "sub_per15",
        "sub_absorbed_per15",
        "control_share",
        "control_allowed_share",
        "body_strike_share",
        "leg_strike_share",
        "clinch_strike_share",
        "ground_strike_share",
    }
)
_RECENT_SUFFIXES = frozenset(
    {
        "count",
        "win_rate",
        "finish_win_rate",
        "sig_landed_per_min",
        "sig_absorbed_per_min",
        "kd_per15",
        "kd_absorbed_per15",
        "td_per15",
        "td_absorbed_per15",
        "sub_per15",
        "sub_absorbed_per15",
        "control_share",
        "control_allowed_share",
    }
)
_SHARED: dict[str, tuple[str, str, str]] = {
    "feature_division_lbs": _OUTCOME,
    "feature_is_womens": _OUTCOME,
    "feature_is_catch_weight": _OUTCOME,
    "feature_min_prior_fights": _OUTCOME,
    "feature_total_prior_fights": _OUTCOME,
    "feature_log_total_career_minutes": _OUTCOME,
    "feature_both_debutants": _OUTCOME,
    "feature_exactly_one_debutant": _OUTCOME,
    "feature_same_known_stance": _PROFILE,
    "feature_any_age_missing": _PROFILE,
    "feature_any_height_missing": _PROFILE,
    "feature_any_reach_missing": _PROFILE,
    "feature_min_division_elo_fights": _ELO,
    "feature_mean_elo_global": _ELO,
}


def _requirement(feature_name: str) -> tuple[str, str, str] | None:
    shared = _SHARED.get(feature_name)
    if shared is not None:
        return shared
    prefix = "feature_a_minus_b_"
    if not feature_name.startswith(prefix):
        return None
    suffix = feature_name[len(prefix) :]
    if suffix in _PROFILE_SUFFIXES:
        return _PROFILE
    if suffix in _OUTCOME_SUFFIXES:
        return _OUTCOME
    if suffix in _ELO_SUFFIXES:
        return _ELO
    if suffix in _ACTIVITY_SUFFIXES:
        return _ACTIVITY
    if suffix in _PERFORMANCE_SUFFIXES:
        return _PERFORMANCE
    if suffix.startswith("recent5_") and suffix.removeprefix("recent5_") in _RECENT_SUFFIXES:
        recent = (
            _OUTCOME
            if suffix in {"recent5_count", "recent5_win_rate", "recent5_finish_win_rate"}
            else _PERFORMANCE
        )
        return recent
    return None


def build_feature_coverage_report(
    dictionary_path: Path,
    *,
    expected_feature_count: int,
) -> dict[str, Any]:
    if not dictionary_path.is_file():
        raise DatasetValidationError(f"feature dictionary does not exist: {dictionary_path}")
    with dictionary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    feature_names = [row["column"] for row in rows if row.get("role") == "model_feature"]
    if len(feature_names) != expected_feature_count:
        raise DatasetValidationError(
            f"expected {expected_feature_count} model features, found {len(feature_names)}"
        )
    coverage: dict[str, dict[str, str]] = {}
    unknown: list[str] = []
    for feature_name in feature_names:
        requirement = _requirement(feature_name)
        if requirement is None:
            unknown.append(feature_name)
            continue
        page_types, tables, primitives = requirement
        coverage[feature_name] = {
            "page_types": page_types,
            "normalized_tables": tables,
            "raw_primitives": primitives,
            "construction": (
                "requires a separate chronological feature builder using state strictly before "
                "the target fight"
            ),
        }
    if unknown:
        raise DatasetValidationError(f"features have no crawler-source mapping: {unknown}")
    return {
        "status": "mapping_complete",
        "scope": "static_feature_to_source_mapping_only",
        "actual_crawl_data_validated": False,
        "feature_count": len(feature_names),
        "page_types": [
            "completed event index",
            "event detail",
            "fight detail",
            "fighter A-Z directory",
            "fighter detail",
        ],
        "historical_leakage_rule": (
            "Build a matchup row before applying the target fight. Never use current fighter-page "
            "career summaries for historical features."
        ),
        "limitations": (
            "This report does not inspect normalized CSV schemas, values, page completeness, "
            "chronology, smoothing, imputation, Elo updates, or processed feature rows."
        ),
        "features": coverage,
    }


__all__ = ["build_feature_coverage_report"]
