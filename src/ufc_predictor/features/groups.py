"""Deterministic semantic grouping for model features."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum


class FeatureGroup(StrEnum):
    """High-level feature families used in reports and ablation checks."""

    PHYSICAL = "physical"
    EXPERIENCE = "experience"
    RATINGS = "ratings"
    ACTIVITY = "activity"
    STRIKING = "striking"
    GRAPPLING = "grappling"
    RECENT_FORM = "recent_form"
    CONTEXT = "context"
    MISSINGNESS = "missingness"
    OTHER = "other"


REQUIRED_ABLATION_GROUPS = (
    "elo",
    "physical_attributes",
    "career_performance",
    "recent_five_performance",
    "activity_and_layoff",
    "stance_and_style",
    "experience",
)


def classify_feature(column: str) -> FeatureGroup:
    """Classify one feature using ordered, stable naming rules.

    Rule order matters. For example, recent striking features belong to
    ``RECENT_FORM`` rather than ``STRIKING``.
    """

    name = column.casefold()

    if "missing" in name or "debut" in name:
        return FeatureGroup.MISSINGNESS
    if "recent" in name:
        return FeatureGroup.RECENT_FORM
    if "elo" in name:
        return FeatureGroup.RATINGS
    if "layoff" in name or "activity_" in name:
        return FeatureGroup.ACTIVITY
    if any(token in name for token in ("age", "height", "reach", "stance", "southpaw", "switch")):
        return FeatureGroup.PHYSICAL
    if any(
        token in name
        for token in (
            "prior_fights",
            "career_minutes",
            "win_rate",
            "loss_rate",
            "current_streak",
        )
    ):
        return FeatureGroup.EXPERIENCE
    if any(
        token in name
        for token in (
            "_sig_",
            "_kd_",
            "strike_share",
            "clinch_strike",
            "ground_strike",
        )
    ):
        return FeatureGroup.STRIKING
    if any(token in name for token in ("_td_", "_sub_", "control_share", "control_allowed")):
        return FeatureGroup.GRAPPLING
    if any(token in name for token in ("division_lbs", "is_womens", "is_catch_weight")):
        return FeatureGroup.CONTEXT
    return FeatureGroup.OTHER


def group_features(
    columns: Iterable[str],
) -> dict[FeatureGroup, tuple[str, ...]]:
    """Group columns while preserving their supplied canonical order."""

    grouped: dict[FeatureGroup, list[str]] = {group: [] for group in FeatureGroup}
    for column in columns:
        grouped[classify_feature(column)].append(column)
    return {group: tuple(grouped[group]) for group in FeatureGroup if grouped[group]}


def ablation_feature_groups(
    columns: Iterable[str],
) -> dict[str, tuple[str, ...]]:
    """Build the seven required, potentially overlapping ablation groups."""

    ordered = tuple(columns)

    def matching(*tokens: str, exclude: tuple[str, ...] = ()) -> tuple[str, ...]:
        return tuple(
            column
            for column in ordered
            if any(token in column.casefold() for token in tokens)
            and not any(token in column.casefold() for token in exclude)
        )

    groups = {
        "elo": matching("elo"),
        "physical_attributes": matching("age", "height", "reach"),
        "career_performance": matching(
            "win_rate",
            "finish_",
            "ko_win",
            "submission_win",
            "sig_",
            "kd_",
            "td_",
            "sub_",
            "control_",
            exclude=("recent5",),
        ),
        "recent_five_performance": matching("recent5"),
        "activity_and_layoff": matching("activity_", "layoff"),
        "stance_and_style": matching(
            "stance",
            "southpaw",
            "switch",
            "strike_share",
        ),
        "experience": matching(
            "prior_fights",
            "career_minutes",
            "debut",
            "current_streak",
            "division_elo_fights",
        ),
    }
    return {name: values for name, values in groups.items() if values}


__all__ = [
    "REQUIRED_ABLATION_GROUPS",
    "FeatureGroup",
    "ablation_feature_groups",
    "classify_feature",
    "group_features",
]
