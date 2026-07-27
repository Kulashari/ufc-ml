"""Prediction-confidence policy and user-facing warnings.

The probability produced by a classifier is not, by itself, a statement about
data quality.  Confidence here describes how well the requested matchup is
supported by the model's training distribution and available fighter history.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ConfidenceTier(StrEnum):
    STANDARD = "standard"
    REDUCED = "reduced"
    LOW = "low"
    UNSUPPORTED = "unsupported"


class WarningSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class WarningCode(StrEnum):
    DEBUT = "debut"
    LIMITED_HISTORY = "limited_history"
    STALE = "stale"
    AFTER_CUTOFF = "after_cutoff"
    ORIENTATION_DISAGREEMENT = "orientation_disagreement"
    OUT_OF_RANGE = "out_of_range"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class PredictionWarning:
    code: WarningCode
    severity: WarningSeverity
    message: str
    details: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "severity": self.severity.value,
            "message": self.message,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class ConfidenceAssessment:
    tier: ConfidenceTier
    score: float
    warnings: tuple[PredictionWarning, ...]
    orientation_disagreement: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier.value,
            "score": self.score,
            "orientation_disagreement": self.orientation_disagreement,
            "warnings": [warning.to_dict() for warning in self.warnings],
        }


def _tier_for_score(score: float, *, unsupported: bool = False) -> ConfidenceTier:
    if unsupported:
        return ConfidenceTier.UNSUPPORTED
    if score >= 0.85:
        return ConfidenceTier.STANDARD
    if score >= 0.65:
        return ConfidenceTier.REDUCED
    if score >= 0.40:
        return ConfidenceTier.LOW
    return ConfidenceTier.UNSUPPORTED


def assess_prediction_confidence(
    *,
    prior_fights_a: int,
    prior_fights_b: int,
    orientation_disagreement: float,
    snapshot_age_days_a: int = 0,
    snapshot_age_days_b: int = 0,
    layoff_days_a: int | None = None,
    layoff_days_b: int | None = None,
    days_after_cutoff: int = 0,
    out_of_range: Mapping[str, Any] | Iterable[str] = (),
    unsupported_reasons: Sequence[str] = (),
    limited_history_threshold: int = 3,
    stale_snapshot_days: int = 365,
    stale_fighter_days: int = 1_095,
    disagreement_warning_threshold: float = 0.10,
    disagreement_critical_threshold: float = 0.20,
) -> ConfidenceAssessment:
    """Assess support for a prediction independently of its win probability."""

    if prior_fights_a < 0 or prior_fights_b < 0:
        raise ValueError("prior fight counts cannot be negative")
    if not 0.0 <= orientation_disagreement <= 1.0:
        raise ValueError("orientation_disagreement must be between 0 and 1")
    if limited_history_threshold < 0:
        raise ValueError("limited_history_threshold cannot be negative")
    if stale_snapshot_days < 0 or stale_fighter_days < 0:
        raise ValueError("staleness thresholds cannot be negative")
    if not 0.0 <= disagreement_warning_threshold <= 1.0:
        raise ValueError("disagreement_warning_threshold must be between 0 and 1")
    if not disagreement_warning_threshold <= disagreement_critical_threshold <= 1.0:
        raise ValueError(
            "disagreement_critical_threshold must be between the warning threshold and 1"
        )

    warnings: list[PredictionWarning] = []
    score = 1.0

    debutants = [
        label
        for label, count in (("fighter_a", prior_fights_a), ("fighter_b", prior_fights_b))
        if count == 0
    ]
    if debutants:
        both_debuting = len(debutants) == 2
        score -= 0.42 if both_debuting else 0.30
        warnings.append(
            PredictionWarning(
                code=WarningCode.DEBUT,
                severity=(WarningSeverity.CRITICAL if both_debuting else WarningSeverity.WARNING),
                message=(
                    "Both fighters have no recorded UFC history."
                    if both_debuting
                    else "One fighter has no recorded UFC history."
                ),
                details={"fighters": debutants},
            )
        )

    minimum_history = min(prior_fights_a, prior_fights_b)
    if minimum_history < limited_history_threshold:
        score -= 0.14
        warnings.append(
            PredictionWarning(
                code=WarningCode.LIMITED_HISTORY,
                severity=WarningSeverity.WARNING,
                message=(
                    "At least one fighter has limited UFC history; rate estimates "
                    "will be less stable."
                ),
                details={
                    "fighter_a_prior_fights": prior_fights_a,
                    "fighter_b_prior_fights": prior_fights_b,
                    "threshold": limited_history_threshold,
                },
            )
        )

    stale_details: dict[str, Any] = {}
    snapshot_ages = {
        "fighter_a_snapshot_age_days": max(0, snapshot_age_days_a),
        "fighter_b_snapshot_age_days": max(0, snapshot_age_days_b),
    }
    if max(snapshot_ages.values()) > stale_snapshot_days:
        stale_details.update(snapshot_ages)
        stale_details["snapshot_threshold_days"] = stale_snapshot_days
        score -= 0.16 if max(snapshot_ages.values()) <= 730 else 0.25

    layoff_values = {
        "fighter_a_layoff_days": layoff_days_a,
        "fighter_b_layoff_days": layoff_days_b,
    }
    known_layoffs = [value for value in layoff_values.values() if value is not None]
    if known_layoffs and max(known_layoffs) > stale_fighter_days:
        stale_details.update(layoff_values)
        stale_details["layoff_threshold_days"] = stale_fighter_days
        score -= 0.13

    if stale_details:
        warnings.append(
            PredictionWarning(
                code=WarningCode.STALE,
                severity=WarningSeverity.WARNING,
                message=(
                    "The prediction relies on a stale snapshot or a long period "
                    "of fighter inactivity."
                ),
                details=stale_details,
            )
        )

    if days_after_cutoff > 0:
        if days_after_cutoff <= 180:
            cutoff_penalty = 0.03
            severity = WarningSeverity.INFO
        elif days_after_cutoff <= 365:
            cutoff_penalty = 0.07
            severity = WarningSeverity.WARNING
        else:
            cutoff_penalty = 0.15
            severity = WarningSeverity.WARNING
        score -= cutoff_penalty
        warnings.append(
            PredictionWarning(
                code=WarningCode.AFTER_CUTOFF,
                severity=severity,
                message=("The current prediction time is after the model's training data cutoff."),
                details={
                    "days_after_cutoff": days_after_cutoff,
                    "rolling_activity_note": (
                        "Rolling activity counts come from the selected snapshot; "
                        "only windows guaranteed empty by the recalculated layoff "
                        "are reset to zero."
                    ),
                },
            )
        )

    if orientation_disagreement >= disagreement_warning_threshold:
        is_critical = orientation_disagreement >= disagreement_critical_threshold
        score -= 0.32 if is_critical else 0.17
        warnings.append(
            PredictionWarning(
                code=WarningCode.ORIENTATION_DISAGREEMENT,
                severity=(WarningSeverity.CRITICAL if is_critical else WarningSeverity.WARNING),
                message=(
                    "Swapping fighter order materially changes the model's implied probability."
                ),
                details={
                    "absolute_disagreement": orientation_disagreement,
                    "warning_threshold": disagreement_warning_threshold,
                    "critical_threshold": disagreement_critical_threshold,
                },
            )
        )

    if isinstance(out_of_range, Mapping):
        out_of_range_details = dict(out_of_range)
        out_of_range_names = tuple(out_of_range_details)
    else:
        out_of_range_names = tuple(sorted(set(out_of_range)))
        out_of_range_details = {"features": list(out_of_range_names)}
    if out_of_range_names:
        score -= min(0.26, 0.06 + 0.02 * len(out_of_range_names))
        warnings.append(
            PredictionWarning(
                code=WarningCode.OUT_OF_RANGE,
                severity=(
                    WarningSeverity.CRITICAL
                    if len(out_of_range_names) >= 8
                    else WarningSeverity.WARNING
                ),
                message=(
                    "One or more matchup features fall outside the model's recorded training range."
                ),
                details=out_of_range_details,
            )
        )

    clean_unsupported = tuple(
        dict.fromkeys(reason.strip() for reason in unsupported_reasons if reason.strip())
    )
    if clean_unsupported:
        score -= min(0.55, 0.35 + 0.05 * (len(clean_unsupported) - 1))
        warnings.append(
            PredictionWarning(
                code=WarningCode.UNSUPPORTED,
                severity=WarningSeverity.CRITICAL,
                message=(
                    "The matchup includes conditions not represented by the "
                    "supported inference contract."
                ),
                details={"reasons": list(clean_unsupported)},
            )
        )

    score = min(1.0, max(0.0, score))
    return ConfidenceAssessment(
        tier=_tier_for_score(score, unsupported=bool(clean_unsupported)),
        score=round(score, 6),
        warnings=tuple(warnings),
        orientation_disagreement=orientation_disagreement,
    )


__all__ = [
    "ConfidenceAssessment",
    "ConfidenceTier",
    "PredictionWarning",
    "WarningCode",
    "WarningSeverity",
    "assess_prediction_confidence",
]
