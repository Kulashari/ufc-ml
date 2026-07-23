"""Leakage-safe, order-symmetric UFC fight inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from math import isfinite, log1p
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from ..exceptions import UFCPredictorError
from .confidence import ConfidenceAssessment, assess_prediction_confidence
from .fighter_lookup import (
    FighterCandidate,
    FighterLookup,
    FighterSnapshot,
    coerce_date,
)

_DIFFERENCE_PREFIX = "feature_a_minus_b_"
_SECONDS_PER_DAY = 86_400
_DAYS_PER_YEAR = 365.2425


@dataclass(frozen=True)
class DivisionSpec:
    code: str
    pounds: float | None
    is_womens: bool
    is_catch_weight: bool = False
    supported: bool = True


_DIVISIONS: dict[str, DivisionSpec] = {
    "M_FLY": DivisionSpec("M_FLY", 125.0, False),
    "M_BANTAM": DivisionSpec("M_BANTAM", 135.0, False),
    "M_FEATHER": DivisionSpec("M_FEATHER", 145.0, False),
    "M_LIGHT": DivisionSpec("M_LIGHT", 155.0, False),
    "M_WELTER": DivisionSpec("M_WELTER", 170.0, False),
    "M_MIDDLE": DivisionSpec("M_MIDDLE", 185.0, False),
    "M_LIGHT_HEAVY": DivisionSpec("M_LIGHT_HEAVY", 205.0, False),
    "M_HEAVY": DivisionSpec("M_HEAVY", 265.0, False),
    "W_STRAW": DivisionSpec("W_STRAW", 115.0, True),
    "W_FLY": DivisionSpec("W_FLY", 125.0, True),
    "W_BANTAM": DivisionSpec("W_BANTAM", 135.0, True),
    "W_FEATHER": DivisionSpec("W_FEATHER", 145.0, True),
    "CATCH": DivisionSpec("CATCH", None, False, is_catch_weight=True),
}

_DIVISION_ALIASES: dict[str, str] = {
    "flyweight": "M_FLY",
    "mens flyweight": "M_FLY",
    "men flyweight": "M_FLY",
    "bantamweight": "M_BANTAM",
    "mens bantamweight": "M_BANTAM",
    "men bantamweight": "M_BANTAM",
    "featherweight": "M_FEATHER",
    "mens featherweight": "M_FEATHER",
    "men featherweight": "M_FEATHER",
    "lightweight": "M_LIGHT",
    "mens lightweight": "M_LIGHT",
    "men lightweight": "M_LIGHT",
    "welterweight": "M_WELTER",
    "mens welterweight": "M_WELTER",
    "men welterweight": "M_WELTER",
    "middleweight": "M_MIDDLE",
    "mens middleweight": "M_MIDDLE",
    "men middleweight": "M_MIDDLE",
    "light heavyweight": "M_LIGHT_HEAVY",
    "mens light heavyweight": "M_LIGHT_HEAVY",
    "men light heavyweight": "M_LIGHT_HEAVY",
    "heavyweight": "M_HEAVY",
    "mens heavyweight": "M_HEAVY",
    "men heavyweight": "M_HEAVY",
    "womens strawweight": "W_STRAW",
    "women strawweight": "W_STRAW",
    "strawweight": "W_STRAW",
    "womens flyweight": "W_FLY",
    "women flyweight": "W_FLY",
    "womens bantamweight": "W_BANTAM",
    "women bantamweight": "W_BANTAM",
    "womens featherweight": "W_FEATHER",
    "women featherweight": "W_FEATHER",
    "catchweight": "CATCH",
    "catch weight": "CATCH",
}


@dataclass(frozen=True)
class MatchupContext:
    """Bout-level values that cannot be inferred from fighter identity alone."""

    division: str | None = None
    division_lbs: float | None = None
    is_womens: bool | None = None
    is_catch_weight: bool | None = None
    extra_features: Mapping[str, float] | None = None


@dataclass(frozen=True)
class OrientationPrediction:
    first_fighter_id: str
    second_fighter_id: str
    raw_probability_first: float
    calibrated_probability_first: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "first_fighter_id": self.first_fighter_id,
            "second_fighter_id": self.second_fighter_id,
            "raw_probability_first": self.raw_probability_first,
            "calibrated_probability_first": self.calibrated_probability_first,
        }


@dataclass(frozen=True)
class FightPrediction:
    fighter_a: FighterCandidate
    fighter_b: FighterCandidate
    prediction_date: date
    model_cutoff: date | None
    division: str
    probability_a: float
    probability_b: float
    prior_fights_a: int
    prior_fights_b: int
    snapshot_date_a: date
    snapshot_date_b: date
    orientation_a_vs_b: OrientationPrediction
    orientation_b_vs_a: OrientationPrediction
    confidence: ConfidenceAssessment
    features_a_vs_b: Mapping[str, float]
    features_b_vs_a: Mapping[str, float]

    @property
    def predicted_winner(self) -> FighterCandidate | None:
        if abs(self.probability_a - 0.5) <= 1e-12:
            return None
        return self.fighter_a if self.probability_a > 0.5 else self.fighter_b

    def to_dict(self, *, include_features: bool = False) -> dict[str, Any]:
        winner = self.predicted_winner
        result: dict[str, Any] = {
            "fighter_a": self.fighter_a.to_dict(),
            "fighter_b": self.fighter_b.to_dict(),
            "prediction_date": self.prediction_date.isoformat(),
            "model_cutoff": (self.model_cutoff.isoformat() if self.model_cutoff else None),
            "dataset_cutoff": (self.model_cutoff.isoformat() if self.model_cutoff else None),
            "division": self.division,
            "probability_a": self.probability_a,
            "probability_b": self.probability_b,
            "prior_ufc_fights_a": self.prior_fights_a,
            "prior_ufc_fights_b": self.prior_fights_b,
            "snapshot_date_a": self.snapshot_date_a.isoformat(),
            "snapshot_date_b": self.snapshot_date_b.isoformat(),
            "predicted_winner_id": winner.fighter_id if winner else None,
            "predicted_winner_name": winner.display_name if winner else None,
            "is_even_probability": winner is None,
            "orientation_a_vs_b": self.orientation_a_vs_b.to_dict(),
            "orientation_b_vs_a": self.orientation_b_vs_a.to_dict(),
            "orientation_disagreement": self.confidence.orientation_disagreement,
            "confidence_tier": self.confidence.tier.value,
            "warnings": [warning.to_dict() for warning in self.confidence.warnings],
            "confidence": self.confidence.to_dict(),
        }
        if include_features:
            result["features_a_vs_b"] = dict(self.features_a_vs_b)
            result["features_b_vs_a"] = dict(self.features_b_vs_a)
        return result


class InferenceError(UFCPredictorError, RuntimeError):
    """Base class for inference failures."""


class SameFighterError(InferenceError):
    pass


class FeatureConstructionError(InferenceError):
    pass


class UnsupportedMatchupError(FeatureConstructionError):
    pass


class InvalidModelOutputError(InferenceError):
    pass


def _candidate_at_snapshot(
    candidate: FighterCandidate,
    snapshot: FighterSnapshot,
) -> FighterCandidate:
    values = snapshot.values
    fighter_name = str(values.get("fighter_name") or candidate.fighter_name).strip()
    display_name = str(values.get("display_name") or fighter_name or candidate.display_name).strip()
    division_value = values.get("last_division") or values.get("division")
    division = (
        str(division_value).strip()
        if division_value is not None and str(division_value).strip()
        else None
    )
    return FighterCandidate(
        fighter_id=candidate.fighter_id,
        fighter_name=fighter_name,
        display_name=display_name,
        division=division,
        dob=_optional_date(values.get("dob")),
        as_of_date=snapshot.as_of_date,
        aliases=candidate.aliases,
    )


def _optional_date(value: Any) -> date | None:
    if value is None:
        return None
    try:
        if pd.isna(value) or str(value).strip() == "":
            return None
    except (TypeError, ValueError):
        pass
    return coerce_date(value, field_name="date")


def _as_float(value: Any, *, field_name: str, default: float | None = None) -> float:
    if value is None or (isinstance(value, str) and not value.strip()):
        if default is not None:
            return float(default)
        raise FeatureConstructionError(f"{field_name} is missing")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise FeatureConstructionError(f"{field_name} must be numeric, got {value!r}") from exc
    if not isfinite(numeric):
        if default is not None:
            return float(default)
        raise FeatureConstructionError(f"{field_name} is not finite")
    return numeric


def _as_nonnegative_int(value: Any, *, field_name: str) -> int:
    numeric = _as_float(value, field_name=field_name, default=0.0)
    if numeric < 0:
        raise FeatureConstructionError(f"{field_name} cannot be negative")
    return round(numeric)


def refresh_dynamic_snapshot(
    snapshot: FighterSnapshot,
    prediction_date: date | datetime | str,
    *,
    default_layoff_days: int = 365,
) -> dict[str, Any]:
    """Refresh date-dependent fields without altering historical aggregates."""

    target_date = coerce_date(prediction_date, field_name="prediction_date")
    if not snapshot.as_of_date < target_date:
        raise FeatureConstructionError(
            "snapshot date must be strictly earlier than prediction date"
        )
    if default_layoff_days < 0:
        raise ValueError("default_layoff_days cannot be negative")

    values = snapshot.to_dict()
    dob = _optional_date(values.get("dob"))
    if dob is not None:
        if dob >= target_date:
            raise FeatureConstructionError(
                f"fighter {snapshot.fighter_id} has DOB on/after prediction date"
            )
        values["feature_age_years"] = (target_date - dob).days / _DAYS_PER_YEAR
        values["feature_age_missing"] = 0.0
    else:
        # Preserve the builder's fitted/imputed value.  Unknown age must remain
        # explicitly marked missing instead of being fabricated at inference.
        values["feature_age_years"] = _as_float(
            values.get("feature_age_years"),
            field_name="feature_age_years",
        )
        values["feature_age_missing"] = 1.0

    last_fight_date = _optional_date(values.get("last_fight_date"))
    if last_fight_date is None:
        layoff_days = default_layoff_days
    else:
        layoff_days = (target_date - last_fight_date).days
        if layoff_days < 0:
            raise FeatureConstructionError(
                f"fighter {snapshot.fighter_id} has a future last_fight_date"
            )
    values["feature_log_layoff_days"] = log1p(layoff_days)
    values["_inference_layoff_days"] = None if last_fight_date is None else layoff_days
    if last_fight_date is None or layoff_days > 365:
        values["feature_activity_365d"] = 0.0
    if last_fight_date is None or layoff_days > 730:
        values["feature_activity_730d"] = 0.0
    return values


def _clean_division_label(value: str) -> str:
    label = value.strip().replace("_", " ")
    label = label.casefold().replace("'", "")
    for removable in (
        "ufc ",
        " interim ",
        " title ",
        " tournament ",
        " superfight ",
        " bout",
    ):
        label = label.replace(removable, " ")
    return " ".join(label.split())


def resolve_division(
    division: str | None,
    *,
    division_lbs: float | None = None,
    is_womens: bool | None = None,
    is_catch_weight: bool | None = None,
) -> DivisionSpec:
    """Normalize a division code/label and validate supplied bout context."""

    code: str
    if division:
        direct_code = division.strip().upper()
        if direct_code in _DIVISIONS:
            code = direct_code
        else:
            label = _clean_division_label(division)
            try:
                code = _DIVISION_ALIASES[label]
            except KeyError as exc:
                if division_lbs is None or is_womens is None:
                    raise UnsupportedMatchupError(
                        f"unsupported division {division!r}; supply division_lbs "
                        "and is_womens to score it explicitly"
                    ) from exc
                custom_pounds = _as_float(division_lbs, field_name="division_lbs")
                return DivisionSpec(
                    code=division.strip(),
                    pounds=custom_pounds,
                    is_womens=bool(is_womens),
                    is_catch_weight=bool(is_catch_weight),
                    supported=False,
                )
    elif division_lbs is not None and is_womens is not None:
        return DivisionSpec(
            code="UNSPECIFIED",
            pounds=_as_float(division_lbs, field_name="division_lbs"),
            is_womens=bool(is_womens),
            is_catch_weight=bool(is_catch_weight),
            supported=False,
        )
    else:
        raise UnsupportedMatchupError(
            "division is required when it cannot be inferred from both snapshots"
        )

    base = _DIVISIONS[code]
    resolved_pounds = (
        _as_float(division_lbs, field_name="division_lbs")
        if division_lbs is not None
        else base.pounds
    )
    if resolved_pounds is None:
        raise UnsupportedMatchupError("catchweight predictions require an explicit division_lbs")
    if (
        base.pounds is not None
        and division_lbs is not None
        and abs(resolved_pounds - base.pounds) > 1e-9
    ):
        raise FeatureConstructionError(
            f"{base.code} requires {base.pounds:g} lb, got {resolved_pounds:g}"
        )
    if is_womens is not None and bool(is_womens) != base.is_womens:
        raise FeatureConstructionError(f"is_womens conflicts with normalized division {base.code}")
    if is_catch_weight is not None and bool(is_catch_weight) != base.is_catch_weight:
        raise FeatureConstructionError(
            f"is_catch_weight conflicts with normalized division {base.code}"
        )
    return DivisionSpec(
        code=base.code,
        pounds=resolved_pounds,
        is_womens=base.is_womens,
        is_catch_weight=base.is_catch_weight,
        supported=base.supported,
    )


def _snapshot_division_code(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw or raw.upper() in {"UNKNOWN", "OPEN"}:
        return None
    direct = raw.upper()
    if direct in _DIVISIONS:
        return direct
    return _DIVISION_ALIASES.get(_clean_division_label(raw))


def _validate_division_specific_state(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    division: DivisionSpec,
    feature_names: Sequence[str],
) -> None:
    uses_division_state = any(
        token in feature_name
        for feature_name in feature_names
        for token in ("elo_division", "division_elo_fights")
    )
    if not uses_division_state:
        return
    for label, values in (("fighter A", first), ("fighter B", second)):
        prior_fights = _as_nonnegative_int(
            values.get("feature_prior_fights"),
            field_name=f"{label}.feature_prior_fights",
        )
        if prior_fights == 0:
            # The zero-history division prior is reconstructable for debutants.
            continue
        snapshot_division = _snapshot_division_code(values.get("last_division"))
        if snapshot_division is None:
            raise UnsupportedMatchupError(
                f"{label} has UFC history but no reconstructable division-specific rating state"
            )
        if snapshot_division != division.code:
            raise UnsupportedMatchupError(
                f"{label}'s snapshot stores division-specific ratings for "
                f"{snapshot_division}, not requested division {division.code}; "
                "per-division history is required to score a division change"
            )


def _known_stance(values: Mapping[str, Any]) -> str | None:
    if (
        _as_float(
            values.get("feature_stance_missing"),
            field_name="feature_stance_missing",
            default=1.0,
        )
        >= 0.5
    ):
        return None
    stance = str(values.get("stance", "")).strip().casefold()
    return stance if stance and stance not in {"unknown", "nan", "none"} else None


def build_matchup_features(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    feature_names: Sequence[str],
    division: DivisionSpec,
    extra_features: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Construct one directional row in the persisted feature order."""

    extras = dict(extra_features or {})
    prior_first = _as_nonnegative_int(
        first.get("feature_prior_fights"), field_name="feature_prior_fights"
    )
    prior_second = _as_nonnegative_int(
        second.get("feature_prior_fights"), field_name="feature_prior_fights"
    )
    career_first = _as_float(
        first.get("feature_career_minutes"),
        field_name="feature_career_minutes",
        default=0.0,
    )
    career_second = _as_float(
        second.get("feature_career_minutes"),
        field_name="feature_career_minutes",
        default=0.0,
    )
    if career_first < 0 or career_second < 0:
        raise FeatureConstructionError("career minutes cannot be negative")

    stance_first = _known_stance(first)
    stance_second = _known_stance(second)
    if division.pounds is None:
        raise FeatureConstructionError("division pounds are required")
    shared: dict[str, float] = {
        "feature_division_lbs": float(division.pounds),
        "feature_is_womens": float(division.is_womens),
        "feature_is_catch_weight": float(division.is_catch_weight),
        "feature_min_prior_fights": float(min(prior_first, prior_second)),
        "feature_total_prior_fights": float(prior_first + prior_second),
        "feature_log_total_career_minutes": log1p(career_first + career_second),
        "feature_both_debutants": float(prior_first == 0 and prior_second == 0),
        "feature_exactly_one_debutant": float((prior_first == 0) != (prior_second == 0)),
        "feature_same_known_stance": float(
            stance_first is not None and stance_second is not None and stance_first == stance_second
        ),
        "feature_any_age_missing": max(
            _as_float(
                first.get("feature_age_missing"),
                field_name="feature_age_missing",
                default=1.0,
            ),
            _as_float(
                second.get("feature_age_missing"),
                field_name="feature_age_missing",
                default=1.0,
            ),
        ),
        "feature_any_height_missing": max(
            _as_float(
                first.get("feature_height_missing"),
                field_name="feature_height_missing",
                default=1.0,
            ),
            _as_float(
                second.get("feature_height_missing"),
                field_name="feature_height_missing",
                default=1.0,
            ),
        ),
        "feature_any_reach_missing": max(
            _as_float(
                first.get("feature_reach_missing"),
                field_name="feature_reach_missing",
                default=1.0,
            ),
            _as_float(
                second.get("feature_reach_missing"),
                field_name="feature_reach_missing",
                default=1.0,
            ),
        ),
        "feature_min_division_elo_fights": min(
            _as_float(
                first.get("feature_division_elo_fights"),
                field_name="feature_division_elo_fights",
                default=0.0,
            ),
            _as_float(
                second.get("feature_division_elo_fights"),
                field_name="feature_division_elo_fights",
                default=0.0,
            ),
        ),
        "feature_mean_elo_global": (
            _as_float(
                first.get("feature_elo_global"),
                field_name="feature_elo_global",
            )
            + _as_float(
                second.get("feature_elo_global"),
                field_name="feature_elo_global",
            )
        )
        / 2.0,
    }

    result: dict[str, float] = {}
    for feature_name in feature_names:
        if feature_name.startswith(_DIFFERENCE_PREFIX):
            suffix = feature_name[len(_DIFFERENCE_PREFIX) :]
            snapshot_feature = f"feature_{suffix}"
            first_value = _as_float(first.get(snapshot_feature), field_name=snapshot_feature)
            second_value = _as_float(second.get(snapshot_feature), field_name=snapshot_feature)
            value = first_value - second_value
        elif feature_name in shared:
            value = shared[feature_name]
        elif feature_name in extras:
            value = _as_float(extras[feature_name], field_name=feature_name)
        else:
            raise FeatureConstructionError(
                f"saved feature {feature_name!r} has no inference-time builder"
            )
        if not isfinite(value):
            raise FeatureConstructionError(f"constructed feature {feature_name!r} is not finite")
        result[feature_name] = float(value)
    return result


def _extract_positive_column(
    values: Any,
    estimator: Any,
    *,
    positive_class: Any,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        result = array
    elif array.ndim == 2 and array.shape[1] == 1:
        result = array[:, 0]
    elif array.ndim == 2:
        classes = getattr(estimator, "classes_", None)
        if classes is None and hasattr(estimator, "named_steps"):
            try:
                classes = next(reversed(estimator.named_steps.values())).classes_
            except (AttributeError, StopIteration):
                classes = None
        column = None
        if classes is not None:
            for index, class_value in enumerate(classes):
                if class_value == positive_class or str(class_value) == str(positive_class):
                    column = index
                    break
        if column is None:
            if array.shape[1] != 2:
                raise InvalidModelOutputError("cannot identify positive-class probability column")
            column = 1
        result = array[:, column]
    else:
        raise InvalidModelOutputError(f"unexpected model output shape {array.shape!r}")
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise InvalidModelOutputError("model returned non-finite probabilities")
    return result.astype(float, copy=False)


def _predict_probabilities(
    estimator: Any,
    frame: pd.DataFrame,
    *,
    positive_class: Any,
) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        probabilities = _extract_positive_column(
            estimator.predict_proba(frame),
            estimator,
            positive_class=positive_class,
        )
    elif hasattr(estimator, "decision_function"):
        scores = _extract_positive_column(
            estimator.decision_function(frame),
            estimator,
            positive_class=positive_class,
        )
        clipped_scores = np.clip(scores, -709.0, 709.0)
        probabilities = 1.0 / (1.0 + np.exp(-clipped_scores))
    else:
        raise InvalidModelOutputError("estimator must expose predict_proba or decision_function")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise InvalidModelOutputError("model probabilities fall outside [0, 1]")
    return probabilities


def _calibrate_probabilities(
    calibrator: Any,
    raw_probabilities: np.ndarray,
    frame: pd.DataFrame,
    *,
    positive_class: Any,
    calibrator_input: str,
) -> np.ndarray:
    if calibrator is None:
        return raw_probabilities.copy()
    if calibrator_input == "features":
        return _predict_probabilities(calibrator, frame, positive_class=positive_class)
    if calibrator_input != "probability":
        raise ValueError("calibrator_input must be 'probability' or 'features'")

    active_calibrator = getattr(calibrator, "calibrator", calibrator)
    column = raw_probabilities.reshape(-1, 1)
    if hasattr(active_calibrator, "transform"):
        calibrated = _extract_positive_column(
            active_calibrator.transform(raw_probabilities),
            active_calibrator,
            positive_class=positive_class,
        )
    elif hasattr(active_calibrator, "predict_proba"):
        calibrated = _extract_positive_column(
            active_calibrator.predict_proba(column),
            active_calibrator,
            positive_class=positive_class,
        )
    elif hasattr(active_calibrator, "predict"):
        calibrated = _extract_positive_column(
            active_calibrator.predict(raw_probabilities),
            active_calibrator,
            positive_class=positive_class,
        )
    elif callable(active_calibrator):
        calibrated = _extract_positive_column(
            active_calibrator(raw_probabilities),
            active_calibrator,
            positive_class=positive_class,
        )
    else:
        raise InvalidModelOutputError(
            "calibrator must expose predict_proba, predict, transform, or __call__"
        )
    if np.any((calibrated < 0.0) | (calibrated > 1.0)):
        raise InvalidModelOutputError("calibrated probabilities fall outside [0, 1]")
    return calibrated


def _normalize_feature_ranges(
    ranges: Mapping[str, Any] | None,
) -> dict[str, tuple[float, float]]:
    if not ranges:
        return {}
    possible_nested_keys = ("feature_ranges", "training_ranges", "ranges")
    source: Mapping[str, Any] = ranges
    for key in possible_nested_keys:
        nested = ranges.get(key)
        if isinstance(nested, Mapping):
            source = nested
            break

    normalized: dict[str, tuple[float, float]] = {}
    for feature_name, specification in source.items():
        lower: Any
        upper: Any
        if isinstance(specification, Mapping):
            lower = specification.get("min", specification.get("lower"))
            upper = specification.get("max", specification.get("upper"))
        elif (
            isinstance(specification, Sequence)
            and not isinstance(specification, (str, bytes))
            and len(specification) == 2
        ):
            lower, upper = specification
        else:
            continue
        if lower is None or upper is None:
            continue
        lower_number = _as_float(lower, field_name=f"{feature_name}.min")
        upper_number = _as_float(upper, field_name=f"{feature_name}.max")
        if lower_number > upper_number:
            raise ValueError(f"invalid training range for {feature_name!r}")
        normalized[str(feature_name)] = (lower_number, upper_number)
    return normalized


def _out_of_range_features(
    rows: Sequence[Mapping[str, float]],
    feature_ranges: Mapping[str, tuple[float, float]],
) -> dict[str, Any]:
    violations: dict[str, Any] = {}
    labels = ("a_vs_b", "b_vs_a")
    for feature_name, (lower, upper) in feature_ranges.items():
        values: dict[str, float] = {}
        for label, row in zip(labels, rows, strict=False):
            if feature_name in row:
                value = row[feature_name]
                if value < lower or value > upper:
                    values[label] = value
        if values:
            violations[feature_name] = {
                "training_min": lower,
                "training_max": upper,
                **values,
            }
    return violations


class FightPredictor:
    """Resolve fighters, rebuild point-in-time features, and predict symmetrically."""

    def __init__(
        self,
        *,
        pipeline: Any,
        snapshots: FighterLookup | pd.DataFrame | Sequence[Mapping[str, Any]] | str,
        feature_names: Sequence[str],
        calibrator: Any | None = None,
        aliases: Mapping[str, str | Sequence[str]] | None = None,
        model_cutoff: date | datetime | str | None = None,
        metadata: Mapping[str, Any] | None = None,
        feature_ranges: Mapping[str, Any] | None = None,
        positive_class: Any = 1,
        calibrator_input: str = "probability",
        default_layoff_days: int = 365,
        supported_divisions: Sequence[str] | None = None,
        stale_after_days: int = 180,
        stale_fighter_days: int = 1_095,
        limited_history_threshold: int = 3,
        orientation_disagreement_threshold: float = 0.10,
        allow_post_cutoff_prediction: bool = True,
    ) -> None:
        cleaned_features = tuple(str(name) for name in feature_names)
        if not cleaned_features:
            raise ValueError("feature_names cannot be empty")
        if len(set(cleaned_features)) != len(cleaned_features):
            raise ValueError("feature_names must be unique")
        if default_layoff_days < 0:
            raise ValueError("default_layoff_days cannot be negative")
        if stale_after_days < 0 or stale_fighter_days < 0:
            raise ValueError("staleness thresholds cannot be negative")
        if limited_history_threshold < 0:
            raise ValueError("limited_history_threshold cannot be negative")
        if not 0.0 <= orientation_disagreement_threshold <= 1.0:
            raise ValueError("orientation_disagreement_threshold must be between 0 and 1")
        if calibrator_input not in {"probability", "features"}:
            raise ValueError("calibrator_input must be 'probability' or 'features'")

        self.pipeline = pipeline
        self.calibrator = calibrator
        self.feature_names = cleaned_features
        self.lookup = (
            snapshots
            if isinstance(snapshots, FighterLookup)
            else FighterLookup(snapshots, aliases=aliases)
        )
        if isinstance(snapshots, FighterLookup) and aliases:
            self.lookup.add_aliases(aliases)
        self.model_cutoff = (
            coerce_date(model_cutoff, field_name="model_cutoff")
            if model_cutoff is not None
            else None
        )
        self.metadata = dict(metadata or {})
        self.feature_ranges = _normalize_feature_ranges(feature_ranges)
        self.positive_class = positive_class
        self.calibrator_input = calibrator_input
        self.default_layoff_days = default_layoff_days
        self.stale_after_days = stale_after_days
        self.stale_fighter_days = stale_fighter_days
        self.limited_history_threshold = limited_history_threshold
        self.orientation_disagreement_threshold = orientation_disagreement_threshold
        self.allow_post_cutoff_prediction = allow_post_cutoff_prediction
        self.supported_divisions = frozenset(
            supported_divisions or tuple(spec.code for spec in _DIVISIONS.values())
        )

    @classmethod
    def from_artifacts(
        cls,
        artifacts: Any,
        *,
        snapshots: FighterLookup | pd.DataFrame | Sequence[Mapping[str, Any]] | str,
        aliases: Mapping[str, str | Sequence[str]] | None = None,
        **overrides: Any,
    ) -> FightPredictor:
        """Construct from a loaded artifact bundle without importing artifacts."""

        metadata = dict(getattr(artifacts, "metadata", {}) or {})
        schema = dict(getattr(artifacts, "schema", {}) or {})
        config = dict(getattr(artifacts, "config", {}) or {})
        inference_config = config.get("inference", config)
        if not isinstance(inference_config, Mapping):
            inference_config = {}
        defaults = {
            "pipeline": artifacts.pipeline,
            "calibrator": getattr(artifacts, "calibrator", None),
            "feature_names": artifacts.feature_names,
            "model_cutoff": getattr(artifacts, "cutoff_date", None),
            "metadata": metadata,
            "feature_ranges": (
                schema.get("feature_ranges")
                or schema.get("training_ranges")
                or metadata.get("feature_ranges")
            ),
            "positive_class": metadata.get("positive_class", 1),
            "calibrator_input": metadata.get("calibrator_input", "probability"),
            "default_layoff_days": inference_config.get("default_layoff_days", 365),
            "supported_divisions": metadata.get("supported_divisions"),
            "stale_after_days": inference_config.get("stale_after_days", 180),
            "stale_fighter_days": inference_config.get("stale_fighter_days", 1_095),
            "limited_history_threshold": inference_config.get("limited_history_threshold", 3),
            "orientation_disagreement_threshold": inference_config.get(
                "orientation_disagreement_threshold", 0.10
            ),
            "allow_post_cutoff_prediction": inference_config.get(
                "allow_post_cutoff_prediction", True
            ),
        }
        defaults.update(overrides)
        return cls(snapshots=snapshots, aliases=aliases, **defaults)

    def predict(
        self,
        fighter_a: str,
        fighter_b: str,
        *,
        prediction_date: date | datetime | str | None = None,
        fighter_a_id: str | None = None,
        fighter_b_id: str | None = None,
        context: MatchupContext | None = None,
    ) -> FightPrediction:
        target_date = (
            date.today()
            if prediction_date is None
            else coerce_date(prediction_date, field_name="prediction_date")
        )
        if (
            self.model_cutoff is not None
            and target_date > self.model_cutoff
            and not self.allow_post_cutoff_prediction
        ):
            raise UnsupportedMatchupError(
                "prediction_date is after the model cutoff and post-cutoff prediction is disabled"
            )
        candidate_a = self.lookup.resolve(fighter_a, fighter_id=fighter_a_id)
        candidate_b = self.lookup.resolve(fighter_b, fighter_id=fighter_b_id)
        if candidate_a.fighter_id == candidate_b.fighter_id:
            raise SameFighterError("fighter_a and fighter_b resolve to the same ID")

        snapshot_a = self.lookup.select_snapshot(candidate_a.fighter_id, target_date)
        snapshot_b = self.lookup.select_snapshot(candidate_b.fighter_id, target_date)
        candidate_a = _candidate_at_snapshot(candidate_a, snapshot_a)
        candidate_b = _candidate_at_snapshot(candidate_b, snapshot_b)
        refreshed_a = refresh_dynamic_snapshot(
            snapshot_a,
            target_date,
            default_layoff_days=self.default_layoff_days,
        )
        refreshed_b = refresh_dynamic_snapshot(
            snapshot_b,
            target_date,
            default_layoff_days=self.default_layoff_days,
        )

        matchup_context = context or MatchupContext()
        division_label = matchup_context.division
        if not division_label:
            division_a = str(refreshed_a.get("last_division", "")).strip()
            division_b = str(refreshed_b.get("last_division", "")).strip()
            if (
                division_a
                and division_a == division_b
                and division_a
                not in {
                    "UNKNOWN",
                    "OPEN",
                }
            ):
                division_label = division_a

        division = resolve_division(
            division_label,
            division_lbs=matchup_context.division_lbs,
            is_womens=matchup_context.is_womens,
            is_catch_weight=matchup_context.is_catch_weight,
        )
        _validate_division_specific_state(
            refreshed_a,
            refreshed_b,
            division=division,
            feature_names=self.feature_names,
        )
        unsupported_reasons: list[str] = []
        if not division.supported or division.code not in self.supported_divisions:
            unsupported_reasons.append(
                f"division {division.code!r} was not declared supported by the model"
            )
        if self.model_cutoff is None:
            unsupported_reasons.append("model cutoff metadata is unavailable")

        features_ab = build_matchup_features(
            refreshed_a,
            refreshed_b,
            feature_names=self.feature_names,
            division=division,
            extra_features=matchup_context.extra_features,
        )
        features_ba = build_matchup_features(
            refreshed_b,
            refreshed_a,
            feature_names=self.feature_names,
            division=division,
            extra_features=matchup_context.extra_features,
        )
        frame = pd.DataFrame(
            [features_ab, features_ba],
            columns=list(self.feature_names),
            dtype=float,
        )
        if tuple(frame.columns) != self.feature_names:
            raise FeatureConstructionError("feature order changed during construction")

        raw = _predict_probabilities(self.pipeline, frame, positive_class=self.positive_class)
        calibrated = _calibrate_probabilities(
            self.calibrator,
            raw,
            frame,
            positive_class=self.positive_class,
            calibrator_input=self.calibrator_input,
        )
        if raw.shape[0] != 2 or calibrated.shape[0] != 2:
            raise InvalidModelOutputError("model must return one probability for each orientation")

        probability_a_from_ab = float(calibrated[0])
        probability_a_from_ba = 1.0 - float(calibrated[1])
        probability_a = (probability_a_from_ab + probability_a_from_ba) / 2.0
        probability_a = min(1.0, max(0.0, probability_a))
        probability_b = 1.0 - probability_a
        disagreement = abs(probability_a_from_ab - probability_a_from_ba)

        range_violations = _out_of_range_features((features_ab, features_ba), self.feature_ranges)
        prior_a = _as_nonnegative_int(
            refreshed_a.get("feature_prior_fights"),
            field_name="feature_prior_fights",
        )
        prior_b = _as_nonnegative_int(
            refreshed_b.get("feature_prior_fights"),
            field_name="feature_prior_fights",
        )
        days_after_cutoff = (
            max(0, (target_date - self.model_cutoff).days) if self.model_cutoff else 0
        )
        confidence = assess_prediction_confidence(
            prior_fights_a=prior_a,
            prior_fights_b=prior_b,
            orientation_disagreement=disagreement,
            snapshot_age_days_a=(target_date - snapshot_a.as_of_date).days,
            snapshot_age_days_b=(target_date - snapshot_b.as_of_date).days,
            layoff_days_a=refreshed_a.get("_inference_layoff_days"),
            layoff_days_b=refreshed_b.get("_inference_layoff_days"),
            days_after_cutoff=days_after_cutoff,
            out_of_range=range_violations,
            unsupported_reasons=unsupported_reasons,
            limited_history_threshold=self.limited_history_threshold,
            stale_snapshot_days=self.stale_after_days,
            stale_fighter_days=self.stale_fighter_days,
            disagreement_warning_threshold=self.orientation_disagreement_threshold,
            disagreement_critical_threshold=max(
                0.20,
                min(1.0, self.orientation_disagreement_threshold * 2.0),
            ),
        )

        return FightPrediction(
            fighter_a=candidate_a,
            fighter_b=candidate_b,
            prediction_date=target_date,
            model_cutoff=self.model_cutoff,
            division=division.code,
            probability_a=probability_a,
            probability_b=probability_b,
            prior_fights_a=prior_a,
            prior_fights_b=prior_b,
            snapshot_date_a=snapshot_a.as_of_date,
            snapshot_date_b=snapshot_b.as_of_date,
            orientation_a_vs_b=OrientationPrediction(
                first_fighter_id=candidate_a.fighter_id,
                second_fighter_id=candidate_b.fighter_id,
                raw_probability_first=float(raw[0]),
                calibrated_probability_first=float(calibrated[0]),
            ),
            orientation_b_vs_a=OrientationPrediction(
                first_fighter_id=candidate_b.fighter_id,
                second_fighter_id=candidate_a.fighter_id,
                raw_probability_first=float(raw[1]),
                calibrated_probability_first=float(calibrated[1]),
            ),
            confidence=confidence,
            features_a_vs_b=MappingProxyType(features_ab),
            features_b_vs_a=MappingProxyType(features_ba),
        )


__all__ = [
    "DivisionSpec",
    "FeatureConstructionError",
    "FightPrediction",
    "FightPredictor",
    "InferenceError",
    "InvalidModelOutputError",
    "MatchupContext",
    "OrientationPrediction",
    "SameFighterError",
    "UnsupportedMatchupError",
    "build_matchup_features",
    "refresh_dynamic_snapshot",
    "resolve_division",
]
