"""Validation-driven model selection and callback-based feature ablation."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelScore:
    """A fitted candidate's held-out validation score."""

    name: str
    family: str
    validation_log_loss: float
    estimator: Any
    validation_calibration_error: float | None = None
    worst_subgroup_log_loss: float | None = None
    subgroup_log_losses: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.family.strip():
            raise ValueError("Model name and family must not be empty.")
        if not math.isfinite(self.validation_log_loss):
            raise ValueError("validation_log_loss must be finite.")
        for name, value in (
            ("validation_calibration_error", self.validation_calibration_error),
            ("worst_subgroup_log_loss", self.worst_subgroup_log_loss),
        ):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite when supplied.")
        if self.subgroup_log_losses is not None and any(
            not key.strip() or not math.isfinite(value)
            for key, value in self.subgroup_log_losses.items()
        ):
            raise ValueError("subgroup_log_losses must have non-empty keys and finite values.")


@dataclass(frozen=True, slots=True)
class ModelSelectionDecision:
    selected: ModelScore
    raw_best: ModelScore
    ranked: tuple[ModelScore, ...]
    effective_tie_tolerance: float
    preferred_logistic_on_tie: bool
    rationale: str


def select_model(
    scores: Sequence[ModelScore],
    *,
    effective_tie_tolerance: float = 0.002,
    logistic_family_names: Sequence[str] = ("logistic", "logistic_regression"),
    max_calibration_error_regression: float = 0.01,
    max_subgroup_log_loss_regression: float = 0.02,
) -> ModelSelectionDecision:
    """Choose from validation diagnostics with a conservative tree-model gate.

    A non-logistic candidate must improve log loss beyond the effective-tie
    tolerance and, when diagnostics are supplied, must not materially regress
    calibration or worst-subgroup log loss relative to the best logistic
    candidate.
    """

    if not 0.0 <= effective_tie_tolerance <= 0.02:
        raise ValueError("effective_tie_tolerance must be in [0, 0.02].")
    if not 0.0 <= max_calibration_error_regression <= 0.1:
        raise ValueError("max_calibration_error_regression must be in [0, 0.1].")
    if not 0.0 <= max_subgroup_log_loss_regression <= 0.2:
        raise ValueError("max_subgroup_log_loss_regression must be in [0, 0.2].")
    candidates = tuple(scores)
    if not candidates:
        raise ValueError("At least one model score is required.")
    names = [candidate.name for candidate in candidates]
    if len(names) != len(set(names)):
        raise ValueError("Model score names must be unique.")

    ranked = tuple(sorted(candidates, key=lambda candidate: candidate.validation_log_loss))
    raw_best = ranked[0]
    logistic_names = {name.casefold() for name in logistic_family_names}
    logistic_candidates = [
        candidate for candidate in ranked if candidate.family.casefold() in logistic_names
    ]
    best_logistic = logistic_candidates[0] if logistic_candidates else None

    selected = raw_best
    rationale: str
    if best_logistic is None or raw_best.family.casefold() in logistic_names:
        rationale = (
            f"{raw_best.name} has the best validation log loss; no tree-model "
            "guardrail changed the ranking."
        )
    elif raw_best.validation_log_loss >= (
        best_logistic.validation_log_loss - effective_tie_tolerance
    ):
        selected = best_logistic
        rationale = (
            f"{raw_best.name} does not improve validation log loss by more than "
            f"{effective_tie_tolerance:.4f}; {best_logistic.name} is preferred "
            "for stability and interpretability."
        )
    else:
        guardrail_failures: list[str] = []
        if (
            raw_best.validation_calibration_error is not None
            and best_logistic.validation_calibration_error is not None
            and raw_best.validation_calibration_error
            > best_logistic.validation_calibration_error + max_calibration_error_regression
        ):
            guardrail_failures.append("calibration error")
        raw_subgroups = raw_best.subgroup_log_losses
        logistic_subgroups = best_logistic.subgroup_log_losses
        aligned_subgroups = (
            set(raw_subgroups) & set(logistic_subgroups)
            if raw_subgroups is not None and logistic_subgroups is not None
            else set()
        )
        if aligned_subgroups:
            assert raw_subgroups is not None
            assert logistic_subgroups is not None
            largest_regression = max(
                raw_subgroups[key] - logistic_subgroups[key] for key in aligned_subgroups
            )
            if largest_regression > max_subgroup_log_loss_regression:
                guardrail_failures.append("aligned subgroup log loss")
        elif (
            raw_best.worst_subgroup_log_loss is not None
            and best_logistic.worst_subgroup_log_loss is not None
            and raw_best.worst_subgroup_log_loss
            > best_logistic.worst_subgroup_log_loss + max_subgroup_log_loss_regression
        ):
            guardrail_failures.append("worst-subgroup log loss")
        if guardrail_failures:
            selected = best_logistic
            rationale = (
                f"{raw_best.name} improved overall validation log loss but exceeded "
                f"the configured {' and '.join(guardrail_failures)} guardrail; "
                f"{best_logistic.name} was selected."
            )
        else:
            rationale = (
                f"{raw_best.name} meaningfully improves validation log loss and "
                "passes the available calibration and subgroup guardrails."
            )
    preferred = selected is not raw_best
    return ModelSelectionDecision(
        selected=selected,
        raw_best=raw_best,
        ranked=ranked,
        effective_tie_tolerance=effective_tie_tolerance,
        preferred_logistic_on_tie=preferred,
        rationale=rationale,
    )


@dataclass(frozen=True, slots=True)
class FeatureAblation:
    group: str
    dropped_features: tuple[str, ...]
    kept_feature_count: int
    score: float
    degradation: float


@dataclass(frozen=True, slots=True)
class FeatureAblationReport:
    baseline_score: float
    lower_is_better: bool
    baseline_features: tuple[str, ...]
    ablations: tuple[FeatureAblation, ...]


def run_feature_ablation(
    feature_names: Sequence[str],
    feature_groups: Mapping[str, Sequence[str]],
    evaluate_features: Callable[[tuple[str, ...]], float],
    *,
    lower_is_better: bool = True,
    maximum_groups: int = 32,
) -> FeatureAblationReport:
    """Evaluate feature groups through a caller-supplied train/evaluate callback.

    ``evaluate_features`` receives the feature names to keep and must return one
    validation score.  It may fit a fresh estimator, but this module makes no
    assumptions about dataframe or model type.  The callback is invoked only
    when this function is explicitly called.
    """

    all_features = tuple(str(feature) for feature in feature_names)
    if not all_features or len(all_features) != len(set(all_features)):
        raise ValueError("feature_names must be non-empty and unique.")
    groups = tuple(feature_groups.items())
    if not groups:
        raise ValueError("At least one feature group is required.")
    if not 1 <= maximum_groups <= 64:
        raise ValueError("maximum_groups must be in [1, 64].")
    if len(groups) > maximum_groups:
        raise ValueError(f"Received {len(groups)} groups; maximum is {maximum_groups}.")
    known = set(all_features)
    normalized_groups: list[tuple[str, tuple[str, ...]]] = []
    for name, group_features in groups:
        if not str(name).strip():
            raise ValueError("Feature group names must not be empty.")
        dropped = tuple(str(feature) for feature in group_features)
        if not dropped:
            raise ValueError(f"Feature group {name!r} is empty.")
        unknown = set(dropped) - known
        if unknown:
            raise ValueError(f"Feature group {name!r} contains unknown features: {sorted(unknown)}")
        normalized_groups.append((str(name), dropped))

    baseline_score = float(evaluate_features(all_features))
    if not math.isfinite(baseline_score):
        raise ValueError("Baseline evaluator score must be finite.")
    rows: list[FeatureAblation] = []
    for name, dropped in normalized_groups:
        drop_set = set(dropped)
        kept = tuple(feature for feature in all_features if feature not in drop_set)
        if not kept:
            raise ValueError(f"Feature group {name!r} drops every feature.")
        score = float(evaluate_features(kept))
        if not math.isfinite(score):
            raise ValueError(f"Ablation score for {name!r} is not finite.")
        degradation = score - baseline_score if lower_is_better else baseline_score - score
        rows.append(
            FeatureAblation(
                group=name,
                dropped_features=dropped,
                kept_feature_count=len(kept),
                score=score,
                degradation=degradation,
            )
        )
    rows.sort(key=lambda row: row.degradation, reverse=True)
    return FeatureAblationReport(
        baseline_score=baseline_score,
        lower_is_better=lower_is_better,
        baseline_features=all_features,
        ablations=tuple(rows),
    )
