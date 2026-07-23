"""Deterministic logistic-regression baselines for binary fight outcomes.

Nothing in this module fits a model at import time.  Training occurs only when
``fit_logistic_candidate`` or ``search_logistic_candidates`` is called.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

Penalty = Literal["l2", "elasticnet"]
ClassWeight = Literal["balanced"] | None
MAX_LOGISTIC_CANDIDATES = 24


@dataclass(frozen=True, slots=True)
class LogisticCandidate:
    """A bounded logistic-regression configuration."""

    name: str
    penalty: Penalty
    c: float
    l1_ratio: float | None = None
    max_iter: int = 2_000
    tolerance: float = 1e-4
    class_weight: ClassWeight = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Candidate name must not be empty.")
        if self.penalty not in {"l2", "elasticnet"}:
            raise ValueError("penalty must be 'l2' or 'elasticnet'.")
        if not 1e-4 <= self.c <= 1e3:
            raise ValueError("c must be in the bounded interval [1e-4, 1e3].")
        if self.penalty == "elasticnet":
            if self.l1_ratio is None or not 0.0 <= self.l1_ratio <= 1.0:
                raise ValueError("elastic-net candidates require l1_ratio in [0, 1].")
        elif self.l1_ratio is not None:
            raise ValueError("l1_ratio is only valid for elastic-net candidates.")
        if not 100 <= self.max_iter <= 20_000:
            raise ValueError("max_iter must be in [100, 20_000].")
        if not 1e-8 <= self.tolerance <= 1e-1:
            raise ValueError("tolerance must be in [1e-8, 1e-1].")
        if self.class_weight not in {None, "balanced"}:
            raise ValueError("class_weight must be None or 'balanced'.")


DEFAULT_LOGISTIC_CANDIDATES: tuple[LogisticCandidate, ...] = (
    LogisticCandidate("l2_c0.05", "l2", 0.05),
    LogisticCandidate("l2_c0.2", "l2", 0.2),
    LogisticCandidate("l2_c1", "l2", 1.0),
    LogisticCandidate("l2_c5", "l2", 5.0),
    LogisticCandidate("elastic_c0.1_l10.25", "elasticnet", 0.1, 0.25),
    LogisticCandidate("elastic_c0.3_l10.5", "elasticnet", 0.3, 0.5),
    LogisticCandidate("elastic_c1_l10.25", "elasticnet", 1.0, 0.25),
    LogisticCandidate("elastic_c1_l10.75", "elasticnet", 1.0, 0.75),
    LogisticCandidate("elastic_c3_l10.5", "elasticnet", 3.0, 0.5),
)


@dataclass(frozen=True, slots=True)
class LogisticFitResult:
    """A fitted pipeline and its convergence information."""

    estimator: Pipeline
    candidate: LogisticCandidate
    converged: bool
    retried: bool
    max_iter_used: int
    iterations_used: int


@dataclass(frozen=True, slots=True)
class LogisticTrial:
    """Validation result for one candidate."""

    candidate: LogisticCandidate
    validation_log_loss: float | None
    converged: bool
    retried: bool
    iterations_used: int | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class LogisticSearchResult:
    """The validation-selected logistic model and all candidate outcomes."""

    estimator: Pipeline
    candidate: LogisticCandidate
    validation_log_loss: float
    trials: tuple[LogisticTrial, ...]


@dataclass(frozen=True, slots=True)
class LogisticCoefficient:
    """Coefficient on both standardized and original feature scales."""

    feature: str
    standardized_coefficient: float
    raw_unit_coefficient: float
    odds_ratio_per_standard_deviation: float


def _as_binary_target(
    values: Any,
    *,
    name: str,
    require_both_classes: bool,
) -> np.ndarray:
    target = np.asarray(values).reshape(-1)
    if target.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isin(target, (0, 1))):
        raise ValueError(f"{name} must contain only binary values 0 and 1.")
    target = target.astype(np.int8, copy=False)
    if require_both_classes and np.unique(target).size != 2:
        raise ValueError(f"{name} must contain both binary classes.")
    return target


def _make_pipeline(
    candidate: LogisticCandidate,
    *,
    random_state: int,
    max_iter: int,
    scale_with_mean: bool,
) -> Pipeline:
    solver = "lbfgs" if candidate.penalty == "l2" else "saga"
    classifier = LogisticRegression(
        C=candidate.c,
        penalty=candidate.penalty,
        l1_ratio=candidate.l1_ratio,
        solver=solver,
        max_iter=max_iter,
        tol=candidate.tolerance,
        class_weight=candidate.class_weight,
        random_state=random_state,
        n_jobs=1,
    )
    return Pipeline(
        steps=(
            ("scale", StandardScaler(with_mean=scale_with_mean)),
            ("classifier", classifier),
        )
    )


def _fit_once(
    x_train: Any,
    y_train: np.ndarray,
    candidate: LogisticCandidate,
    *,
    random_state: int,
    max_iter: int,
    scale_with_mean: bool,
) -> tuple[Pipeline, bool, int]:
    estimator = _make_pipeline(
        candidate,
        random_state=random_state,
        max_iter=max_iter,
        scale_with_mean=scale_with_mean,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(x_train, y_train)
    converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
    classifier = estimator.named_steps["classifier"]
    iterations = int(np.max(np.asarray(classifier.n_iter_)))
    return estimator, converged, iterations


def fit_logistic_candidate(
    x_train: Any,
    y_train: Any,
    candidate: LogisticCandidate,
    *,
    random_state: int = 42,
    retry_on_nonconvergence: bool = True,
    retry_multiplier: int = 3,
    max_iter_cap: int = 20_000,
    scale_with_mean: bool = True,
    require_convergence: bool = True,
) -> LogisticFitResult:
    """Fit one explicitly supplied candidate.

    A convergence warning triggers one clean refit with a larger iteration
    budget.  Callers may set ``require_convergence=False`` to inspect a model
    that still did not converge, but search excludes such models by default.

    ``scale_with_mean=False`` supports sparse feature matrices.
    """

    target = _as_binary_target(y_train, name="y_train", require_both_classes=True)
    if retry_multiplier < 2:
        raise ValueError("retry_multiplier must be at least 2.")
    if max_iter_cap < candidate.max_iter or max_iter_cap > 100_000:
        raise ValueError("max_iter_cap must be at least candidate.max_iter and at most 100,000.")

    estimator, converged, iterations = _fit_once(
        x_train,
        target,
        candidate,
        random_state=random_state,
        max_iter=candidate.max_iter,
        scale_with_mean=scale_with_mean,
    )
    retried = False
    max_iter_used = candidate.max_iter

    if not converged and retry_on_nonconvergence:
        retried = True
        max_iter_used = min(candidate.max_iter * retry_multiplier, max_iter_cap)
        estimator, converged, iterations = _fit_once(
            x_train,
            target,
            candidate,
            random_state=random_state,
            max_iter=max_iter_used,
            scale_with_mean=scale_with_mean,
        )

    if require_convergence and not converged:
        raise RuntimeError(
            f"Logistic candidate {candidate.name!r} failed to converge after "
            f"{max_iter_used:,} iterations."
        )

    return LogisticFitResult(
        estimator=estimator,
        candidate=candidate,
        converged=converged,
        retried=retried,
        max_iter_used=max_iter_used,
        iterations_used=iterations,
    )


def predict_positive_probability(estimator: Pipeline, features: Any) -> np.ndarray:
    """Return finite positive-class probabilities from a fitted pipeline."""

    probabilities = np.asarray(estimator.predict_proba(features), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("Expected binary predict_proba output with two columns.")
    positive = probabilities[:, 1]
    if not np.all(np.isfinite(positive)):
        raise ValueError("Estimator returned non-finite probabilities.")
    return positive


def search_logistic_candidates(
    x_train: Any,
    y_train: Any,
    x_validation: Any,
    y_validation: Any,
    *,
    candidates: Sequence[LogisticCandidate] = DEFAULT_LOGISTIC_CANDIDATES,
    random_state: int = 42,
    max_candidates: int = MAX_LOGISTIC_CANDIDATES,
    scale_with_mean: bool = True,
) -> LogisticSearchResult:
    """Select a bounded candidate using validation log loss only."""

    candidate_list = tuple(candidates)
    if not candidate_list:
        raise ValueError("At least one logistic candidate is required.")
    if not 1 <= max_candidates <= MAX_LOGISTIC_CANDIDATES:
        raise ValueError(f"max_candidates must be in [1, {MAX_LOGISTIC_CANDIDATES}].")
    if len(candidate_list) > max_candidates:
        raise ValueError(
            f"Received {len(candidate_list)} candidates; the configured bound is {max_candidates}."
        )
    names = [candidate.name for candidate in candidate_list]
    if len(names) != len(set(names)):
        raise ValueError("Logistic candidate names must be unique.")

    train_target = _as_binary_target(y_train, name="y_train", require_both_classes=True)
    validation_target = _as_binary_target(
        y_validation,
        name="y_validation",
        require_both_classes=False,
    )

    trials: list[LogisticTrial] = []
    successful: list[tuple[float, LogisticFitResult]] = []
    for candidate in candidate_list:
        try:
            fitted = fit_logistic_candidate(
                x_train,
                train_target,
                candidate,
                random_state=random_state,
                scale_with_mean=scale_with_mean,
                require_convergence=True,
            )
            probability = predict_positive_probability(fitted.estimator, x_validation)
            score = float(log_loss(validation_target, probability, labels=(0, 1)))
            successful.append((score, fitted))
            trials.append(
                LogisticTrial(
                    candidate=candidate,
                    validation_log_loss=score,
                    converged=fitted.converged,
                    retried=fitted.retried,
                    iterations_used=fitted.iterations_used,
                )
            )
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            trials.append(
                LogisticTrial(
                    candidate=candidate,
                    validation_log_loss=None,
                    converged=False,
                    retried=False,
                    iterations_used=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    if not successful:
        details = "; ".join(
            f"{trial.candidate.name}: {trial.error or 'did not converge'}" for trial in trials
        )
        raise RuntimeError(f"All logistic candidates failed. {details}")

    # True validation loss is primary.  At numerical ties, prefer L2 and then
    # stronger regularization for a more stable baseline.
    score, selected = min(
        successful,
        key=lambda item: (
            item[0],
            0 if item[1].candidate.penalty == "l2" else 1,
            item[1].candidate.c,
        ),
    )
    return LogisticSearchResult(
        estimator=selected.estimator,
        candidate=selected.candidate,
        validation_log_loss=score,
        trials=tuple(trials),
    )


def extract_logistic_coefficients(
    estimator: Pipeline,
    feature_names: Sequence[str],
    *,
    sort_by_magnitude: bool = True,
) -> tuple[LogisticCoefficient, ...]:
    """Extract interpretable coefficients from a fitted standardizing pipeline."""

    if "scale" not in estimator.named_steps or "classifier" not in estimator.named_steps:
        raise ValueError("Expected a pipeline with 'scale' and 'classifier' steps.")
    scaler = estimator.named_steps["scale"]
    classifier = estimator.named_steps["classifier"]
    coefficient = np.asarray(classifier.coef_, dtype=float)
    if coefficient.shape[0] != 1:
        raise ValueError("Only binary logistic estimators are supported.")
    standardized = coefficient[0]
    names = tuple(str(name) for name in feature_names)
    if len(names) != standardized.size:
        raise ValueError("feature_names length does not match the fitted coefficient count.")
    scale = np.asarray(scaler.scale_, dtype=float)
    scale = np.where(scale == 0.0, 1.0, scale)
    raw = standardized / scale
    rows = [
        LogisticCoefficient(
            feature=name,
            standardized_coefficient=float(value),
            raw_unit_coefficient=float(raw_value),
            odds_ratio_per_standard_deviation=float(math.exp(value)),
        )
        for name, value, raw_value in zip(names, standardized, raw, strict=True)
    ]
    if sort_by_magnitude:
        rows.sort(key=lambda row: abs(row.standardized_coefficient), reverse=True)
    return tuple(rows)


def raw_scale_intercept(estimator: Pipeline) -> float:
    """Return the intercept corresponding to unstandardized input features."""

    scaler = estimator.named_steps["scale"]
    classifier = estimator.named_steps["classifier"]
    intercept = float(np.asarray(classifier.intercept_, dtype=float)[0])
    if not getattr(scaler, "with_mean", False):
        return intercept
    coefficient = np.asarray(classifier.coef_, dtype=float)[0]
    scale = np.where(np.asarray(scaler.scale_, dtype=float) == 0.0, 1.0, scaler.scale_)
    mean = np.asarray(scaler.mean_, dtype=float)
    return float(intercept - np.dot(coefficient / scale, mean))
