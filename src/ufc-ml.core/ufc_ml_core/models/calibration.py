"""Post-hoc probability calibration fitted on held-out validation data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

CalibrationMethod = Literal["none", "sigmoid", "isotonic"]


class ProbabilityCalibrator(Protocol):
    """A fitted mapping from uncalibrated to calibrated probabilities."""

    def transform(self, probabilities: Any) -> np.ndarray: ...


def _probability_vector(values: Any) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float).reshape(-1)
    if probabilities.size == 0:
        raise ValueError("Probability vector must not be empty.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities must be finite.")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("Probabilities must lie in [0, 1].")
    return probabilities


def _binary_target(values: Any, *, require_both: bool = True) -> np.ndarray:
    target = np.asarray(values).reshape(-1)
    if target.size == 0 or not np.all(np.isin(target, (0, 1))):
        raise ValueError("Target must be a non-empty binary vector.")
    target = target.astype(np.int8, copy=False)
    if require_both and np.unique(target).size != 2:
        raise ValueError("Calibration data must contain both target classes.")
    return target


@dataclass(frozen=True, slots=True)
class IdentityCalibrator:
    def transform(self, probabilities: Any) -> np.ndarray:
        return _probability_vector(probabilities).copy()


@dataclass(frozen=True, slots=True)
class SigmoidCalibrator:
    estimator: LogisticRegression
    epsilon: float = 1e-6

    def transform(self, probabilities: Any) -> np.ndarray:
        base = _probability_vector(probabilities)
        clipped = np.clip(base, self.epsilon, 1.0 - self.epsilon)
        logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        return np.asarray(self.estimator.predict_proba(logit), dtype=float)[:, 1]


@dataclass(frozen=True, slots=True)
class IsotonicCalibrator:
    estimator: IsotonicRegression

    def transform(self, probabilities: Any) -> np.ndarray:
        base = _probability_vector(probabilities)
        return np.asarray(self.estimator.predict(base), dtype=float)


@dataclass(frozen=True, slots=True)
class CalibrationFit:
    """A validation-fitted calibrator with before/after diagnostics."""

    method: CalibrationMethod
    calibrator: ProbabilityCalibrator
    sample_count: int
    positive_count: int
    validation_log_loss_before: float
    validation_log_loss_after: float
    validation_brier_before: float
    validation_brier_after: float


@dataclass(frozen=True, slots=True)
class CalibratedBinaryClassifier:
    """Inference-only wrapper around an already fitted base estimator."""

    base_estimator: Any
    calibrator: ProbabilityCalibrator

    def predict_proba(self, features: Any) -> np.ndarray:
        base = np.asarray(self.base_estimator.predict_proba(features), dtype=float)
        if base.ndim != 2 or base.shape[1] != 2:
            raise ValueError("Base estimator must return two probability columns.")
        positive = self.calibrator.transform(base[:, 1])
        return np.column_stack((1.0 - positive, positive))

    def predict(self, features: Any, *, threshold: float = 0.5) -> np.ndarray:
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must lie strictly between 0 and 1.")
        return (self.predict_proba(features)[:, 1] >= threshold).astype(np.int8)


def fit_probability_calibrator(
    y_validation: Any,
    validation_probabilities: Any,
    *,
    method: CalibrationMethod = "sigmoid",
    min_isotonic_samples: int = 300,
    min_isotonic_class_count: int = 30,
    random_state: int = 42,
) -> CalibrationFit:
    """Fit calibration exclusively from held-out validation predictions.

    Do not pass training or final-test predictions.  Model selection should be
    complete before this function is called.  Isotonic calibration is rejected
    on small validation samples because it otherwise overfits easily.
    """

    target = _binary_target(y_validation)
    probability = _probability_vector(validation_probabilities)
    if target.size != probability.size:
        raise ValueError("y_validation and validation_probabilities must have equal length.")
    if method not in {"none", "sigmoid", "isotonic"}:
        raise ValueError("method must be 'none', 'sigmoid', or 'isotonic'.")

    if method == "none":
        calibrator: ProbabilityCalibrator = IdentityCalibrator()
    elif method == "sigmoid":
        epsilon = 1e-6
        clipped = np.clip(probability, epsilon, 1.0 - epsilon)
        logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        # A very weak penalty provides stable behavior across supported
        # scikit-learn versions while closely approximating unpenalized Platt
        # scaling.
        estimator = LogisticRegression(
            C=1e6,
            penalty="l2",
            solver="lbfgs",
            max_iter=2_000,
            random_state=random_state,
        )
        estimator.fit(logit, target)
        calibrator = SigmoidCalibrator(estimator=estimator, epsilon=epsilon)
    else:
        counts = np.bincount(target, minlength=2)
        if target.size < min_isotonic_samples:
            raise ValueError(
                f"Isotonic calibration requires at least {min_isotonic_samples} validation rows."
            )
        if int(np.min(counts)) < min_isotonic_class_count:
            raise ValueError("Isotonic calibration has too few examples in one target class.")
        if np.unique(probability).size < 10:
            raise ValueError("Isotonic calibration requires at least 10 distinct probabilities.")
        estimator = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        estimator.fit(probability, target)
        calibrator = IsotonicCalibrator(estimator=estimator)

    calibrated = calibrator.transform(probability)
    return CalibrationFit(
        method=method,
        calibrator=calibrator,
        sample_count=int(target.size),
        positive_count=int(target.sum()),
        validation_log_loss_before=float(log_loss(target, probability, labels=(0, 1))),
        validation_log_loss_after=float(log_loss(target, calibrated, labels=(0, 1))),
        validation_brier_before=float(brier_score_loss(target, probability)),
        validation_brier_after=float(brier_score_loss(target, calibrated)),
    )


def apply_calibration(
    calibration: CalibrationFit | ProbabilityCalibrator,
    probabilities: Any,
) -> np.ndarray:
    """Apply an already fitted calibrator without touching the base model."""

    calibrator = calibration.calibrator if isinstance(calibration, CalibrationFit) else calibration
    calibrated = np.asarray(calibrator.transform(probabilities), dtype=float)
    if not np.all(np.isfinite(calibrated)):
        raise ValueError("Calibrator returned non-finite probabilities.")
    return np.asarray(np.clip(calibrated, 0.0, 1.0), dtype=float)
