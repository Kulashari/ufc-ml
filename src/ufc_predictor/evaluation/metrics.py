"""Binary probability metrics and reliability diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

BinStrategy = Literal["uniform", "quantile"]


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    index: int
    lower_bound: float
    upper_bound: float
    count: int
    mean_probability: float
    observed_rate: float
    absolute_gap: float


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    sample_count: int
    positive_count: int
    prevalence: float
    log_loss: float
    brier_score: float
    roc_auc: float | None
    average_precision: float | None
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int
    threshold: float
    expected_calibration_error: float
    maximum_calibration_error: float


def validate_binary_predictions(y_true: Any, probabilities: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return validated one-dimensional target and probability arrays."""

    target = np.asarray(y_true).reshape(-1)
    prediction = np.asarray(probabilities, dtype=float).reshape(-1)
    if target.size == 0:
        raise ValueError("y_true must not be empty.")
    if target.size != prediction.size:
        raise ValueError("y_true and probabilities must have equal length.")
    if not np.all(np.isin(target, (0, 1))):
        raise ValueError("y_true must contain only 0 and 1.")
    if not np.all(np.isfinite(prediction)):
        raise ValueError("Probabilities must be finite.")
    if np.any((prediction < 0.0) | (prediction > 1.0)):
        raise ValueError("Probabilities must lie in [0, 1].")
    return target.astype(np.int8, copy=False), prediction


def _bin_edges(
    probabilities: np.ndarray,
    *,
    n_bins: int,
    strategy: BinStrategy,
) -> np.ndarray:
    if not 2 <= n_bins <= 100:
        raise ValueError("n_bins must be in [2, 100].")
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if strategy != "quantile":
        raise ValueError("strategy must be 'uniform' or 'quantile'.")

    edges = np.unique(np.quantile(probabilities, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 2:
        return np.asarray((0.0, 1.0), dtype=float)
    edges = edges.astype(float, copy=True)
    edges[0] = 0.0
    edges[-1] = 1.0
    return edges


def reliability_bins(
    y_true: Any,
    probabilities: Any,
    *,
    n_bins: int = 10,
    strategy: BinStrategy = "uniform",
) -> tuple[ReliabilityBin, ...]:
    """Aggregate a reliability curve, omitting empty bins."""

    target, prediction = validate_binary_predictions(y_true, probabilities)
    edges = _bin_edges(prediction, n_bins=n_bins, strategy=strategy)
    assignments = np.digitize(prediction, edges[1:-1], right=False)
    bins: list[ReliabilityBin] = []
    for index in range(edges.size - 1):
        mask = assignments == index
        count = int(mask.sum())
        if count == 0:
            continue
        mean_probability = float(np.mean(prediction[mask]))
        observed_rate = float(np.mean(target[mask]))
        bins.append(
            ReliabilityBin(
                index=index,
                lower_bound=float(edges[index]),
                upper_bound=float(edges[index + 1]),
                count=count,
                mean_probability=mean_probability,
                observed_rate=observed_rate,
                absolute_gap=abs(mean_probability - observed_rate),
            )
        )
    return tuple(bins)


def expected_calibration_error(
    y_true: Any,
    probabilities: Any,
    *,
    n_bins: int = 10,
    strategy: BinStrategy = "uniform",
) -> float:
    """Compute count-weighted expected calibration error."""

    target, prediction = validate_binary_predictions(y_true, probabilities)
    bins = reliability_bins(target, prediction, n_bins=n_bins, strategy=strategy)
    return float(sum(item.count * item.absolute_gap for item in bins) / target.size)


def compute_binary_metrics(
    y_true: Any,
    probabilities: Any,
    *,
    threshold: float = 0.5,
    n_bins: int = 10,
    bin_strategy: BinStrategy = "uniform",
) -> BinaryMetrics:
    """Compute discrimination, accuracy, and probability-quality metrics."""

    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must lie strictly between 0 and 1.")
    target, prediction = validate_binary_predictions(y_true, probabilities)
    labels = (prediction >= threshold).astype(np.int8)
    has_both_classes = np.unique(target).size == 2
    bins = reliability_bins(target, prediction, n_bins=n_bins, strategy=bin_strategy)
    ece = float(sum(item.count * item.absolute_gap for item in bins) / target.size)
    maximum_gap = max((item.absolute_gap for item in bins), default=0.0)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        target,
        labels,
        labels=(0, 1),
    ).ravel()
    return BinaryMetrics(
        sample_count=int(target.size),
        positive_count=int(target.sum()),
        prevalence=float(np.mean(target)),
        log_loss=float(log_loss(target, prediction, labels=(0, 1))),
        brier_score=float(brier_score_loss(target, prediction)),
        roc_auc=(float(roc_auc_score(target, prediction)) if has_both_classes else None),
        average_precision=(
            float(average_precision_score(target, prediction)) if has_both_classes else None
        ),
        accuracy=float(accuracy_score(target, labels)),
        precision=float(precision_score(target, labels, zero_division=0.0)),
        recall=float(recall_score(target, labels, zero_division=0.0)),
        f1_score=float(f1_score(target, labels, zero_division=0.0)),
        true_negative=int(true_negative),
        false_positive=int(false_positive),
        false_negative=int(false_negative),
        true_positive=int(true_positive),
        threshold=threshold,
        expected_calibration_error=ece,
        maximum_calibration_error=float(maximum_gap),
    )
