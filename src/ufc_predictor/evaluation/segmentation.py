"""Subgroup diagnostics for fight-model predictions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .metrics import (
    BinaryMetrics,
    BinStrategy,
    compute_binary_metrics,
    validate_binary_predictions,
)


@dataclass(frozen=True, slots=True)
class SubgroupEvaluation:
    segment: str
    value: str
    sample_count: int
    positive_count: int
    metrics: BinaryMetrics | None
    skipped_reason: str | None = None


def _display_value(value: Any) -> str:
    if value is None:
        return "<missing>"
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return "<missing>"
    return str(value)


def evaluate_subgroups(
    y_true: Any,
    probabilities: Any,
    segments: Mapping[str, Sequence[Any] | np.ndarray],
    *,
    min_samples: int = 50,
    include_small_groups: bool = True,
    threshold: float = 0.5,
    n_bins: int = 10,
    bin_strategy: BinStrategy = "uniform",
) -> tuple[SubgroupEvaluation, ...]:
    """Compute the same metrics for every value of each supplied segment."""

    target, prediction = validate_binary_predictions(y_true, probabilities)
    if not 1 <= min_samples <= target.size:
        raise ValueError("min_samples must be between 1 and the sample count.")
    if not segments:
        return ()

    results: list[SubgroupEvaluation] = []
    for segment_name, raw_values in segments.items():
        if not str(segment_name).strip():
            raise ValueError("Segment names must not be empty.")
        values = np.asarray(raw_values, dtype=object).reshape(-1)
        if values.size != target.size:
            raise ValueError(
                f"Segment {segment_name!r} has {values.size} rows; expected {target.size}."
            )
        displayed = np.asarray(tuple(_display_value(value) for value in values), dtype=object)
        for value in sorted(set(displayed.tolist())):
            mask = displayed == value
            count = int(mask.sum())
            positives = int(np.asarray(target[mask], dtype=int).sum())
            if count < min_samples:
                if include_small_groups:
                    results.append(
                        SubgroupEvaluation(
                            segment=str(segment_name),
                            value=value,
                            sample_count=count,
                            positive_count=positives,
                            metrics=None,
                            skipped_reason=(f"Fewer than min_samples={min_samples} rows."),
                        )
                    )
                continue
            metrics = compute_binary_metrics(
                target[mask],
                prediction[mask],
                threshold=threshold,
                n_bins=min(n_bins, max(2, count // 10)),
                bin_strategy=bin_strategy,
            )
            results.append(
                SubgroupEvaluation(
                    segment=str(segment_name),
                    value=value,
                    sample_count=count,
                    positive_count=positives,
                    metrics=metrics,
                )
            )
    return tuple(results)


def experience_bands(prior_fight_counts: Any) -> np.ndarray:
    """Create stable UFC-experience groups from pre-fight counts."""

    counts = np.asarray(prior_fight_counts, dtype=float).reshape(-1)
    if counts.size == 0 or not np.all(np.isfinite(counts)):
        raise ValueError("prior_fight_counts must be a non-empty finite vector.")
    if np.any(counts < 0):
        raise ValueError("prior_fight_counts cannot be negative.")
    labels = np.full(counts.size, "11_plus", dtype=object)
    labels[counts == 0] = "debut"
    labels[(counts >= 1) & (counts <= 2)] = "1_to_2"
    labels[(counts >= 3) & (counts <= 5)] = "3_to_5"
    labels[(counts >= 6) & (counts <= 10)] = "6_to_10"
    return labels


def matchup_experience_bands(
    fighter_a_prior_fights: Any,
    fighter_b_prior_fights: Any,
) -> np.ndarray:
    """Segment matchups by the less-experienced fighter's UFC history."""

    fighter_a = np.asarray(fighter_a_prior_fights, dtype=float).reshape(-1)
    fighter_b = np.asarray(fighter_b_prior_fights, dtype=float).reshape(-1)
    if fighter_a.size != fighter_b.size:
        raise ValueError("Fighter experience arrays must have equal length.")
    return experience_bands(np.minimum(fighter_a, fighter_b))


def confidence_bands(probabilities: Any) -> np.ndarray:
    """Group predictions by distance from an even matchup."""

    prediction = np.asarray(probabilities, dtype=float).reshape(-1)
    if prediction.size == 0 or not np.all(np.isfinite(prediction)):
        raise ValueError("probabilities must be a non-empty finite vector.")
    if np.any((prediction < 0.0) | (prediction > 1.0)):
        raise ValueError("probabilities must lie in [0, 1].")
    confidence = np.maximum(prediction, 1.0 - prediction)
    labels = np.full(prediction.size, "0.80_plus", dtype=object)
    labels[confidence < 0.60] = "0.50_to_0.60"
    labels[(confidence >= 0.60) & (confidence < 0.70)] = "0.60_to_0.70"
    labels[(confidence >= 0.70) & (confidence < 0.80)] = "0.70_to_0.80"
    return labels
