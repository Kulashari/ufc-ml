"""Serializable and Markdown evaluation reports."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, cast

from .metrics import (
    BinaryMetrics,
    BinStrategy,
    ReliabilityBin,
    compute_binary_metrics,
    reliability_bins,
)
from .segmentation import SubgroupEvaluation, evaluate_subgroups


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    model_name: str
    split_name: str
    metrics: BinaryMetrics
    reliability: tuple[ReliabilityBin, ...]
    subgroups: tuple[SubgroupEvaluation, ...]
    bin_strategy: BinStrategy
    n_bins: int
    notes: tuple[str, ...] = ()
    metadata: tuple[tuple[str, Any], ...] = ()


def build_evaluation_report(
    model_name: str,
    split_name: str,
    y_true: Any,
    probabilities: Any,
    *,
    segments: Mapping[str, Sequence[Any]] | None = None,
    min_subgroup_samples: int = 50,
    threshold: float = 0.5,
    n_bins: int = 10,
    bin_strategy: BinStrategy = "uniform",
    notes: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> EvaluationReport:
    """Evaluate already-generated predictions without fitting any model."""

    if not model_name.strip() or not split_name.strip():
        raise ValueError("model_name and split_name must not be empty.")
    metrics = compute_binary_metrics(
        y_true,
        probabilities,
        threshold=threshold,
        n_bins=n_bins,
        bin_strategy=bin_strategy,
    )
    reliability = reliability_bins(y_true, probabilities, n_bins=n_bins, strategy=bin_strategy)
    subgroups = (
        evaluate_subgroups(
            y_true,
            probabilities,
            segments,
            min_samples=min_subgroup_samples,
            threshold=threshold,
            n_bins=n_bins,
            bin_strategy=bin_strategy,
        )
        if segments
        else ()
    )
    metadata_items = tuple(sorted((str(key), value) for key, value in (metadata or {}).items()))
    return EvaluationReport(
        model_name=model_name,
        split_name=split_name,
        metrics=metrics,
        reliability=reliability,
        subgroups=subgroups,
        bin_strategy=bin_strategy,
        n_bins=n_bins,
        notes=tuple(str(note) for note in notes),
        metadata=metadata_items,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def evaluation_report_dict(report: EvaluationReport) -> dict[str, Any]:
    """Convert a report into JSON-safe built-in containers."""

    payload = asdict(report)
    payload["metadata"] = dict(report.metadata)
    safe_payload = _json_safe(payload)
    if not isinstance(safe_payload, dict):
        raise TypeError("Evaluation report serialization did not produce a mapping.")
    return cast(dict[str, Any], safe_payload)


def evaluation_report_json(
    report: EvaluationReport,
    *,
    indent: int | None = 2,
) -> str:
    """Serialize a report without writing to the filesystem."""

    if indent is not None and not 0 <= indent <= 8:
        raise ValueError("indent must be None or in [0, 8].")
    return json.dumps(
        evaluation_report_dict(report),
        indent=indent,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )


def _format_metric(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_markdown_report(report: EvaluationReport) -> str:
    """Render a compact human-readable report."""

    metrics = report.metrics
    lines = [
        f"# {report.model_name} - {report.split_name}",
        "",
        "## Overall metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Rows | {metrics.sample_count:,} |",
        f"| Positive rate | {_format_metric(metrics.prevalence)} |",
        f"| Log loss | {_format_metric(metrics.log_loss)} |",
        f"| Brier score | {_format_metric(metrics.brier_score)} |",
        f"| ROC-AUC | {_format_metric(metrics.roc_auc)} |",
        f"| Average precision | {_format_metric(metrics.average_precision)} |",
        f"| Accuracy @ {metrics.threshold:.2f} | {_format_metric(metrics.accuracy)} |",
        f"| Precision @ {metrics.threshold:.2f} | {_format_metric(metrics.precision)} |",
        f"| Recall @ {metrics.threshold:.2f} | {_format_metric(metrics.recall)} |",
        f"| F1 @ {metrics.threshold:.2f} | {_format_metric(metrics.f1_score)} |",
        f"| Confusion matrix (TN/FP/FN/TP) | {metrics.true_negative}/"
        f"{metrics.false_positive}/{metrics.false_negative}/{metrics.true_positive} |",
        f"| ECE | {_format_metric(metrics.expected_calibration_error)} |",
        f"| Maximum calibration error | {_format_metric(metrics.maximum_calibration_error)} |",
        "",
        "## Reliability",
        "",
        "| Bin | Rows | Mean probability | Observed rate | Gap |",
        "|---|---:|---:|---:|---:|",
    ]
    for reliability_bin in report.reliability:
        lines.append(
            f"| [{reliability_bin.lower_bound:.2f}, "
            f"{reliability_bin.upper_bound:.2f}] | "
            f"{reliability_bin.count:,} | "
            f"{reliability_bin.mean_probability:.4f} | "
            f"{reliability_bin.observed_rate:.4f} | "
            f"{reliability_bin.absolute_gap:.4f} |"
        )

    if report.subgroups:
        lines.extend(
            (
                "",
                "## Subgroups",
                "",
                "| Segment | Value | Rows | Log loss | Brier | ROC-AUC | ECE |",
                "|---|---|---:|---:|---:|---:|---:|",
            )
        )
        for subgroup in report.subgroups:
            if subgroup.metrics is None:
                lines.append(
                    f"| {subgroup.segment} | {subgroup.value} | "
                    f"{subgroup.sample_count:,} | skipped | skipped | skipped | "
                    "skipped |"
                )
                continue
            subgroup_metrics = subgroup.metrics
            lines.append(
                f"| {subgroup.segment} | {subgroup.value} | "
                f"{subgroup.sample_count:,} | "
                f"{_format_metric(subgroup_metrics.log_loss)} | "
                f"{_format_metric(subgroup_metrics.brier_score)} | "
                f"{_format_metric(subgroup_metrics.roc_auc)} | "
                f"{_format_metric(subgroup_metrics.expected_calibration_error)} |"
            )

    if report.notes:
        lines.extend(("", "## Notes", ""))
        lines.extend(f"- {note}" for note in report.notes)
    return "\n".join(lines) + "\n"
