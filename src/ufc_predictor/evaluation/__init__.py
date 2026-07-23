"""Prediction-only evaluation and reporting APIs."""

from .metrics import (
    BinaryMetrics,
    BinStrategy,
    ReliabilityBin,
    compute_binary_metrics,
    expected_calibration_error,
    reliability_bins,
    validate_binary_predictions,
)
from .reporting import (
    EvaluationReport,
    build_evaluation_report,
    evaluation_report_dict,
    evaluation_report_json,
    render_markdown_report,
)
from .segmentation import (
    SubgroupEvaluation,
    confidence_bands,
    evaluate_subgroups,
    experience_bands,
    matchup_experience_bands,
)

__all__ = [
    "BinStrategy",
    "BinaryMetrics",
    "EvaluationReport",
    "ReliabilityBin",
    "SubgroupEvaluation",
    "build_evaluation_report",
    "compute_binary_metrics",
    "confidence_bands",
    "evaluate_subgroups",
    "evaluation_report_dict",
    "evaluation_report_json",
    "expected_calibration_error",
    "experience_bands",
    "matchup_experience_bands",
    "reliability_bins",
    "render_markdown_report",
    "validate_binary_predictions",
]
