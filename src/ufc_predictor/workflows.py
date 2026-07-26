"""Explicit end-to-end workflows for training, final evaluation, and inference.

Nothing in this module runs on import.  Every fitting or held-out evaluation
operation is reached only through a direct function call (normally from the
corresponding CLI command).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from ufc_predictor.artifacts import (
    LoadedArtifacts,
    collect_runtime_versions,
    load_artifacts,
    save_artifact_version,
    update_artifact_metrics,
)
from ufc_predictor.config import AppConfig
from ufc_predictor.data import (
    DatasetBundle,
    DatasetValidationSummary,
    SplitFrames,
    assign_configured_splits,
    file_sha256,
    load_dataset_bundle,
    load_fighter_snapshots,
    split_frames,
    validate_feature_dictionary,
    validate_model_dataset,
    validate_snapshot_frame,
)
from ufc_predictor.evaluation import (
    EvaluationReport,
    build_evaluation_report,
    confidence_bands,
    evaluation_report_dict,
    experience_bands,
    render_markdown_report,
)
from ufc_predictor.exceptions import DataValidationError
from ufc_predictor.features import FeatureRegistry, ablation_feature_groups
from ufc_predictor.inference import FightPredictor, MatchupContext
from ufc_predictor.models import (
    LogisticCandidate,
    ModelScore,
    OptunaXGBoostSpace,
    XGBoostCandidate,
    apply_calibration,
    extract_logistic_coefficients,
    fit_logistic_candidate,
    fit_probability_calibrator,
    fit_xgboost_candidate,
    make_optuna_xgboost_objective,
    require_optuna,
    run_feature_ablation,
    search_logistic_candidates,
    select_model,
)

ModelFamily = Literal["logistic", "xgboost", "all"]
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True, slots=True)
class PreparedData:
    """Validated assets and immutable chronological partitions."""

    bundle: DatasetBundle
    frame: pd.DataFrame
    registry: FeatureRegistry
    splits: SplitFrames
    validation: DatasetValidationSummary


@dataclass(frozen=True, slots=True)
class TrainingResult:
    run_dir: Path
    report_json: Path
    report_markdown: Path
    selected_model: str
    validation_log_loss: float
    calibration_fit_log_loss: float
    calibration_report_json: Path
    calibration_report_markdown: Path
    selection_rationale: str


@dataclass(frozen=True, slots=True)
class FinalEvaluationResult:
    run_dir: Path
    report_json: Path
    report_markdown: Path
    test_log_loss: float
    test_rows: int


@dataclass(frozen=True, slots=True)
class _CandidateOutcome:
    name: str
    family: Literal["logistic", "xgboost"]
    estimator: Any
    candidate: LogisticCandidate | XGBoostCandidate
    validation_probabilities: np.ndarray
    report: EvaluationReport
    training_details: Mapping[str, Any]


def prepare_data(config: AppConfig) -> PreparedData:
    """Load and validate configured data without fitting or predicting."""

    bundle = load_dataset_bundle(config.data)
    frame = bundle.model.frame
    if config.data.split_column not in frame.columns:
        frame = assign_configured_splits(frame, config.data)

    registry = validate_feature_dictionary(
        bundle.feature_dictionary.frame,
        target_column=config.data.target_column,
        expected_feature_count=config.data.expected_feature_count,
    )
    validation = validate_model_dataset(
        frame,
        config.data,
        registry=registry,
        enforce_expected_counts=True,
    )
    validate_snapshot_frame(
        bundle.snapshots.frame,
        config.data,
        expected_cutoff=config.data.dataset_cutoff,
    )

    profile_ids = bundle.profiles.frame[config.data.fighter_id_column].astype(str)
    snapshot_ids = bundle.snapshots.frame[config.data.fighter_id_column].astype(str)
    if profile_ids.duplicated().any():
        raise DataValidationError("Fighter profiles contain duplicate fighter IDs")
    if set(profile_ids) != set(snapshot_ids):
        raise DataValidationError("Fighter profile and snapshot ID sets must match before training")

    return PreparedData(
        bundle=bundle,
        frame=frame,
        registry=registry,
        splits=split_frames(frame, config.data),
        validation=validation,
    )


def _positive_probability(estimator: Any, features: pd.DataFrame) -> np.ndarray:
    values = np.asarray(estimator.predict_proba(features), dtype=float)
    if values.ndim != 2 or values.shape[0] != len(features) or values.shape[1] != 2:
        raise ValueError("Binary estimator returned an invalid predict_proba matrix")
    classes = np.asarray(getattr(estimator, "classes_", (0, 1)), dtype=object)
    matches = np.flatnonzero(classes == 1)
    if matches.size != 1:
        raise ValueError("Fitted estimator does not expose exactly one positive class")
    probabilities = values[:, int(matches[0])]
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Fitted estimator returned non-finite probabilities")
    return probabilities


def _segments(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
) -> dict[str, Sequence[Any]]:
    minimum_history = frame["feature_min_prior_fights"].to_numpy(dtype=float)
    segments: dict[str, Sequence[Any]] = {
        "division": frame["division"].astype(str).tolist(),
        "both_fighters_have_ufc_history": np.where(minimum_history >= 1, "yes", "no").tolist(),
        "at_least_one_ufc_debutant": np.where(minimum_history == 0, "yes", "no").tolist(),
        "both_fighters_have_three_plus_fights": np.where(
            minimum_history >= 3, "yes", "no"
        ).tolist(),
        "experience_band": experience_bands(minimum_history).tolist(),
        "probability_band": confidence_bands(probabilities).tolist(),
    }
    if "is_title_bout" in frame.columns:
        segments["title_bout"] = np.where(
            frame["is_title_bout"].to_numpy(dtype=float) >= 0.5,
            "yes",
            "no",
        ).tolist()
    return segments


def _build_report(
    *,
    name: str,
    split_name: str,
    frame: pd.DataFrame,
    target_column: str,
    probabilities: np.ndarray,
    n_bins: int,
    notes: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> EvaluationReport:
    return build_evaluation_report(
        name,
        split_name,
        frame[target_column].to_numpy(dtype=np.int8),
        probabilities,
        segments=_segments(frame, probabilities),
        min_subgroup_samples=50,
        n_bins=n_bins,
        notes=notes,
        metadata=metadata,
    )


def _worst_subgroup_log_loss(report: EvaluationReport) -> float | None:
    values = [
        subgroup.metrics.log_loss for subgroup in report.subgroups if subgroup.metrics is not None
    ]
    return max(values) if values else None


def _subgroup_log_losses(report: EvaluationReport) -> dict[str, float]:
    return {
        f"{subgroup.segment}::{subgroup.value}": subgroup.metrics.log_loss
        for subgroup in report.subgroups
        if subgroup.metrics is not None and subgroup.segment != "probability_band"
    }


def _logistic_candidates(config: AppConfig) -> tuple[LogisticCandidate, ...]:
    candidates: list[LogisticCandidate] = []
    if config.logistic.penalty in {"l2", "both"}:
        candidates.extend(
            LogisticCandidate(
                name=f"l2_c_{value:g}",
                penalty="l2",
                c=value,
                max_iter=config.logistic.max_iter,
                tolerance=config.logistic.tolerance,
                class_weight=config.logistic.class_weight,
            )
            for value in config.logistic.c_values
        )
    if config.logistic.penalty in {"elasticnet", "both"}:
        candidates.extend(
            LogisticCandidate(
                name=f"elasticnet_c_{value:g}_l1_{ratio:g}",
                penalty="elasticnet",
                c=value,
                l1_ratio=ratio,
                max_iter=config.logistic.max_iter,
                tolerance=config.logistic.tolerance,
                class_weight=config.logistic.class_weight,
            )
            for value in config.logistic.c_values
            for ratio in config.logistic.l1_ratios
        )
    if len(candidates) > 24:
        raise ValueError("The configured logistic grid exceeds the hard limit of 24 candidates")
    return tuple(candidates)


def _fit_logistic(
    config: AppConfig,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_validation: pd.DataFrame,
    y_validation: np.ndarray,
) -> tuple[Any, LogisticCandidate, float, Mapping[str, Any]]:
    result = search_logistic_candidates(
        x_train,
        y_train,
        x_validation,
        y_validation,
        candidates=_logistic_candidates(config),
        random_state=config.project.random_seed,
        max_candidates=24,
    )
    details = {
        "selected_candidate": asdict(result.candidate),
        "trials": [asdict(trial) for trial in result.trials],
    }
    return result.estimator, result.candidate, result.validation_log_loss, details


def _configured_xgboost_candidate(config: AppConfig) -> XGBoostCandidate:
    values = config.xgboost
    return XGBoostCandidate(
        name="configured_default",
        max_depth=values.max_depth,
        learning_rate=values.learning_rate,
        min_child_weight=values.min_child_weight,
        subsample=values.subsample,
        colsample_bytree=values.colsample_bytree,
        reg_alpha=values.reg_alpha,
        reg_lambda=values.reg_lambda,
        gamma=values.gamma,
        n_estimators=values.n_estimators,
        max_bin=values.max_bin,
    )


def _optuna_space(config: AppConfig) -> OptunaXGBoostSpace:
    bounds = config.xgboost.search_space
    return OptunaXGBoostSpace(
        min_depth=bounds.max_depth[0],
        max_depth=bounds.max_depth[1],
        min_learning_rate=bounds.learning_rate[0],
        max_learning_rate=bounds.learning_rate[1],
        min_child_weight=bounds.min_child_weight[0],
        max_child_weight=bounds.min_child_weight[1],
        min_subsample=bounds.subsample[0],
        max_subsample=bounds.subsample[1],
        min_colsample=bounds.colsample_bytree[0],
        max_colsample=bounds.colsample_bytree[1],
        min_reg_alpha=bounds.reg_alpha[0],
        max_reg_alpha=bounds.reg_alpha[1],
        min_reg_lambda=bounds.reg_lambda[0],
        max_reg_lambda=bounds.reg_lambda[1],
        min_gamma=bounds.gamma[0],
        max_gamma=bounds.gamma[1],
        min_max_bin=bounds.max_bin[0],
        max_max_bin=bounds.max_bin[1],
        min_n_estimators=bounds.n_estimators[0],
        max_n_estimators=bounds.n_estimators[1],
    )


def _fit_xgboost(
    config: AppConfig,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_validation: pd.DataFrame,
    y_validation: np.ndarray,
    *,
    tune: bool,
) -> tuple[Any, XGBoostCandidate, float, Mapping[str, Any]]:
    candidate = _configured_xgboost_candidate(config)
    tuning_details: dict[str, Any] = {"tuned": False}
    if tune:
        if config.xgboost.tuning_trials < 1:
            raise ValueError("xgboost.tuning_trials must be positive when --tune is used")
        optuna = require_optuna()
        sampler = optuna.samplers.TPESampler(seed=config.project.random_seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        objective = make_optuna_xgboost_objective(
            x_train,
            y_train,
            x_validation,
            y_validation,
            space=_optuna_space(config),
            device=config.xgboost.device,
            allow_cpu_fallback=True,
            early_stopping_rounds=config.xgboost.early_stopping_rounds,
            random_state=config.project.random_seed,
            n_jobs=1,
        )
        study.optimize(
            objective,
            n_trials=config.xgboost.tuning_trials,
            show_progress_bar=False,
        )
        best = study.best_params
        candidate = XGBoostCandidate(
            name=f"optuna_best_trial_{study.best_trial.number}",
            max_depth=int(best["max_depth"]),
            learning_rate=float(best["learning_rate"]),
            min_child_weight=float(best["min_child_weight"]),
            subsample=float(best["subsample"]),
            colsample_bytree=float(best["colsample_bytree"]),
            reg_alpha=float(best["reg_alpha"]),
            reg_lambda=float(best["reg_lambda"]),
            gamma=float(best["gamma"]),
            n_estimators=int(best["n_estimators"]),
            max_bin=int(best["max_bin"]),
        )
        tuning_details = {
            "tuned": True,
            "trial_count": len(study.trials),
            "best_trial": study.best_trial.number,
            "best_validation_log_loss": float(study.best_value),
            "best_parameters": dict(best),
        }

    result = fit_xgboost_candidate(
        x_train,
        y_train,
        x_validation,
        y_validation,
        candidate,
        device=config.xgboost.device,
        allow_cpu_fallback=True,
        early_stopping_rounds=config.xgboost.early_stopping_rounds,
        random_state=config.project.random_seed,
        n_jobs=1,
    )
    details = {
        **tuning_details,
        "selected_candidate": asdict(result.candidate),
        "device_used": result.device_used,
        "best_iteration": result.best_iteration,
        "cuda_probe": (asdict(result.cuda_probe) if result.cuda_probe is not None else None),
    }
    return result.estimator, result.candidate, result.validation_log_loss, details


def _candidate_outcome(
    *,
    config: AppConfig,
    family: Literal["logistic", "xgboost"],
    estimator: Any,
    candidate: LogisticCandidate | XGBoostCandidate,
    feature_names: Sequence[str],
    validation_frame: pd.DataFrame,
    validation_log_loss: float,
    details: Mapping[str, Any],
) -> _CandidateOutcome:
    features = validation_frame.loc[:, list(feature_names)]
    probabilities = _positive_probability(estimator, features)
    report = _build_report(
        name=f"{family}:{candidate.name}",
        split_name=config.data.split_labels.validation,
        frame=validation_frame,
        target_column=config.data.target_column,
        probabilities=probabilities,
        n_bins=config.calibration.n_bins,
        metadata={"validation_log_loss_from_search": validation_log_loss},
    )
    return _CandidateOutcome(
        name=f"{family}:{candidate.name}",
        family=family,
        estimator=estimator,
        candidate=candidate,
        validation_probabilities=probabilities,
        report=report,
        training_details=details,
    )


def _select_outcome(
    outcomes: Sequence[_CandidateOutcome],
    config: AppConfig,
) -> tuple[_CandidateOutcome, str]:
    scores = [
        ModelScore(
            name=outcome.name,
            family=outcome.family,
            validation_log_loss=outcome.report.metrics.log_loss,
            validation_calibration_error=(outcome.report.metrics.expected_calibration_error),
            worst_subgroup_log_loss=_worst_subgroup_log_loss(outcome.report),
            subgroup_log_losses=_subgroup_log_losses(outcome.report),
            estimator=outcome.estimator,
        )
        for outcome in outcomes
    ]
    decision = select_model(
        scores,
        effective_tie_tolerance=config.selection.log_loss_tie_tolerance,
        max_calibration_error_regression=(config.selection.max_calibration_error_regression),
        max_subgroup_log_loss_regression=(config.selection.max_subgroup_log_loss_regression),
    )
    selected = next(outcome for outcome in outcomes if outcome.name == decision.selected.name)
    return selected, decision.rationale


def _feature_ranges(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
) -> dict[str, dict[str, float]]:
    return {
        feature: {
            "min": float(frame[feature].min()),
            "max": float(frame[feature].max()),
        }
        for feature in feature_names
    }


def _data_fingerprint(
    bundle: DatasetBundle,
    registry: FeatureRegistry,
) -> dict[str, Any]:
    digest = sha256()
    for name, value in sorted(bundle.fingerprints.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    digest.update(registry.fingerprint.encode("ascii"))
    return {
        "algorithm": "sha256",
        "combined_sha256": digest.hexdigest(),
        "files": bundle.fingerprints,
        "feature_registry_sha256": registry.fingerprint,
    }


def _feature_importance(
    outcome: _CandidateOutcome,
    feature_names: Sequence[str],
) -> Mapping[str, Any]:
    if outcome.family == "logistic":
        return {
            "kind": "logistic_coefficients",
            "values": [
                asdict(value)
                for value in extract_logistic_coefficients(
                    outcome.estimator,
                    feature_names,
                )
            ],
        }
    values = np.asarray(
        getattr(outcome.estimator, "feature_importances_", ()),
        dtype=float,
    )
    if values.size != len(feature_names):
        raise ValueError("XGBoost feature importance length does not match feature order")
    ranked_pairs = sorted(
        zip(feature_names, values, strict=True),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    ranked = [
        {"feature": feature, "importance": float(importance)}
        for feature, importance in ranked_pairs
    ]
    return {"kind": "xgboost_gain_proxy", "values": ranked}


def _write_report(
    report: EvaluationReport,
    *,
    reports_dir: Path,
    stem: str,
) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / f"{stem}.json"
    markdown_path = reports_dir / f"{stem}.md"
    json_path.write_text(
        json.dumps(
            evaluation_report_dict(report),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")
    return json_path, markdown_path


def _validated_run_id(value: str) -> str:
    if not _RUN_ID_PATTERN.fullmatch(value):
        raise ValueError(
            "run_id must start with an alphanumeric character and contain only "
            "letters, numbers, '.', '_', or '-'"
        )
    return value


def _calibration_summary(calibration: Any) -> dict[str, Any]:
    return {
        "method": calibration.method,
        "sample_count": calibration.sample_count,
        "positive_count": calibration.positive_count,
        "validation_log_loss_before": calibration.validation_log_loss_before,
        "validation_log_loss_after": calibration.validation_log_loss_after,
        "validation_brier_before": calibration.validation_brier_before,
        "validation_brier_after": calibration.validation_brier_after,
    }


def _run_ablation(
    *,
    config: AppConfig,
    selected: _CandidateOutcome,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    feature_names: tuple[str, ...],
) -> Mapping[str, Any]:
    y_train = train_frame[config.data.target_column].to_numpy(dtype=np.int8)
    y_validation = validation_frame[config.data.target_column].to_numpy(dtype=np.int8)

    def evaluate(kept: tuple[str, ...]) -> float:
        if isinstance(selected.candidate, LogisticCandidate):
            fitted = fit_logistic_candidate(
                train_frame.loc[:, list(kept)],
                y_train,
                selected.candidate,
                random_state=config.project.random_seed,
            )
            estimator = fitted.estimator
        else:
            fitted_tree = fit_xgboost_candidate(
                train_frame.loc[:, list(kept)],
                y_train,
                validation_frame.loc[:, list(kept)],
                y_validation,
                selected.candidate,
                device=config.xgboost.device,
                allow_cpu_fallback=True,
                early_stopping_rounds=config.xgboost.early_stopping_rounds,
                random_state=config.project.random_seed,
                n_jobs=1,
            )
            estimator = fitted_tree.estimator
        probabilities = _positive_probability(
            estimator,
            validation_frame.loc[:, list(kept)],
        )
        report = _build_report(
            name=f"{selected.name}:ablation",
            split_name=config.data.split_labels.validation,
            frame=validation_frame,
            target_column=config.data.target_column,
            probabilities=probabilities,
            n_bins=config.calibration.n_bins,
        )
        return report.metrics.log_loss

    report = run_feature_ablation(
        feature_names,
        ablation_feature_groups(feature_names),
        evaluate,
    )
    return asdict(report)


def train_model(
    config: AppConfig,
    *,
    model: ModelFamily,
    run_id: str | None = None,
    tune_xgboost: bool = False,
    ablation: bool = False,
) -> TrainingResult:
    """Fit and persist a model using train/validation only.

    This is intentionally the sole high-level fitting entry point.  It never
    generates predictions for the configured final-test split.
    """

    if model not in {"logistic", "xgboost", "all"}:
        raise ValueError("model must be 'logistic', 'xgboost', or 'all'")
    if model == "logistic" and tune_xgboost:
        raise ValueError("--tune-xgboost is not valid with --model logistic")

    prepared = prepare_data(config)
    feature_names = prepared.registry.names
    train_frame = prepared.splits.train
    validation_frame = prepared.splits.validation
    x_train = prepared.registry.select(train_frame)
    x_validation = prepared.registry.select(validation_frame)
    y_train = train_frame[config.data.target_column].to_numpy(dtype=np.int8)
    y_validation = validation_frame[config.data.target_column].to_numpy(dtype=np.int8)

    outcomes: list[_CandidateOutcome] = []
    if model in {"logistic", "all"}:
        estimator, logistic_candidate, score, details = _fit_logistic(
            config,
            x_train,
            y_train,
            x_validation,
            y_validation,
        )
        outcomes.append(
            _candidate_outcome(
                config=config,
                family="logistic",
                estimator=estimator,
                candidate=logistic_candidate,
                feature_names=feature_names,
                validation_frame=validation_frame.loc[
                    :, [*feature_names, config.data.target_column, "division", "is_title_bout"]
                ],
                validation_log_loss=score,
                details=details,
            )
        )
    if model in {"xgboost", "all"}:
        estimator, xgboost_candidate, score, details = _fit_xgboost(
            config,
            x_train,
            y_train,
            x_validation,
            y_validation,
            tune=tune_xgboost,
        )
        outcomes.append(
            _candidate_outcome(
                config=config,
                family="xgboost",
                estimator=estimator,
                candidate=xgboost_candidate,
                feature_names=feature_names,
                validation_frame=validation_frame.loc[
                    :, [*feature_names, config.data.target_column, "division", "is_title_bout"]
                ],
                validation_log_loss=score,
                details=details,
            )
        )

    selected, rationale = _select_outcome(outcomes, config)
    calibration = fit_probability_calibrator(
        y_validation,
        selected.validation_probabilities,
        method=config.calibration.method,
        random_state=config.project.random_seed,
    )
    calibrated_probabilities = apply_calibration(
        calibration,
        selected.validation_probabilities,
    )
    calibration_fit_report = _build_report(
        name=selected.name,
        split_name=f"{config.data.split_labels.validation}_calibration_fit",
        frame=validation_frame,
        target_column=config.data.target_column,
        probabilities=calibrated_probabilities,
        n_bins=config.calibration.n_bins,
        notes=(
            "The calibrator was fit and diagnosed on these same validation labels.",
            "These calibrated metrics are fit diagnostics, not an unbiased held-out estimate.",
            "The final test split was not evaluated by this command.",
            rationale,
        ),
        metadata={
            "calibration": _calibration_summary(calibration),
            "raw_validation_metrics": asdict(selected.report.metrics),
        },
    )

    ablation_report = (
        _run_ablation(
            config=config,
            selected=selected,
            train_frame=train_frame,
            validation_frame=validation_frame,
            feature_names=feature_names,
        )
        if ablation
        else {"status": "not_requested"}
    )
    resolved_run_id = _validated_run_id(
        run_id or f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{selected.family}"
    )
    groups = {group.value: members for group, members in prepared.registry.groups.items()}
    ranges = _feature_ranges(train_frame, feature_names)
    candidate_reports = {
        outcome.name: evaluation_report_dict(outcome.report) for outcome in outcomes
    }
    metadata = {
        "project": config.project.name,
        "selected_model": selected.name,
        "model_family": selected.family,
        "candidate": asdict(selected.candidate),
        "positive_class": 1,
        "calibrator_input": "probability",
        "selection_rationale": rationale,
        "training_rows": len(train_frame),
        "validation_rows": len(validation_frame),
        "test_rows_used": 0,
        "supported_divisions": sorted(
            value
            for value in train_frame["division"].astype(str).unique().tolist()
            if value.startswith(("M_", "W_")) or value == "CATCH"
        ),
    }
    metrics = {
        "validation": evaluation_report_dict(selected.report),
        "calibration_fit": evaluation_report_dict(calibration_fit_report),
        "candidate_validation": candidate_reports,
        "candidate_training": {
            outcome.name: dict(outcome.training_details) for outcome in outcomes
        },
        "calibration": _calibration_summary(calibration),
        "ablation": ablation_report,
        "test": {"status": "not_evaluated"},
    }
    schema = {
        "date_column": config.data.date_column,
        "split_column": config.data.split_column,
        "target_column": config.data.target_column,
        "fight_id_column": config.data.fight_id_column,
        "fighter_id_column": config.data.fighter_id_column,
        "fighter_name_column": config.data.fighter_name_column,
        "snapshot_date_column": config.data.snapshot_date_column,
        "feature_prefix": config.data.feature_prefix,
        "feature_registry_sha256": prepared.registry.fingerprint,
        "validated_model_frame_sha256": prepared.validation.frame_sha256,
        "split_summaries": [asdict(summary) for summary in prepared.validation.splits],
        "feature_ranges": ranges,
    }
    report_json, report_markdown = _write_report(
        selected.report,
        reports_dir=config.artifacts.reports_dir,
        stem=f"{resolved_run_id}-validation",
    )
    calibration_report_json, calibration_report_markdown = _write_report(
        calibration_fit_report,
        reports_dir=config.artifacts.reports_dir,
        stem=f"{resolved_run_id}-calibration-fit",
    )
    run_dir = save_artifact_version(
        config.artifacts.root_dir,
        artifact_version=resolved_run_id,
        pipeline=selected.estimator,
        calibrator=calibration.calibrator,
        metadata=metadata,
        feature_names=feature_names,
        feature_groups=groups,
        schema=schema,
        config=config.model_dump(mode="json", exclude={"project_root"}),
        data_fingerprint=_data_fingerprint(prepared.bundle, prepared.registry),
        cutoff_date=config.data.dataset_cutoff,
        metrics=metrics,
        feature_importances=_feature_importance(selected, feature_names),
        seeds={"project_random_seed": config.project.random_seed},
        package_versions=collect_runtime_versions(
            (
                "joblib",
                "numpy",
                "optuna",
                "pandas",
                "pydantic",
                "PyYAML",
                "scikit-learn",
                "typer",
                "xgboost",
            )
        ),
        repository=config.project_root,
    )
    return TrainingResult(
        run_dir=run_dir,
        report_json=report_json,
        report_markdown=report_markdown,
        selected_model=selected.name,
        validation_log_loss=selected.report.metrics.log_loss,
        calibration_fit_log_loss=calibration_fit_report.metrics.log_loss,
        calibration_report_json=calibration_report_json,
        calibration_report_markdown=calibration_report_markdown,
        selection_rationale=rationale,
    )


def _verify_artifact_data(
    artifacts: LoadedArtifacts,
    prepared: PreparedData,
    config: AppConfig,
) -> None:
    if artifacts.feature_names != prepared.registry.names:
        raise DataValidationError(
            "Artifact feature order does not match the configured feature dictionary"
        )
    expected = _data_fingerprint(prepared.bundle, prepared.registry)
    recorded_files = artifacts.data_fingerprint.get("files")
    if (
        not isinstance(recorded_files, Mapping)
        or dict(recorded_files) != expected["files"]
        or artifacts.data_fingerprint.get("combined_sha256") != expected["combined_sha256"]
        or artifacts.data_fingerprint.get("feature_registry_sha256")
        != expected["feature_registry_sha256"]
    ):
        raise DataValidationError(
            "Configured dataset fingerprints do not match the training artifact"
        )
    if artifacts.cutoff_date != config.data.dataset_cutoff:
        raise DataValidationError("Configured data cutoff does not match the training artifact")
    if artifacts.schema.get("validated_model_frame_sha256") != prepared.validation.frame_sha256:
        raise DataValidationError("Validated split assignments do not match the training artifact")


def evaluate_final(
    config: AppConfig,
    *,
    run_dir: str | Path,
    overwrite: bool = False,
) -> FinalEvaluationResult:
    """Evaluate the untouched chronological test split exactly when requested."""

    artifacts = load_artifacts(run_dir)
    prior_test = artifacts.metrics.get("test")
    if (
        isinstance(prior_test, Mapping)
        and prior_test.get("status") == "evaluated"
        and not overwrite
    ):
        raise FileExistsError(
            "final test metrics already exist; pass --overwrite explicitly "
            "before any test rows are loaded or scored"
        )
    prepared = prepare_data(config)
    _verify_artifact_data(artifacts, prepared, config)
    test_frame = prepared.splits.test
    probabilities = _positive_probability(
        artifacts.pipeline,
        test_frame.loc[:, list(artifacts.feature_names)],
    )
    if artifacts.calibrator is not None:
        probabilities = np.asarray(
            artifacts.calibrator.transform(probabilities),
            dtype=float,
        )
    report = _build_report(
        name=str(artifacts.metadata.get("selected_model", artifacts.artifact_version)),
        split_name=config.data.split_labels.test,
        frame=test_frame,
        target_column=config.data.target_column,
        probabilities=probabilities,
        n_bins=config.calibration.n_bins,
        notes=(
            "This is the explicit one-time held-out chronological test evaluation.",
            "No fitting, tuning, calibration, or feature selection used test rows.",
        ),
        metadata={
            "artifact_version": artifacts.artifact_version,
            "dataset_fingerprint": artifacts.data_fingerprint.get("combined_sha256"),
        },
    )
    metrics = dict(artifacts.metrics)
    metrics["test"] = {
        "status": "evaluated",
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "report": evaluation_report_dict(report),
    }
    report_json, report_markdown = _write_report(
        report,
        reports_dir=config.artifacts.reports_dir,
        stem=f"{artifacts.artifact_version}-test",
    )
    update_artifact_metrics(artifacts.path, metrics, overwrite=overwrite)
    return FinalEvaluationResult(
        run_dir=artifacts.path,
        report_json=report_json,
        report_markdown=report_markdown,
        test_log_loss=report.metrics.log_loss,
        test_rows=report.metrics.sample_count,
    )


def predict_fight(
    config: AppConfig,
    *,
    run_dir: str | Path,
    fighter_a: str,
    fighter_b: str,
    fighter_a_id: str | None = None,
    fighter_b_id: str | None = None,
    division: str | None = None,
    division_lbs: float | None = None,
    is_womens: bool | None = None,
    is_catch_weight: bool | None = None,
    include_features: bool = False,
) -> dict[str, Any]:
    """Load a trusted artifact and produce an order-symmetric matchup forecast.

    The predictor records one UTC timestamp for the request and uses its UTC calendar
    date to refresh date-dependent features such as age and inactivity.
    """
    artifacts = load_artifacts(run_dir)
    if artifacts.cutoff_date != config.data.dataset_cutoff:
        raise DataValidationError("Configured snapshot cutoff does not match the training artifact")
    snapshots = load_fighter_snapshots(config.data)
    validate_snapshot_frame(
        snapshots,
        config.data,
        expected_cutoff=artifacts.cutoff_date,
    )
    recorded_files = artifacts.data_fingerprint.get("files")
    recorded_snapshot = (
        recorded_files.get("fighter_snapshots") if isinstance(recorded_files, Mapping) else None
    )
    actual_snapshot = file_sha256(config.data.fighter_snapshots_path)
    if recorded_snapshot != actual_snapshot:
        raise DataValidationError("Configured fighter snapshots do not match the training artifact")
    predictor = FightPredictor.from_artifacts(
        artifacts,
        snapshots=snapshots,
        aliases=config.inference.aliases,
        limited_history_threshold=config.inference.limited_history_threshold,
        stale_after_days=config.inference.stale_after_days,
        orientation_disagreement_threshold=(config.inference.orientation_disagreement_threshold),
    )
    prediction = predictor.predict(
        fighter_a,
        fighter_b,
        fighter_a_id=fighter_a_id,
        fighter_b_id=fighter_b_id,
        context=MatchupContext(
            division=division,
            division_lbs=division_lbs,
            is_womens=is_womens,
            is_catch_weight=is_catch_weight,
        ),
    )
    result = prediction.to_dict(include_features=include_features)
    result["artifact"] = artifacts.summary()
    result["model_metadata"] = dict(artifacts.metadata)
    result["source_snapshot_sha256"] = actual_snapshot
    return result


__all__ = [
    "FinalEvaluationResult",
    "ModelFamily",
    "PreparedData",
    "TrainingResult",
    "evaluate_final",
    "predict_fight",
    "prepare_data",
    "train_model",
]
