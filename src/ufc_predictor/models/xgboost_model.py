"""Optional, explicitly trained XGBoost models.

XGBoost is imported lazily so the logistic baseline and evaluation utilities
remain usable when the optional dependency is not installed.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.metrics import log_loss

DevicePreference = Literal["auto", "cpu", "cuda"]
MAX_XGBOOST_CANDIDATES = 16


@dataclass(frozen=True, slots=True)
class XGBoostCandidate:
    """A deliberately bounded tree configuration."""

    name: str
    max_depth: int
    learning_rate: float
    min_child_weight: float
    subsample: float
    colsample_bytree: float
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    gamma: float = 0.0
    n_estimators: int = 2_000
    max_bin: int = 256

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Candidate name must not be empty.")
        if not 2 <= self.max_depth <= 10:
            raise ValueError("max_depth must be in [2, 10].")
        if not 0.005 <= self.learning_rate <= 0.3:
            raise ValueError("learning_rate must be in [0.005, 0.3].")
        if not 0.0 <= self.min_child_weight <= 50.0:
            raise ValueError("min_child_weight must be in [0, 50].")
        if not 0.5 <= self.subsample <= 1.0:
            raise ValueError("subsample must be in [0.5, 1].")
        if not 0.5 <= self.colsample_bytree <= 1.0:
            raise ValueError("colsample_bytree must be in [0.5, 1].")
        if not 0.0 <= self.reg_alpha <= 100.0:
            raise ValueError("reg_alpha must be in [0, 100].")
        if not 0.0 <= self.reg_lambda <= 100.0:
            raise ValueError("reg_lambda must be in [0, 100].")
        if not 0.0 <= self.gamma <= 20.0:
            raise ValueError("gamma must be in [0, 20].")
        if not 50 <= self.n_estimators <= 5_000:
            raise ValueError("n_estimators must be in [50, 5,000].")
        if not 32 <= self.max_bin <= 2_048:
            raise ValueError("max_bin must be in [32, 2,048].")


DEFAULT_XGBOOST_CANDIDATES: tuple[XGBoostCandidate, ...] = (
    XGBoostCandidate("depth2_conservative", 2, 0.03, 5.0, 0.9, 0.9, 0.0, 5.0),
    XGBoostCandidate("depth3_balanced", 3, 0.03, 3.0, 0.9, 0.9, 0.0, 3.0),
    XGBoostCandidate("depth3_regularized", 3, 0.05, 5.0, 0.8, 0.8, 0.1, 8.0),
    XGBoostCandidate("depth4_balanced", 4, 0.025, 4.0, 0.85, 0.85, 0.0, 5.0),
    XGBoostCandidate("depth5_slow", 5, 0.015, 6.0, 0.8, 0.8, 0.25, 10.0),
)


@dataclass(frozen=True, slots=True)
class CudaProbeResult:
    """Result of an explicit CUDA capability probe."""

    available: bool
    xgboost_version: str
    reason: str


@dataclass(frozen=True, slots=True)
class XGBoostFitResult:
    estimator: Any
    candidate: XGBoostCandidate
    validation_log_loss: float
    device_used: Literal["cpu", "cuda"]
    cuda_probe: CudaProbeResult | None
    best_iteration: int | None


@dataclass(frozen=True, slots=True)
class XGBoostTrial:
    candidate: XGBoostCandidate
    validation_log_loss: float | None
    device_used: str | None
    best_iteration: int | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class XGBoostSearchResult:
    estimator: Any
    candidate: XGBoostCandidate
    validation_log_loss: float
    device_used: Literal["cpu", "cuda"]
    cuda_probe: CudaProbeResult | None
    trials: tuple[XGBoostTrial, ...]


def _load_xgboost() -> Any:
    try:
        import xgboost
    except ImportError as exc:
        raise ImportError(
            "XGBoost is optional. Install the project's XGBoost extra before "
            "calling XGBoost training APIs."
        ) from exc
    return xgboost


def _version_tuple(version: str) -> tuple[int, int]:
    parts: list[int] = []
    for token in version.split(".")[:2]:
        digits = "".join(character for character in token if character.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 2:
        parts.append(0)
    return parts[0], parts[1]


def _device_parameters(xgboost: Any, device: Literal["cpu", "cuda"]) -> dict[str, Any]:
    if device == "cpu":
        return (
            {"tree_method": "hist", "device": "cpu"}
            if _version_tuple(xgboost.__version__) >= (2, 0)
            else {"tree_method": "hist"}
        )
    if _version_tuple(xgboost.__version__) >= (2, 0):
        return {"tree_method": "hist", "device": "cuda"}
    return {"tree_method": "gpu_hist", "predictor": "gpu_predictor"}


def _config_uses_cuda(estimator: Any) -> bool:
    try:
        config = json.loads(estimator.get_booster().save_config())
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
        return False

    def walk(value: Any) -> bool:
        if isinstance(value, dict):
            for key, nested in value.items():
                if key == "device" and str(nested).lower().startswith("cuda"):
                    return True
                if key == "gpu_id":
                    try:
                        if int(nested) >= 0:
                            return True
                    except (TypeError, ValueError):
                        pass
                if walk(nested):
                    return True
        elif isinstance(value, list):
            return any(walk(item) for item in value)
        return False

    return walk(config)


def probe_xgboost_cuda(*, random_state: int = 42) -> CudaProbeResult:
    """Explicitly fit a one-tree probe and verify the booster stayed on CUDA."""

    xgboost = _load_xgboost()
    features = np.asarray(
        (
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.2, 0.8),
            (0.8, 0.2),
            (0.1, 0.3),
            (0.9, 0.7),
        ),
        dtype=np.float32,
    )
    target = np.asarray((0, 0, 0, 1, 0, 1, 0, 1), dtype=np.int8)
    parameters = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": 1,
        "max_depth": 1,
        "learning_rate": 0.3,
        "random_state": random_state,
        "n_jobs": 1,
        "verbosity": 0,
        **_device_parameters(xgboost, "cuda"),
    }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator = xgboost.XGBClassifier(**parameters)
            estimator.fit(features, target, verbose=False)
        if not _config_uses_cuda(estimator):
            return CudaProbeResult(
                available=False,
                xgboost_version=str(xgboost.__version__),
                reason="The probe completed but XGBoost configured a CPU device.",
            )
    except Exception as exc:  # The optional backend raises version-specific errors.
        return CudaProbeResult(
            available=False,
            xgboost_version=str(xgboost.__version__),
            reason=f"{type(exc).__name__}: {exc}",
        )
    return CudaProbeResult(
        available=True,
        xgboost_version=str(xgboost.__version__),
        reason="CUDA booster probe succeeded.",
    )


def _resolve_device(
    preference: DevicePreference,
    *,
    allow_cpu_fallback: bool,
    probe: CudaProbeResult | None,
    random_state: int,
) -> tuple[Literal["cpu", "cuda"], CudaProbeResult | None]:
    if preference not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'.")
    if preference == "cpu":
        return "cpu", probe
    result = probe or probe_xgboost_cuda(random_state=random_state)
    if result.available:
        return "cuda", result
    if preference == "cuda" and not allow_cpu_fallback:
        raise RuntimeError(f"CUDA was requested but unavailable: {result.reason}")
    return "cpu", result


def _validate_binary_target(values: Any, *, name: str, require_both_classes: bool) -> np.ndarray:
    target = np.asarray(values).reshape(-1)
    if target.size == 0 or not np.all(np.isin(target, (0, 1))):
        raise ValueError(f"{name} must be a non-empty binary target.")
    target = target.astype(np.int8, copy=False)
    if require_both_classes and np.unique(target).size != 2:
        raise ValueError(f"{name} must contain both classes.")
    return target


def _fit_with_early_stopping(
    xgboost: Any,
    parameters: dict[str, Any],
    x_train: Any,
    y_train: np.ndarray,
    x_validation: Any,
    y_validation: np.ndarray,
    early_stopping_rounds: int,
) -> Any:
    if _version_tuple(xgboost.__version__) >= (1, 6):
        estimator = xgboost.XGBClassifier(**parameters, early_stopping_rounds=early_stopping_rounds)
        estimator.fit(
            x_train,
            y_train,
            eval_set=((x_validation, y_validation),),
            verbose=False,
        )
        return estimator

    estimator = xgboost.XGBClassifier(**parameters)
    estimator.fit(
        x_train,
        y_train,
        eval_set=((x_validation, y_validation),),
        early_stopping_rounds=early_stopping_rounds,
        verbose=False,
    )
    return estimator


def fit_xgboost_candidate(
    x_train: Any,
    y_train: Any,
    x_validation: Any,
    y_validation: Any,
    candidate: XGBoostCandidate,
    *,
    device: DevicePreference = "auto",
    allow_cpu_fallback: bool = True,
    cuda_probe: CudaProbeResult | None = None,
    early_stopping_rounds: int = 75,
    random_state: int = 42,
    n_jobs: int = 1,
) -> XGBoostFitResult:
    """Fit one XGBoost candidate with validation early stopping."""

    if not 5 <= early_stopping_rounds <= 500:
        raise ValueError("early_stopping_rounds must be in [5, 500].")
    if not 1 <= n_jobs <= 64:
        raise ValueError("n_jobs must be in [1, 64].")
    train_target = _validate_binary_target(y_train, name="y_train", require_both_classes=True)
    validation_target = _validate_binary_target(
        y_validation,
        name="y_validation",
        require_both_classes=False,
    )
    xgboost = _load_xgboost()
    selected_device, probe = _resolve_device(
        device,
        allow_cpu_fallback=allow_cpu_fallback,
        probe=cuda_probe,
        random_state=random_state,
    )
    parameters: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": candidate.n_estimators,
        "max_depth": candidate.max_depth,
        "learning_rate": candidate.learning_rate,
        "min_child_weight": candidate.min_child_weight,
        "subsample": candidate.subsample,
        "colsample_bytree": candidate.colsample_bytree,
        "reg_alpha": candidate.reg_alpha,
        "reg_lambda": candidate.reg_lambda,
        "gamma": candidate.gamma,
        "max_bin": candidate.max_bin,
        "random_state": random_state,
        "n_jobs": n_jobs,
        "verbosity": 0,
        **_device_parameters(xgboost, selected_device),
    }
    estimator = _fit_with_early_stopping(
        xgboost,
        parameters,
        x_train,
        train_target,
        x_validation,
        validation_target,
        early_stopping_rounds,
    )
    probability = np.asarray(estimator.predict_proba(x_validation), dtype=float)[:, 1]
    score = float(log_loss(validation_target, probability, labels=(0, 1)))
    best_iteration_value = getattr(estimator, "best_iteration", None)
    best_iteration = int(best_iteration_value) if best_iteration_value is not None else None
    return XGBoostFitResult(
        estimator=estimator,
        candidate=candidate,
        validation_log_loss=score,
        device_used=selected_device,
        cuda_probe=probe,
        best_iteration=best_iteration,
    )


def search_xgboost_candidates(
    x_train: Any,
    y_train: Any,
    x_validation: Any,
    y_validation: Any,
    *,
    candidates: Sequence[XGBoostCandidate] = DEFAULT_XGBOOST_CANDIDATES,
    max_candidates: int = MAX_XGBOOST_CANDIDATES,
    device: DevicePreference = "auto",
    allow_cpu_fallback: bool = True,
    early_stopping_rounds: int = 75,
    random_state: int = 42,
    n_jobs: int = 1,
) -> XGBoostSearchResult:
    """Evaluate a small deterministic candidate set by validation log loss."""

    candidate_list = tuple(candidates)
    if not candidate_list:
        raise ValueError("At least one XGBoost candidate is required.")
    if not 1 <= max_candidates <= MAX_XGBOOST_CANDIDATES:
        raise ValueError(f"max_candidates must be in [1, {MAX_XGBOOST_CANDIDATES}].")
    if len(candidate_list) > max_candidates:
        raise ValueError(
            f"Received {len(candidate_list)} candidates; the bound is {max_candidates}."
        )
    names = [candidate.name for candidate in candidate_list]
    if len(names) != len(set(names)):
        raise ValueError("XGBoost candidate names must be unique.")

    shared_probe: CudaProbeResult | None = None
    if device != "cpu":
        shared_probe = probe_xgboost_cuda(random_state=random_state)

    trials: list[XGBoostTrial] = []
    successful: list[XGBoostFitResult] = []
    for candidate in candidate_list:
        try:
            fitted = fit_xgboost_candidate(
                x_train,
                y_train,
                x_validation,
                y_validation,
                candidate,
                device=device,
                allow_cpu_fallback=allow_cpu_fallback,
                cuda_probe=shared_probe,
                early_stopping_rounds=early_stopping_rounds,
                random_state=random_state,
                n_jobs=n_jobs,
            )
            successful.append(fitted)
            trials.append(
                XGBoostTrial(
                    candidate=candidate,
                    validation_log_loss=fitted.validation_log_loss,
                    device_used=fitted.device_used,
                    best_iteration=fitted.best_iteration,
                )
            )
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            trials.append(
                XGBoostTrial(
                    candidate=candidate,
                    validation_log_loss=None,
                    device_used=None,
                    best_iteration=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    if not successful:
        details = "; ".join(
            f"{trial.candidate.name}: {trial.error or 'failed'}" for trial in trials
        )
        raise RuntimeError(f"All XGBoost candidates failed. {details}")
    selected = min(
        successful,
        key=lambda result: (
            result.validation_log_loss,
            result.candidate.max_depth,
            result.candidate.n_estimators,
        ),
    )
    return XGBoostSearchResult(
        estimator=selected.estimator,
        candidate=selected.candidate,
        validation_log_loss=selected.validation_log_loss,
        device_used=selected.device_used,
        cuda_probe=selected.cuda_probe,
        trials=tuple(trials),
    )
