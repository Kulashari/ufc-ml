"""Bounded tuning helpers.

The Optuna integration returns an objective callable; it never creates or runs
a study on import or while constructing the objective.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from .xgboost_model import (
    DevicePreference,
    XGBoostCandidate,
    fit_xgboost_candidate,
)

CandidateT = TypeVar("CandidateT")
MAX_GENERIC_CANDIDATES = 64


class OptunaTrial(Protocol):
    """The small part of ``optuna.trial.Trial`` used by this package."""

    def suggest_int(self, name: str, low: int, high: int) -> int: ...

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        log: bool = False,
    ) -> float: ...

    def set_user_attr(self, key: str, value: Any) -> None: ...


@dataclass(frozen=True, slots=True)
class OptunaXGBoostSpace:
    """Conservative search bounds suitable for this small tabular dataset."""

    min_depth: int = 2
    max_depth: int = 6
    min_learning_rate: float = 0.01
    max_learning_rate: float = 0.12
    min_child_weight: float = 1.0
    max_child_weight: float = 15.0
    min_subsample: float = 0.65
    max_subsample: float = 1.0
    min_colsample: float = 0.65
    max_colsample: float = 1.0
    min_reg_alpha: float = 0.0
    max_reg_alpha: float = 5.0
    min_reg_lambda: float = 0.5
    max_reg_lambda: float = 20.0
    min_gamma: float = 0.0
    max_gamma: float = 5.0
    min_max_bin: int = 128
    max_max_bin: int = 512
    min_n_estimators: int = 200
    max_n_estimators: int = 2_500

    def __post_init__(self) -> None:
        if not 2 <= self.min_depth <= self.max_depth <= 10:
            raise ValueError("Depth bounds must satisfy 2 <= min <= max <= 10.")
        if not 0.005 <= self.min_learning_rate < self.max_learning_rate <= 0.3:
            raise ValueError("Learning-rate bounds must lie within [0.005, 0.3].")
        if not 0.0 <= self.min_child_weight < self.max_child_weight <= 50.0:
            raise ValueError("Child-weight bounds must lie within [0, 50].")
        if not 0.5 <= self.min_subsample <= self.max_subsample <= 1.0:
            raise ValueError("Subsample bounds must lie within [0.5, 1].")
        if not 0.5 <= self.min_colsample <= self.max_colsample <= 1.0:
            raise ValueError("Column-sample bounds must lie within [0.5, 1].")
        if not 0.0 <= self.min_reg_alpha <= self.max_reg_alpha <= 100.0:
            raise ValueError("Alpha bounds must lie within [0, 100].")
        if not 0.0 <= self.min_reg_lambda < self.max_reg_lambda <= 100.0:
            raise ValueError("Lambda bounds must lie within [0, 100].")
        if not 0.0 <= self.min_gamma <= self.max_gamma <= 20.0:
            raise ValueError("Gamma bounds must lie within [0, 20].")
        if not 32 <= self.min_max_bin <= self.max_max_bin <= 2_048:
            raise ValueError("max_bin bounds must lie within [32, 2,048].")
        if not 50 <= self.min_n_estimators <= self.max_n_estimators <= 5_000:
            raise ValueError("n_estimators bounds must lie within [50, 5,000].")

    def suggest(self, trial: OptunaTrial) -> XGBoostCandidate:
        """Convert one external Optuna trial into a validated candidate."""

        return XGBoostCandidate(
            name=f"optuna_trial_{getattr(trial, 'number', 'unknown')}",
            max_depth=trial.suggest_int("max_depth", self.min_depth, self.max_depth),
            learning_rate=trial.suggest_float(
                "learning_rate",
                self.min_learning_rate,
                self.max_learning_rate,
                log=True,
            ),
            min_child_weight=trial.suggest_float(
                "min_child_weight",
                self.min_child_weight,
                self.max_child_weight,
                log=True,
            ),
            subsample=trial.suggest_float("subsample", self.min_subsample, self.max_subsample),
            colsample_bytree=trial.suggest_float(
                "colsample_bytree",
                self.min_colsample,
                self.max_colsample,
            ),
            reg_alpha=trial.suggest_float(
                "reg_alpha",
                self.min_reg_alpha,
                self.max_reg_alpha,
            ),
            reg_lambda=trial.suggest_float(
                "reg_lambda",
                self.min_reg_lambda,
                self.max_reg_lambda,
                log=True,
            ),
            gamma=trial.suggest_float(
                "gamma",
                self.min_gamma,
                self.max_gamma,
            ),
            n_estimators=trial.suggest_int(
                "n_estimators",
                self.min_n_estimators,
                self.max_n_estimators,
            ),
            max_bin=trial.suggest_int(
                "max_bin",
                self.min_max_bin,
                self.max_max_bin,
            ),
        )


def bounded_candidates(
    candidates: Sequence[CandidateT],
    *,
    maximum: int,
    hard_limit: int = MAX_GENERIC_CANDIDATES,
) -> tuple[CandidateT, ...]:
    """Materialize a candidate collection while enforcing an explicit budget."""

    if not 1 <= maximum <= hard_limit <= MAX_GENERIC_CANDIDATES:
        raise ValueError(
            f"Candidate limits must satisfy 1 <= maximum <= hard_limit <= {MAX_GENERIC_CANDIDATES}."
        )
    materialized = tuple(candidates)
    if not materialized:
        raise ValueError("At least one candidate is required.")
    if len(materialized) > maximum:
        raise ValueError(f"Received {len(materialized)} candidates; maximum is {maximum}.")
    return materialized


def require_optuna() -> Any:
    """Lazily import Optuna when a caller intends to create a study."""

    try:
        import optuna  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "Optuna is optional. Install the project's tuning extra before "
            "creating an Optuna study."
        ) from exc
    return optuna


def make_optuna_xgboost_objective(
    x_train: Any,
    y_train: Any,
    x_validation: Any,
    y_validation: Any,
    *,
    space: OptunaXGBoostSpace | None = None,
    device: DevicePreference = "cpu",
    allow_cpu_fallback: bool = True,
    early_stopping_rounds: int = 75,
    random_state: int = 42,
    n_jobs: int = 1,
) -> Callable[[OptunaTrial], float]:
    """Build, but do not execute, an Optuna-compatible objective.

    The returned callable trains one candidate only when an external caller
    invokes it (normally through ``study.optimize``).  This function does not
    import Optuna because the objective follows a small structural protocol.
    """

    resolved_space = space or OptunaXGBoostSpace()

    def objective(trial: OptunaTrial) -> float:
        candidate = resolved_space.suggest(trial)
        fitted = fit_xgboost_candidate(
            x_train,
            y_train,
            x_validation,
            y_validation,
            candidate,
            device=device,
            allow_cpu_fallback=allow_cpu_fallback,
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        trial.set_user_attr("device_used", fitted.device_used)
        trial.set_user_attr("best_iteration", fitted.best_iteration)
        return fitted.validation_log_loss

    return objective
