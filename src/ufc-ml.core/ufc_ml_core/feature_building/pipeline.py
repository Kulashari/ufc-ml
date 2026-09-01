"""End-to-end, safe raw UFCStats-to-71-feature build workflow."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml

from ufc_ml_core.config import AppConfig
from ufc_ml_core.data.loader import load_csv
from ufc_ml_core.exceptions import DataValidationError
from ufc_ml_core.feature_building.domain import (
    Bout,
    Division,
    FighterProfile,
    division_from_code,
)
from ufc_ml_core.feature_building.sources import (
    LegacyIdentityResolver,
    load_legacy_bouts,
    load_legacy_profiles,
    load_normalized_bouts,
    load_sqlite_profiles,
    merge_chronological_bouts,
)
from ufc_ml_core.feature_building.state import (
    FighterState,
    ProfileImputers,
    apply_bout,
    fighter_features,
    fit_profile_imputers,
)
from ufc_ml_core.features.registry import FeatureRegistry
from ufc_ml_core.inference.predictor import DivisionSpec, build_matchup_features

FEATURE_BUILDER_VERSION = "0.1.0"

# One pre-label legacy fight uses the first of two Bruno Silva UFCStats pages.
# All other duplicate-name occurrences are supplied by the model's ID-only
# metadata bridge.  Keeping this as a named map makes the non-derivable legacy
# identity decision reviewable rather than hidden in a heuristic.
DEFAULT_LEGACY_IDENTITY_OVERRIDES: dict[tuple[str, str], str] = {
    ("57ff0eb2351979c4", "Bruno Silva"): "294aa73dbf37d281",
}


@dataclass(frozen=True, slots=True)
class BuildPaths:
    """Raw inputs and non-destructive candidate-output root."""

    legacy_fights_path: Path
    legacy_profiles_path: Path
    normalized_sqlite_path: Path | None
    candidate_root: Path
    reference_model_path: Path
    reference_snapshot_path: Path
    reference_profile_path: Path
    feature_dictionary_path: Path


@dataclass(frozen=True, slots=True)
class RegressionSummary:
    """Compact numerical comparison with the checked-in model data."""

    reference_rows: int
    generated_rows: int
    common_rows: int
    missing_reference_fights: int
    unexpected_generated_fights: int
    split_reassignments: int
    metadata_mismatches: int
    target_mismatches: int
    feature_exact_at_1e6: int
    feature_total: int
    max_abs_feature_error: float
    mean_abs_feature_error: float
    snapshot_common_fighters: int
    snapshot_max_abs_error: float
    snapshot_mean_abs_error: float

    @property
    def exact(self) -> bool:
        return (
            self.missing_reference_fights == 0
            and self.unexpected_generated_fights == 0
            and self.metadata_mismatches == 0
            and self.target_mismatches == 0
            and self.feature_exact_at_1e6 == self.feature_total
        )


@dataclass(frozen=True, slots=True)
class BuildResult:
    """Published candidate paths and regression evidence for a completed run."""

    run_dir: Path
    model_dataset_path: Path
    snapshots_path: Path
    profiles_path: Path
    feature_dictionary_path: Path
    manifest_path: Path
    regression_path: Path
    generated_label_rows: int
    new_label_rows: int
    baseline_rows_reused: int
    baseline_strategy: str
    generated_history_bouts: int
    through: date
    regression: RegressionSummary


def default_build_paths(config: AppConfig) -> BuildPaths:
    """Construct project-relative inputs using the existing checked-in layout."""

    root = config.project_root or Path.cwd()
    return BuildPaths(
        legacy_fights_path=(root / "data/ufc_gold_dataset_final.csv").resolve(),
        legacy_profiles_path=(root / "data/ufc_fighters_final.csv").resolve(),
        normalized_sqlite_path=(root / "data/interim/ufcstats/ufcstats.sqlite3").resolve(),
        candidate_root=(root / "data/candidates/featurebuilder").resolve(),
        reference_model_path=config.data.model_dataset_path,
        reference_snapshot_path=config.data.fighter_snapshots_path,
        reference_profile_path=config.data.fighter_profiles_path,
        feature_dictionary_path=config.data.feature_dictionary_path,
    )


def _merge_profiles(
    legacy: Iterable[FighterProfile], normalized: Iterable[FighterProfile]
) -> dict[str, FighterProfile]:
    """Keep baseline profile values stable and add IDs first seen in SQLite."""

    result = {profile.fighter_id: profile for profile in legacy}
    for profile in normalized:
        existing = result.get(profile.fighter_id)
        if existing is None:
            result[profile.fighter_id] = profile
            continue
        # A previously blank static field may be filled by a later directory
        # page, but no historical nonblank field is silently replaced with a
        # current career-profile value.
        result[profile.fighter_id] = FighterProfile(
            fighter_id=existing.fighter_id,
            fighter_name=existing.fighter_name or profile.fighter_name,
            height_inches=(
                existing.height_inches
                if existing.height_inches is not None
                else profile.height_inches
            ),
            reach_inches=(
                existing.reach_inches if existing.reach_inches is not None else profile.reach_inches
            ),
            stance=existing.stance if existing.stance is not None else profile.stance,
            dob=existing.dob if existing.dob is not None else profile.dob,
            fighter_url=existing.fighter_url or profile.fighter_url,
        )
    return result


def _model_orientation(bout: Bout) -> tuple[bool, str, str, str, str]:
    """Return A/B direction and target using the fixed hexadecimal contract."""

    first_is_a = int(bout.fight_id[-1], 16) % 2 == 0
    if first_is_a:
        return (
            True,
            bout.fighter_1_id,
            bout.fighter_1_name,
            bout.fighter_2_id,
            bout.fighter_2_name,
        )
    return (
        False,
        bout.fighter_2_id,
        bout.fighter_2_name,
        bout.fighter_1_id,
        bout.fighter_1_name,
    )


def _matchup_division(division: Division) -> DivisionSpec:
    """Adapt the raw-builder division record to the shared feature helper."""

    return DivisionSpec(
        code=division.code,
        pounds=division.pounds,
        is_womens=division.is_womens,
        is_catch_weight=division.is_catch_weight,
    )


def _split_for_date(config: AppConfig, event_date: date) -> str:
    labels = config.data.split_labels
    if event_date <= config.data.train_end:
        return labels.train
    if config.data.validation_start <= event_date <= config.data.validation_end:
        return labels.validation
    # An extended candidate intentionally remains in the held-out/test era
    # until the user deliberately changes split boundaries for a new experiment.
    return labels.test


def _model_row(
    bout: Bout,
    first_features: Mapping[str, Any],
    second_features: Mapping[str, Any],
    *,
    feature_names: Sequence[str],
    config: AppConfig,
) -> dict[str, Any]:
    first_is_a, fighter_a_id, fighter_a_name, fighter_b_id, fighter_b_name = _model_orientation(
        bout
    )
    a_features, b_features = (
        (first_features, second_features) if first_is_a else (second_features, first_features)
    )
    features = build_matchup_features(
        a_features,
        b_features,
        feature_names=feature_names,
        division=_matchup_division(bout.division),
    )
    target = int((bout.winner_id == fighter_a_id) if bout.winner_id is not None else False)
    return {
        "fight_id": bout.fight_id,
        "event_date": bout.event_date.isoformat(),
        "split": _split_for_date(config, bout.event_date),
        "fighter_a_id": fighter_a_id,
        "fighter_a_name": fighter_a_name,
        "fighter_b_id": fighter_b_id,
        "fighter_b_name": fighter_b_name,
        "division": bout.division.code,
        "raw_weight_class": bout.raw_weight_class,
        "is_title_bout": bout.is_title_bout,
        "is_interim_title": bout.is_interim_title,
        "is_tournament_final": bout.is_tournament_final,
        "is_superfight": bout.is_superfight,
        "scheduled_rounds": bout.scheduled_rounds,
        "scheduled_duration_sec": bout.scheduled_duration_sec,
        **features,
        "target_a_win": target,
    }


def _clean_profile_row(profile: FighterProfile) -> dict[str, Any]:
    stance = profile.stance or "Unknown"
    return {
        "fighter_id": profile.fighter_id,
        "fighter_name": profile.fighter_name,
        "display_name": profile.fighter_name,
        "height_inches": profile.height_inches,
        "reach_inches": profile.reach_inches,
        "stance": stance,
        "dob": profile.dob.isoformat() if profile.dob is not None else None,
        "fighter_url": profile.fighter_url,
        "height_missing": int(profile.height_inches is None),
        "reach_missing": int(profile.reach_inches is None),
        "stance_missing": int(profile.stance is None),
        "dob_missing": int(profile.dob is None),
    }


def _snapshot_rows(
    states: Mapping[str, FighterState],
    profiles: Mapping[str, FighterProfile],
    *,
    as_of_date: date,
    imputers: ProfileImputers,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fighter_id, profile in sorted(
        profiles.items(), key=lambda item: (item[1].fighter_name, item[0])
    ):
        state = states[fighter_id]
        division = division_from_code(state.last_division)
        values = fighter_features(
            state,
            profile,
            event_date=as_of_date,
            division=division,
            imputers=imputers,
        )
        last_fight_date = state.last_fight_date
        active = last_fight_date is not None and (as_of_date - last_fight_date).days <= 365 * 3
        rows.append(
            {
                "fighter_id": fighter_id,
                "fighter_name": profile.fighter_name,
                "display_name": profile.fighter_name,
                "as_of_date": as_of_date.isoformat(),
                "last_fight_date": (
                    last_fight_date.isoformat() if last_fight_date is not None else None
                ),
                "last_division": state.last_division,
                "stance": values.pop("stance"),
                "dob": values.pop("dob"),
                "has_ufc_history": int(state.prior_fights > 0),
                "is_recently_active_3y": int(active),
                **values,
            }
        )
    return rows


def _snapshot_date(value: object) -> date | None:
    if value is None or _is_missing(value) or not str(value).strip():
        return None
    parsed = pd.to_datetime(str(value), errors="coerce")
    if not isinstance(parsed, pd.Timestamp) or _is_missing(parsed):
        return None
    return parsed.date()


def _is_missing(value: object) -> bool:
    """Return whether one scalar pandas-backed source value is absent."""

    return bool(pd.isna(cast(Any, value)))


def _snapshot_float(value: object, *, field: str) -> float:
    """Coerce one persisted snapshot scalar with a focused validation error."""

    try:
        return float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"Snapshot {field} must be numeric, got {value!r}") from exc


def _bootstrap_states_from_snapshot(
    states: Mapping[str, FighterState],
    snapshot_path: Path,
) -> int:
    """Overlay exact trusted cutoff state before processing newer normalized fights.

    Raw counters and the last-five observations remain reconstructed from the
    legacy corpus.  The snapshot supplies ratings/averages whose full
    historical per-division vectors were not persisted in the original assets.
    """

    frame = load_csv(
        snapshot_path,
        required_columns=(
            "fighter_id",
            "last_fight_date",
            "last_division",
            "feature_prior_fights",
            "feature_career_minutes",
            "feature_current_streak",
            "feature_elo_global",
            "feature_elo_division",
            "feature_division_elo_fights",
            "feature_avg_opponent_elo",
        ),
    ).set_index("fighter_id")
    bootstrapped = 0
    for fighter_id, state in states.items():
        if fighter_id not in frame.index:
            continue
        selected = frame.loc[fighter_id]
        if isinstance(selected, pd.DataFrame):
            raise DataValidationError(f"Snapshot has duplicate fighter ID {fighter_id}")
        row = selected
        prior_fights = round(
            _snapshot_float(row["feature_prior_fights"], field="feature_prior_fights")
        )
        career_seconds = round(
            _snapshot_float(row["feature_career_minutes"], field="feature_career_minutes") * 60.0
        )
        last_division = str(row["last_division"] or "UNKNOWN").strip() or "UNKNOWN"
        state.prior_fights = prior_fights
        state.career_seconds = career_seconds
        state.current_streak = _snapshot_float(
            row["feature_current_streak"], field="feature_current_streak"
        )
        state.elo_global = _snapshot_float(row["feature_elo_global"], field="feature_elo_global")
        state.last_division = last_division
        state.last_fight_date = _snapshot_date(row["last_fight_date"])
        state.elo_by_division[last_division] = _snapshot_float(
            row["feature_elo_division"], field="feature_elo_division"
        )
        state.division_elo_fights[last_division] = round(
            _snapshot_float(row["feature_division_elo_fights"], field="feature_division_elo_fights")
        )
        state.opponent_elo_count = prior_fights
        state.opponent_elo_sum = (
            _snapshot_float(row["feature_avg_opponent_elo"], field="feature_avg_opponent_elo")
            * (prior_fights + 2)
            - 3000.0
        )
        bootstrapped += 1
    return bootstrapped


def _numeric_regression(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    columns: Sequence[str],
) -> tuple[int, int, float, float]:
    if not columns or expected.empty or actual.empty:
        return (0, 0, 0.0, 0.0)
    left = expected.loc[:, list(columns)].to_numpy(dtype=float)
    right = actual.loc[:, list(columns)].to_numpy(dtype=float)
    errors = np.abs(left - right)
    exact = int(np.isclose(errors, 0.0, atol=5e-7, rtol=0.0).sum())
    return (exact, int(errors.size), float(errors.max()), float(errors.mean()))


def regression_summary(
    generated_model: pd.DataFrame,
    generated_snapshots: pd.DataFrame,
    *,
    reference_model_path: Path,
    reference_snapshot_path: Path,
    feature_names: Sequence[str],
) -> RegressionSummary:
    """Compare candidate values by stable IDs rather than fragile CSV row order."""

    reference_model = load_csv(reference_model_path, required_columns=("fight_id", *feature_names))
    common_ids = sorted(set(reference_model["fight_id"]).intersection(generated_model["fight_id"]))
    expected = reference_model.set_index("fight_id").loc[common_ids]
    actual = generated_model.set_index("fight_id").loc[common_ids]
    split_reassignments = int(expected["split"].astype(str).ne(actual["split"].astype(str)).sum())
    metadata_columns = (
        "event_date",
        "fighter_a_id",
        "fighter_a_name",
        "fighter_b_id",
        "fighter_b_name",
        "division",
        "raw_weight_class",
        "is_title_bout",
        "is_interim_title",
        "is_tournament_final",
        "is_superfight",
        "scheduled_rounds",
        "scheduled_duration_sec",
    )
    available_metadata = [column for column in metadata_columns if column in expected.columns]
    metadata_mismatches = int(
        (~expected.loc[:, available_metadata].eq(actual.loc[:, available_metadata]))
        .any(axis=1)
        .sum()
    )
    target_mismatches = int(
        expected["target_a_win"].astype(int).ne(actual["target_a_win"].astype(int)).sum()
    )
    exact, total, maximum, mean = _numeric_regression(expected, actual, feature_names)

    reference_snapshots = load_csv(reference_snapshot_path, required_columns=("fighter_id",))
    snapshot_features = [
        column
        for column in generated_snapshots.columns
        if column.startswith("feature_") and column in reference_snapshots.columns
    ]
    common_fighters = sorted(
        set(reference_snapshots["fighter_id"]).intersection(generated_snapshots["fighter_id"])
    )
    expected_snapshot = reference_snapshots.set_index("fighter_id").loc[common_fighters]
    actual_snapshot = generated_snapshots.set_index("fighter_id").loc[common_fighters]
    _, _, snapshot_max, snapshot_mean = _numeric_regression(
        expected_snapshot, actual_snapshot, snapshot_features
    )
    return RegressionSummary(
        reference_rows=len(reference_model),
        generated_rows=len(generated_model),
        common_rows=len(common_ids),
        missing_reference_fights=len(
            set(reference_model["fight_id"]) - set(generated_model["fight_id"])
        ),
        unexpected_generated_fights=len(
            set(generated_model["fight_id"]) - set(reference_model["fight_id"])
        ),
        split_reassignments=split_reassignments,
        metadata_mismatches=metadata_mismatches,
        target_mismatches=target_mismatches,
        feature_exact_at_1e6=exact,
        feature_total=total,
        max_abs_feature_error=maximum,
        mean_abs_feature_error=mean,
        snapshot_common_fighters=len(common_fighters),
        snapshot_max_abs_error=snapshot_max,
        snapshot_mean_abs_error=snapshot_mean,
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.6f")
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _suggested_config(
    config: AppConfig,
    *,
    source_config_path: Path | None,
    output_model_path: Path,
    output_snapshot_path: Path,
    output_profiles_path: Path,
    output_dictionary_path: Path,
    row_count: int,
    cutoff: date,
) -> dict[str, Any]:
    """Create a review-only config with paths/count/cutoff updated for the candidate."""

    source_path = source_config_path or (config.project_root or Path.cwd()) / "configs/default.yaml"
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise DataValidationError(f"Candidate source config is not a mapping: {source_path}")
    raw_data = raw.get("data")
    if not isinstance(raw_data, dict):
        raise DataValidationError(f"Candidate source config has no data mapping: {source_path}")
    root = config.project_root or Path.cwd()

    def relative(path: Path) -> str:
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            return str(path)

    raw_data.update(
        {
            "model_dataset_path": relative(output_model_path),
            "fighter_snapshots_path": relative(output_snapshot_path),
            "fighter_profiles_path": relative(output_profiles_path),
            "feature_dictionary_path": relative(output_dictionary_path),
            "expected_fight_count": row_count,
            "dataset_cutoff": cutoff.isoformat(),
        }
    )
    return cast(dict[str, Any], raw)


def build_feature_candidate(
    config: AppConfig,
    *,
    paths: BuildPaths | None = None,
    config_template_path: Path | None = None,
    through: date | None = None,
    run_id: str | None = None,
    bootstrap_baseline: bool = True,
) -> BuildResult:
    """Build a fresh candidate data bundle without touching ``data/processed``.

    It starts from all local legacy raw bouts, adds any normalized fights absent
    from that corpus, emits each label before applying the bout update, and
    records a full comparison with the checked-in pre-cutoff dataset.
    """

    paths = paths or default_build_paths(config)
    legacy_profiles = load_legacy_profiles(paths.legacy_profiles_path)
    resolver = LegacyIdentityResolver(
        legacy_profiles,
        reference_model_path=paths.reference_model_path,
        overrides=DEFAULT_LEGACY_IDENTITY_OVERRIDES,
    )
    legacy_bouts = load_legacy_bouts(paths.legacy_fights_path, resolver)
    normalized_profiles = (
        load_sqlite_profiles(paths.normalized_sqlite_path)
        if paths.normalized_sqlite_path is not None
        else []
    )
    normalized_bouts = (
        load_normalized_bouts(paths.normalized_sqlite_path, starting_order=len(legacy_bouts))
        if paths.normalized_sqlite_path is not None
        else []
    )
    profiles = _merge_profiles(legacy_profiles, normalized_profiles)
    bouts = merge_chronological_bouts(legacy_bouts, normalized_bouts)
    if through is not None:
        bouts = [bout for bout in bouts if bout.event_date <= through]
    if not bouts:
        raise DataValidationError("No completed bouts are available within the requested range")
    latest_available = max(bout.event_date for bout in bouts)
    if bootstrap_baseline and latest_available < config.data.dataset_cutoff:
        raise DataValidationError(
            "--bootstrap-baseline requires history through at least the configured dataset "
            f"cutoff ({config.data.dataset_cutoff.isoformat()}); use "
            "--reconstruct-baseline for an earlier audit."
        )
    missing_profiles = sorted(
        {
            fighter_id
            for bout in bouts
            for fighter_id in (bout.fighter_1_id, bout.fighter_2_id)
            if fighter_id not in profiles
        }
    )
    if missing_profiles:
        raise DataValidationError(
            f"Cannot construct snapshots; fights refer to missing profiles: {missing_profiles[:10]}"
        )
    imputers = fit_profile_imputers(
        legacy_bouts,
        profiles,
        train_end=config.data.train_end,
    )
    dictionary = load_csv(paths.feature_dictionary_path, required_columns=("column", "role"))
    registry = FeatureRegistry.from_dictionary(dictionary)
    feature_names = registry.names
    states = {fighter_id: FighterState() for fighter_id in profiles}
    rows: list[dict[str, Any]] = []
    reference_states: dict[str, FighterState] | None = None
    baseline_bootstrapped = False
    bootstrapped_fighters = 0
    for bout in bouts:
        if reference_states is None and bout.event_date > config.data.dataset_cutoff:
            # The checked-in snapshot includes every fight on the cutoff date.
            # Freeze this state before newer SQLite history mutates it so the
            # regression report compares like-for-like point-in-time assets.
            reference_states = deepcopy(states)
        if (
            bootstrap_baseline
            and not baseline_bootstrapped
            and bout.event_date > config.data.dataset_cutoff
        ):
            bootstrapped_fighters = _bootstrap_states_from_snapshot(
                states, paths.reference_snapshot_path
            )
            baseline_bootstrapped = True
        first = fighter_features(
            states[bout.fighter_1_id],
            profiles[bout.fighter_1_id],
            event_date=bout.event_date,
            division=bout.division,
            imputers=imputers,
        )
        second = fighter_features(
            states[bout.fighter_2_id],
            profiles[bout.fighter_2_id],
            event_date=bout.event_date,
            division=bout.division,
            imputers=imputers,
        )
        if bout.is_label_eligible:
            rows.append(
                _model_row(
                    bout,
                    first,
                    second,
                    feature_names=feature_names,
                    config=config,
                )
            )
        apply_bout(states, bout)
    latest = latest_available
    if reference_states is None:
        reference_states = deepcopy(states)
    if bootstrap_baseline and not baseline_bootstrapped:
        bootstrapped_fighters = _bootstrap_states_from_snapshot(
            states, paths.reference_snapshot_path
        )
        baseline_bootstrapped = True
    model_columns = dictionary.loc[
        dictionary["role"].isin(["metadata", "model_feature", "target"]), "column"
    ].tolist()
    reconstructed_model_frame = pd.DataFrame(rows).loc[:, model_columns]
    reference_model = load_csv(paths.reference_model_path, required_columns=model_columns)
    new_rows = reconstructed_model_frame.loc[
        pd.to_datetime(reconstructed_model_frame["event_date"]).dt.date > config.data.dataset_cutoff
    ].copy()
    if bootstrap_baseline:
        baseline_rows = reference_model.loc[:, model_columns].copy()
        baseline_dates = pd.to_datetime(baseline_rows["event_date"], errors="raise").dt.date
        baseline_rows["split"] = [
            _split_for_date(config, event_date) for event_date in baseline_dates
        ]
        model_frame = pd.concat(
            [baseline_rows, new_rows],
            ignore_index=True,
        )
    else:
        model_frame = reconstructed_model_frame
    snapshots = pd.DataFrame(_snapshot_rows(states, profiles, as_of_date=latest, imputers=imputers))
    regression_snapshots = pd.DataFrame(
        _snapshot_rows(
            reference_states if reference_states is not None else states,
            profiles,
            as_of_date=min(latest, config.data.dataset_cutoff),
            imputers=imputers,
        )
    )
    profile_frame = pd.DataFrame(
        [
            _clean_profile_row(profile)
            for profile in sorted(profiles.values(), key=lambda item: item.fighter_id)
        ]
    )
    regression = regression_summary(
        reconstructed_model_frame,
        regression_snapshots,
        reference_model_path=paths.reference_model_path,
        reference_snapshot_path=paths.reference_snapshot_path,
        feature_names=feature_names,
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    identifier = run_id or timestamp
    if not identifier or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in identifier
    ):
        raise ValueError("run_id may contain only letters, digits, hyphens, and underscores")
    run_dir = paths.candidate_root / f"run-{identifier}"
    if run_dir.exists():
        raise FileExistsError(f"Candidate output already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    model_path = run_dir / "ufc_model_ready.csv"
    snapshots_path = run_dir / "ufc_fighter_latest_features.csv"
    profiles_path = run_dir / "ufc_fighter_profiles_clean.csv"
    dictionary_path = run_dir / "ufc_model_feature_dictionary.csv"
    regression_path = run_dir / "regression.json"
    manifest_path = run_dir / "manifest.json"
    _write_csv(model_path, model_frame)
    _write_csv(snapshots_path, snapshots)
    _write_csv(profiles_path, profile_frame)
    shutil.copy2(paths.feature_dictionary_path, dictionary_path)
    _write_json(regression_path, asdict(regression) | {"exact": regression.exact})
    suggested_config = _suggested_config(
        config,
        source_config_path=config_template_path,
        output_model_path=model_path,
        output_snapshot_path=snapshots_path,
        output_profiles_path=profiles_path,
        output_dictionary_path=dictionary_path,
        row_count=len(model_frame),
        cutoff=latest,
    )
    (run_dir / "candidate-config.yaml").write_text(
        yaml.safe_dump(suggested_config, sort_keys=False), encoding="utf-8"
    )
    _write_json(
        manifest_path,
        {
            "status": (
                "candidate_built_with_bootstrapped_baseline"
                if bootstrap_baseline
                else "candidate_built"
                if regression.exact
                else "candidate_requires_review"
            ),
            "feature_builder_version": FEATURE_BUILDER_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "through": latest.isoformat(),
            "history_bouts": len(bouts),
            "label_rows": len(model_frame),
            "new_label_rows": len(new_rows),
            "baseline_rows_reused": len(reference_model) if bootstrap_baseline else 0,
            "baseline_strategy": "bootstrap" if bootstrap_baseline else "reconstruct",
            "baseline_snapshot_fighters_bootstrapped": bootstrapped_fighters,
            "split_policy": {
                "train_end": config.data.train_end.isoformat(),
                "validation_start": config.data.validation_start.isoformat(),
                "validation_end": config.data.validation_end.isoformat(),
                "test_start": config.data.test_start.isoformat(),
                "row_counts": {
                    str(label): int(count)
                    for label, count in model_frame["split"].value_counts().sort_index().items()
                },
            },
            "profiles": len(profile_frame),
            "legacy_bouts": len(legacy_bouts),
            "normalized_bouts_seen": len(normalized_bouts),
            "normalized_bouts_included": sum(
                bout.fight_id not in {legacy.fight_id for legacy in legacy_bouts} for bout in bouts
            ),
            "input_paths": {
                "legacy_fights": str(paths.legacy_fights_path),
                "legacy_profiles": str(paths.legacy_profiles_path),
                "normalized_sqlite": (
                    str(paths.normalized_sqlite_path)
                    if paths.normalized_sqlite_path is not None
                    else None
                ),
            },
            "output_paths": {
                "model_dataset": str(model_path),
                "snapshots": str(snapshots_path),
                "profiles": str(profiles_path),
                "feature_dictionary": str(dictionary_path),
                "candidate_config": str(run_dir / "candidate-config.yaml"),
            },
            "imputers": {
                "age_by_division": dict(imputers.age_by_division),
                "height_by_division": dict(imputers.height_by_division),
                "reach_offset_by_division": dict(imputers.reach_offset_by_division),
                "fallback_age": imputers.fallback_age,
                "fallback_height": imputers.fallback_height,
                "fallback_reach_offset": imputers.fallback_reach_offset,
            },
            "regression": asdict(regression) | {"exact": regression.exact},
            "processed_assets_modified": False,
            "model_retrained": False,
        },
    )
    return BuildResult(
        run_dir=run_dir,
        model_dataset_path=model_path,
        snapshots_path=snapshots_path,
        profiles_path=profiles_path,
        feature_dictionary_path=dictionary_path,
        manifest_path=manifest_path,
        regression_path=regression_path,
        generated_label_rows=len(model_frame),
        new_label_rows=len(new_rows),
        baseline_rows_reused=len(reference_model) if bootstrap_baseline else 0,
        baseline_strategy="bootstrap" if bootstrap_baseline else "reconstruct",
        generated_history_bouts=len(bouts),
        through=latest,
        regression=regression,
    )


__all__ = [
    "DEFAULT_LEGACY_IDENTITY_OVERRIDES",
    "FEATURE_BUILDER_VERSION",
    "BuildPaths",
    "BuildResult",
    "RegressionSummary",
    "build_feature_candidate",
    "default_build_paths",
    "regression_summary",
]
