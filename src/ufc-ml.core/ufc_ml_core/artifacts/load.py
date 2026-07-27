"""Loading and validation for versioned UFC model artifact bundles."""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

import joblib

from ..exceptions import UFCPredictorError
from .save import (
    ARTIFACT_FORMAT_VERSION,
    MANIFEST_FILENAME,
    sha256_file,
)


class ArtifactLoadError(UFCPredictorError, RuntimeError):
    """Base class for artifact loading failures."""


class ArtifactNotFoundError(ArtifactLoadError):
    pass


class ArtifactIntegrityError(ArtifactLoadError):
    pass


class UnsupportedArtifactVersionError(ArtifactLoadError):
    pass


@dataclass(frozen=True)
class ArtifactVersionInfo:
    artifact_version: str
    path: Path
    created_at_utc: datetime
    artifact_format_version: int


@dataclass(frozen=True)
class LoadedArtifacts:
    path: Path
    artifact_format_version: int
    artifact_version: str
    created_at_utc: datetime
    pipeline: Any
    calibrator: Any | None
    metadata: Mapping[str, Any]
    feature_names: tuple[str, ...]
    feature_groups: Mapping[str, tuple[str, ...]]
    schema: Mapping[str, Any]
    config: Mapping[str, Any]
    data_fingerprint: Mapping[str, Any]
    cutoff_date: date
    metrics: Mapping[str, Any]
    feature_importances: Any
    seeds: Mapping[str, Any]
    versions: Mapping[str, Any]
    git_hash: str | None
    manifest: Mapping[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "artifact_format_version": self.artifact_format_version,
            "artifact_version": self.artifact_version,
            "created_at_utc": self.created_at_utc.isoformat(),
            "cutoff_date": self.cutoff_date.isoformat(),
            "feature_count": len(self.feature_names),
            "has_calibrator": self.calibrator is not None,
            "git_hash": self.git_hash,
        }


_REQUIRED_COMPONENTS = frozenset(
    {
        "pipeline",
        "metadata",
        "feature_names",
        "feature_groups",
        "schema",
        "config",
        "data_fingerprint",
        "cutoff",
        "metrics",
        "feature_importances",
        "seeds",
        "versions",
        "git_hash",
    }
)

_CURRENT_PACKAGE_NAME = "ufc_ml_core"
_LEGACY_PACKAGE_NAME = "ufc_predictor"


def _enable_legacy_artifact_imports() -> None:
    """Allow trusted artifacts saved before the core package rename to load.

    Pickle stores an object's fully qualified module path.  Existing model runs
    were saved under ``ufc_predictor``; expose that name only in memory while
    loading so the source tree can remain on the ``ufc_ml_core`` layout.
    """

    sys.modules.setdefault(_LEGACY_PACKAGE_NAME, sys.modules[_CURRENT_PACKAGE_NAME])


_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ArtifactNotFoundError(f"missing artifact file: {path}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError(f"invalid JSON artifact file: {path}") from exc


def _mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactIntegrityError(f"{field_name} must be a JSON object")
    return dict(value)


def _parse_created_at(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ArtifactIntegrityError("manifest created_at_utc must be a string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ArtifactIntegrityError("invalid manifest created_at_utc") from exc
    if parsed.tzinfo is None:
        raise ArtifactIntegrityError("manifest created_at_utc must include a timezone")
    return parsed


def _parse_cutoff(payload: Any) -> date:
    mapping = _mapping(payload, field_name="cutoff")
    value = mapping.get("training_data_cutoff")
    if not isinstance(value, str):
        raise ArtifactIntegrityError("cutoff.json must contain training_data_cutoff")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ArtifactIntegrityError("invalid training_data_cutoff") from exc


def _safe_component_path(root: Path, filename: Any) -> Path:
    if not isinstance(filename, str) or not filename:
        raise ArtifactIntegrityError("manifest component filename must be a string")
    relative = Path(filename)
    if relative.is_absolute() or len(relative.parts) != 1 or relative.name != filename:
        raise ArtifactIntegrityError(f"manifest component path is unsafe: {filename!r}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ArtifactIntegrityError(
            f"manifest component escapes artifact directory: {filename!r}"
        ) from exc
    return resolved


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = _mapping(_read_json(path / MANIFEST_FILENAME), field_name="manifest")
    for required in (
        "artifact_format_version",
        "artifact_version",
        "created_at_utc",
        "components",
        "integrity",
    ):
        if required not in manifest:
            raise ArtifactIntegrityError(f"manifest is missing required field {required!r}")
    return manifest


def _validate_manifest(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    verify_integrity: bool,
) -> dict[str, Path | None]:
    components = _mapping(manifest["components"], field_name="manifest.components")
    missing_components = sorted(
        component for component in _REQUIRED_COMPONENTS if not components.get(component)
    )
    if missing_components:
        raise ArtifactIntegrityError(
            f"manifest is missing required components: {missing_components}"
        )

    paths: dict[str, Path | None] = {}
    for logical_name, filename in components.items():
        if filename is None:
            if logical_name != "calibrator":
                raise ArtifactIntegrityError(
                    f"only calibrator may be an absent optional component, got {logical_name!r}"
                )
            paths[logical_name] = None
            continue
        paths[logical_name] = _safe_component_path(root, filename)

    integrity = _mapping(manifest["integrity"], field_name="manifest.integrity")
    if integrity.get("algorithm") != "sha256":
        raise ArtifactIntegrityError("unsupported artifact integrity algorithm")
    file_entries = _mapping(integrity.get("files"), field_name="manifest.integrity.files")
    expected_filenames = {path.name for path in paths.values() if path is not None}
    if set(file_entries) != expected_filenames:
        missing = sorted(expected_filenames - set(file_entries))
        unexpected = sorted(set(file_entries) - expected_filenames)
        raise ArtifactIntegrityError(
            f"integrity file set mismatch; missing={missing}, unexpected={unexpected}"
        )

    actual_files = {
        path.name for path in root.iterdir() if path.is_file() and path.name != MANIFEST_FILENAME
    }
    if actual_files != expected_filenames:
        missing = sorted(expected_filenames - actual_files)
        unexpected = sorted(actual_files - expected_filenames)
        raise ArtifactIntegrityError(
            f"artifact file set mismatch; missing={missing}, unexpected={unexpected}"
        )

    if verify_integrity:
        for filename, raw_entry in file_entries.items():
            entry = _mapping(raw_entry, field_name=f"manifest.integrity.files.{filename}")
            expected_hash = entry.get("sha256")
            expected_size = entry.get("size_bytes")
            if not isinstance(expected_hash, str) or not re.fullmatch(
                r"[0-9a-f]{64}", expected_hash
            ):
                raise ArtifactIntegrityError(f"invalid SHA-256 entry for {filename}")
            if not isinstance(expected_size, int) or expected_size < 0:
                raise ArtifactIntegrityError(f"invalid size entry for {filename}")
            path = root / filename
            if path.stat().st_size != expected_size:
                raise ArtifactIntegrityError(f"size mismatch for {filename}")
            if sha256_file(path) != expected_hash:
                raise ArtifactIntegrityError(f"SHA-256 mismatch for {filename}")
    return paths


def _load_feature_names(payload: Any) -> tuple[str, ...]:
    mapping = _mapping(payload, field_name="feature_names")
    values = mapping.get("feature_names")
    if not isinstance(values, list) or not values:
        raise ArtifactIntegrityError("feature_names must be a non-empty list")
    names = tuple(str(value) for value in values)
    if any(not value for value in names) or len(set(names)) != len(names):
        raise ArtifactIntegrityError("feature_names must be non-empty and unique")
    return names


def _load_feature_groups(
    payload: Any, feature_names: tuple[str, ...]
) -> Mapping[str, tuple[str, ...]]:
    wrapper = _mapping(payload, field_name="feature_groups")
    raw_groups = _mapping(wrapper.get("feature_groups"), field_name="feature_groups.feature_groups")
    known = set(feature_names)
    groups: dict[str, tuple[str, ...]] = {}
    for group_name, raw_members in raw_groups.items():
        if not isinstance(raw_members, list):
            raise ArtifactIntegrityError(f"feature group {group_name!r} must be a list")
        members = tuple(str(value) for value in raw_members)
        unknown = sorted(set(members) - known)
        if unknown:
            raise ArtifactIntegrityError(
                f"feature group {group_name!r} contains unknown features: {unknown}"
            )
        groups[str(group_name)] = members
    return MappingProxyType(groups)


def _component_json(paths: Mapping[str, Path | None], name: str) -> Any:
    path = paths.get(name)
    if path is None:
        raise ArtifactIntegrityError(f"required component {name!r} is absent")
    return _read_json(path)


def load_artifacts(
    artifact_directory: str | Path,
    *,
    verify_integrity: bool = True,
    expected_artifact_version: str | None = None,
    supported_format_version: int = ARTIFACT_FORMAT_VERSION,
) -> LoadedArtifacts:
    """Verify a trusted artifact directory before deserializing estimators.

    Joblib/pickle artifacts can execute code while loading.  Integrity checks
    detect corruption or substitution relative to the manifest; they do not
    make an untrusted bundle safe.
    """

    root = Path(artifact_directory).expanduser().resolve()
    if not root.is_dir():
        raise ArtifactNotFoundError(f"artifact directory not found: {root}")
    manifest = _load_manifest(root)

    format_version = manifest["artifact_format_version"]
    if not isinstance(format_version, int):
        raise ArtifactIntegrityError("artifact_format_version must be an integer")
    if format_version != supported_format_version:
        raise UnsupportedArtifactVersionError(
            f"artifact format {format_version} is not supported; "
            f"expected {supported_format_version}"
        )
    artifact_version = manifest["artifact_version"]
    if not isinstance(artifact_version, str) or not _VERSION_PATTERN.fullmatch(artifact_version):
        raise ArtifactIntegrityError("invalid artifact_version")
    if expected_artifact_version is not None and artifact_version != expected_artifact_version:
        raise UnsupportedArtifactVersionError(
            f"loaded artifact version {artifact_version!r}, expected {expected_artifact_version!r}"
        )

    paths = _validate_manifest(root, manifest, verify_integrity=verify_integrity)
    feature_names = _load_feature_names(_component_json(paths, "feature_names"))
    feature_groups = _load_feature_groups(_component_json(paths, "feature_groups"), feature_names)
    schema = _mapping(_component_json(paths, "schema"), field_name="schema")
    data_fingerprint = _mapping(
        _component_json(paths, "data_fingerprint"),
        field_name="data_fingerprint",
    )
    schema_registry = schema.get("feature_registry_sha256")
    fingerprint_registry = data_fingerprint.get("feature_registry_sha256")
    if (
        schema_registry is not None
        and fingerprint_registry is not None
        and schema_registry != fingerprint_registry
    ):
        raise ArtifactIntegrityError(
            "feature registry fingerprint disagrees between schema and data provenance"
        )

    _enable_legacy_artifact_imports()
    pipeline_path = paths["pipeline"]
    assert pipeline_path is not None
    pipeline = joblib.load(pipeline_path)
    estimator_feature_names = getattr(pipeline, "feature_names_in_", None)
    if (
        estimator_feature_names is not None
        and tuple(str(name) for name in estimator_feature_names) != feature_names
    ):
        raise ArtifactIntegrityError(
            "pipeline.feature_names_in_ does not match persisted feature_names"
        )
    calibrator_path = paths.get("calibrator")
    calibrator = joblib.load(calibrator_path) if calibrator_path else None

    git_hash_path = paths["git_hash"]
    assert git_hash_path is not None
    git_hash_value = git_hash_path.read_text(encoding="utf-8").strip()
    git_hash = None if git_hash_value == "unknown" else git_hash_value
    if git_hash is not None and not re.fullmatch(r"[0-9a-fA-F]{40,64}", git_hash):
        raise ArtifactIntegrityError("git_hash.txt contains an invalid hash")

    return LoadedArtifacts(
        path=root,
        artifact_format_version=format_version,
        artifact_version=artifact_version,
        created_at_utc=_parse_created_at(manifest["created_at_utc"]),
        pipeline=pipeline,
        calibrator=calibrator,
        metadata=MappingProxyType(
            _mapping(_component_json(paths, "metadata"), field_name="metadata")
        ),
        feature_names=feature_names,
        feature_groups=feature_groups,
        schema=MappingProxyType(schema),
        config=MappingProxyType(_mapping(_component_json(paths, "config"), field_name="config")),
        data_fingerprint=MappingProxyType(data_fingerprint),
        cutoff_date=_parse_cutoff(_component_json(paths, "cutoff")),
        metrics=MappingProxyType(_mapping(_component_json(paths, "metrics"), field_name="metrics")),
        feature_importances=_component_json(paths, "feature_importances"),
        seeds=MappingProxyType(_mapping(_component_json(paths, "seeds"), field_name="seeds")),
        versions=MappingProxyType(
            _mapping(_component_json(paths, "versions"), field_name="versions")
        ),
        git_hash=git_hash,
        manifest=MappingProxyType(dict(manifest)),
    )


def list_artifact_versions(root: str | Path) -> tuple[ArtifactVersionInfo, ...]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise ArtifactNotFoundError(f"artifact root not found: {root_path}")
    versions: list[ArtifactVersionInfo] = []
    for child in root_path.iterdir():
        if not child.is_dir() or not (child / MANIFEST_FILENAME).is_file():
            continue
        manifest = _load_manifest(child)
        artifact_version = manifest.get("artifact_version")
        format_version = manifest.get("artifact_format_version")
        if not isinstance(artifact_version, str) or not isinstance(format_version, int):
            raise ArtifactIntegrityError(f"invalid manifest in {child}")
        versions.append(
            ArtifactVersionInfo(
                artifact_version=artifact_version,
                path=child.resolve(),
                created_at_utc=_parse_created_at(manifest.get("created_at_utc")),
                artifact_format_version=format_version,
            )
        )
    return tuple(
        sorted(
            versions,
            key=lambda value: (value.created_at_utc, value.artifact_version),
        )
    )


def load_artifact_version(
    root: str | Path,
    artifact_version: str,
    *,
    verify_integrity: bool = True,
) -> LoadedArtifacts:
    if not _VERSION_PATTERN.fullmatch(artifact_version):
        raise ValueError("invalid artifact_version")
    root_path = Path(root).expanduser().resolve()
    target = (root_path / artifact_version).resolve()
    try:
        target.relative_to(root_path)
    except ValueError as exc:
        raise ValueError("artifact_version escapes artifact root") from exc
    return load_artifacts(
        target,
        verify_integrity=verify_integrity,
        expected_artifact_version=artifact_version,
    )


def load_latest_artifacts(root: str | Path, *, verify_integrity: bool = True) -> LoadedArtifacts:
    versions = list_artifact_versions(root)
    if not versions:
        raise ArtifactNotFoundError(f"no artifact versions found under {root}")
    return load_artifacts(
        versions[-1].path,
        verify_integrity=verify_integrity,
        expected_artifact_version=versions[-1].artifact_version,
    )


__all__ = [
    "ArtifactIntegrityError",
    "ArtifactLoadError",
    "ArtifactNotFoundError",
    "ArtifactVersionInfo",
    "LoadedArtifacts",
    "UnsupportedArtifactVersionError",
    "list_artifact_versions",
    "load_artifact_version",
    "load_artifacts",
    "load_latest_artifacts",
]
