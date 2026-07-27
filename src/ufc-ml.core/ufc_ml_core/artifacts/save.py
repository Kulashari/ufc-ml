"""Versioned, integrity-checked model artifact serialization."""

from __future__ import annotations

import json
import platform
import re
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import UTC, date, datetime
from enum import Enum
from hashlib import sha256
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, cast

import joblib

from ..exceptions import UFCPredictorError

ARTIFACT_FORMAT_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

ARTIFACT_FILES: Mapping[str, str] = {
    "pipeline": "pipeline.joblib",
    "calibrator": "calibrator.joblib",
    "metadata": "metadata.json",
    "feature_names": "feature_names.json",
    "feature_groups": "feature_groups.json",
    "schema": "schema.json",
    "config": "config.json",
    "data_fingerprint": "data_fingerprint.json",
    "cutoff": "cutoff.json",
    "metrics": "metrics.json",
    "feature_importances": "feature_importances.json",
    "seeds": "seeds.json",
    "versions": "versions.json",
    "git_hash": "git_hash.txt",
}

_DEFAULT_PACKAGES = (
    "joblib",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
)


class ArtifactSaveError(UFCPredictorError, RuntimeError):
    """Raised when an artifact bundle cannot be written safely."""


def _date_string(value: date | datetime | str) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("cutoff_date cannot be empty")
        try:
            return date.fromisoformat(cleaned[:10]).isoformat()
        except ValueError as exc:
            raise ValueError(f"cutoff_date must be an ISO date, got {value!r}") from exc
    raise TypeError("cutoff_date must be a date, datetime, or ISO date string")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _jsonable(asdict(cast(Any, value)))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_jsonable(item) for item in sorted(value, key=str)]
    if hasattr(value, "to_dict"):
        try:
            return _jsonable(value.to_dict(orient="records"))
        except TypeError:
            return _jsonable(value.to_dict())
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            _jsonable(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_files(
    paths: Iterable[str | Path],
    *,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Fingerprint source files and return a stable combined SHA-256."""

    root_path = Path(root).resolve() if root is not None else None
    entries: list[dict[str, Any]] = []
    for raw_path in sorted((Path(value) for value in paths), key=lambda item: str(item)):
        resolved = raw_path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        if root_path is not None:
            try:
                label = resolved.relative_to(root_path).as_posix()
            except ValueError as exc:
                raise ValueError(f"{resolved} is outside fingerprint root {root_path}") from exc
        else:
            label = resolved.name
        entries.append(
            {
                "path": label,
                "sha256": sha256_file(resolved),
                "size_bytes": resolved.stat().st_size,
            }
        )

    combined = sha256()
    for entry in entries:
        combined.update(entry["path"].encode("utf-8"))
        combined.update(b"\0")
        combined.update(entry["sha256"].encode("ascii"))
        combined.update(b"\0")
        combined.update(str(entry["size_bytes"]).encode("ascii"))
        combined.update(b"\n")
    return {"algorithm": "sha256", "combined_sha256": combined.hexdigest(), "files": entries}


def collect_runtime_versions(
    packages: Sequence[str] = _DEFAULT_PACKAGES,
) -> dict[str, Any]:
    versions: dict[str, Any] = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": {},
    }
    for package in packages:
        try:
            version = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            version = None
        versions["packages"][package] = version
    return versions


def get_git_hash(repository: str | Path | None = None) -> str | None:
    """Return HEAD without invoking a shell, or ``None`` outside a repository."""

    command = ["git"]
    if repository is not None:
        command.extend(["-C", str(Path(repository).resolve())])
    command.extend(["rev-parse", "HEAD"])
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value if re.fullmatch(r"[0-9a-fA-F]{40,64}", value) else None


def _validated_feature_names(feature_names: Sequence[str]) -> tuple[str, ...]:
    result = tuple(str(name).strip() for name in feature_names)
    if not result or any(not name for name in result):
        raise ValueError("feature_names must contain non-empty names")
    if len(set(result)) != len(result):
        raise ValueError("feature_names must be unique and ordered")
    return result


def _validate_estimator_feature_order(
    estimator: Any,
    feature_names: Sequence[str],
) -> None:
    raw_names = getattr(estimator, "feature_names_in_", None)
    if raw_names is None:
        return
    estimator_names = tuple(str(name) for name in raw_names)
    if estimator_names != tuple(feature_names):
        raise ArtifactSaveError(
            "Estimator feature_names_in_ does not match the persisted feature order"
        )


def _validated_feature_groups(
    feature_groups: Mapping[str, Sequence[str]],
    feature_names: Sequence[str],
) -> dict[str, list[str]]:
    known = set(feature_names)
    result: dict[str, list[str]] = {}
    for group_name, members in feature_groups.items():
        cleaned_group = str(group_name).strip()
        if not cleaned_group:
            raise ValueError("feature group names cannot be empty")
        cleaned_members = [str(member).strip() for member in members]
        if any(not member for member in cleaned_members):
            raise ValueError(f"feature group {cleaned_group!r} has an empty member")
        unknown = sorted(set(cleaned_members) - known)
        if unknown:
            raise ValueError(
                f"feature group {cleaned_group!r} references unknown features: {unknown}"
            )
        result[cleaned_group] = cleaned_members
    return result


def _safe_destination(destination: str | Path) -> Path:
    target = Path(destination).expanduser().resolve()
    if target == target.parent:
        raise ValueError("artifact destination cannot be a filesystem root")
    if not target.name or target.name in {".", ".."}:
        raise ValueError("artifact destination must be a named directory")
    return target


def _install_directory(temp_directory: Path, target: Path, *, overwrite: bool) -> None:
    if not target.exists():
        temp_directory.replace(target)
        return
    if not overwrite:
        raise FileExistsError(
            f"artifact directory already exists: {target}; pass overwrite=True explicitly"
        )
    if not target.is_dir():
        raise FileExistsError(f"artifact destination is not a directory: {target}")

    backup = target.with_name(f".{target.name}.backup-{uuid.uuid4().hex}")
    target.replace(backup)
    try:
        temp_directory.replace(target)
    except Exception:
        backup.replace(target)
        raise
    else:
        shutil.rmtree(backup)


def save_artifacts(
    destination: str | Path,
    *,
    artifact_version: str,
    pipeline: Any,
    calibrator: Any | None,
    metadata: Mapping[str, Any],
    feature_names: Sequence[str],
    feature_groups: Mapping[str, Sequence[str]],
    schema: Mapping[str, Any],
    config: Mapping[str, Any],
    data_fingerprint: Mapping[str, Any] | str,
    cutoff_date: date | datetime | str,
    metrics: Mapping[str, Any],
    feature_importances: Any,
    seeds: Mapping[str, Any],
    package_versions: Mapping[str, Any] | None = None,
    git_hash: str | None = None,
    repository: str | Path | None = None,
    overwrite: bool = False,
    compression: int | tuple[str, int] = 3,
) -> Path:
    """Persist a complete model bundle and an integrity manifest atomically."""

    if not _VERSION_PATTERN.fullmatch(artifact_version):
        raise ValueError(
            "artifact_version must start with an alphanumeric character and "
            "contain only letters, numbers, '.', '_', or '-'"
        )
    ordered_features = _validated_feature_names(feature_names)
    _validate_estimator_feature_order(pipeline, ordered_features)
    groups = _validated_feature_groups(feature_groups, ordered_features)
    cutoff = _date_string(cutoff_date)
    target = _safe_destination(destination)
    target.parent.mkdir(parents=True, exist_ok=True)

    temp_directory = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        joblib.dump(
            pipeline,
            temp_directory / ARTIFACT_FILES["pipeline"],
            compress=compression,
        )
        components: dict[str, str | None] = {
            "pipeline": ARTIFACT_FILES["pipeline"],
            "calibrator": None,
        }
        if calibrator is not None:
            joblib.dump(
                calibrator,
                temp_directory / ARTIFACT_FILES["calibrator"],
                compress=compression,
            )
            components["calibrator"] = ARTIFACT_FILES["calibrator"]

        fingerprint_payload = (
            {"algorithm": "sha256", "combined_sha256": data_fingerprint}
            if isinstance(data_fingerprint, str)
            else data_fingerprint
        )
        json_payloads: Mapping[str, Any] = {
            "metadata": metadata,
            "feature_names": {"feature_names": list(ordered_features)},
            "feature_groups": {"feature_groups": groups},
            "schema": schema,
            "config": config,
            "data_fingerprint": fingerprint_payload,
            "cutoff": {"training_data_cutoff": cutoff},
            "metrics": metrics,
            "feature_importances": feature_importances,
            "seeds": seeds,
            "versions": package_versions or collect_runtime_versions(),
        }
        for logical_name, payload in json_payloads.items():
            filename = ARTIFACT_FILES[logical_name]
            _write_json(temp_directory / filename, payload)
            components[logical_name] = filename

        resolved_git_hash = git_hash or get_git_hash(repository)
        (temp_directory / ARTIFACT_FILES["git_hash"]).write_text(
            f"{resolved_git_hash or 'unknown'}\n", encoding="utf-8"
        )
        components["git_hash"] = ARTIFACT_FILES["git_hash"]

        integrity: dict[str, dict[str, Any]] = {}
        for path in sorted(temp_directory.iterdir(), key=lambda item: item.name):
            if not path.is_file():
                raise ArtifactSaveError(
                    f"unexpected non-file entry in artifact bundle: {path.name}"
                )
            integrity[path.name] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }

        manifest = {
            "artifact_format_version": ARTIFACT_FORMAT_VERSION,
            "artifact_version": artifact_version,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "components": components,
            "integrity": {
                "algorithm": "sha256",
                "files": integrity,
            },
        }
        _write_json(temp_directory / MANIFEST_FILENAME, manifest)
        _install_directory(temp_directory, target, overwrite=overwrite)
    except Exception:
        if temp_directory.exists():
            shutil.rmtree(temp_directory)
        raise
    return target


def save_artifact_version(
    root: str | Path,
    *,
    artifact_version: str,
    **kwargs: Any,
) -> Path:
    """Save into ``root/artifact_version`` using the same manifest contract."""

    root_path = Path(root).expanduser().resolve()
    return save_artifacts(
        root_path / artifact_version,
        artifact_version=artifact_version,
        **kwargs,
    )


def update_artifact_metrics(
    artifact_directory: str | Path,
    metrics: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    """Replace ``metrics.json`` while preserving full manifest integrity.

    This supports the explicit final-evaluation workflow: training creates a
    bundle whose test entry is ``not_evaluated`` and the held-out command later
    installs the completed metrics using an atomic directory swap.
    """

    from .load import load_artifacts

    target = _safe_destination(artifact_directory)
    loaded = load_artifacts(target, verify_integrity=True)
    prior_test = loaded.metrics.get("test")
    if (
        isinstance(prior_test, Mapping)
        and prior_test.get("status") == "evaluated"
        and not overwrite
    ):
        raise FileExistsError("final test metrics already exist; pass overwrite=True explicitly")

    manifest = cast(
        dict[str, Any],
        json.loads(json.dumps(dict(loaded.manifest))),
    )
    components = manifest.get("components")
    integrity = manifest.get("integrity")
    if not isinstance(components, dict) or not isinstance(integrity, dict):
        raise ArtifactSaveError("loaded artifact manifest has invalid structure")
    metrics_filename = components.get("metrics")
    integrity_files = integrity.get("files")
    if (
        not isinstance(metrics_filename, str)
        or Path(metrics_filename).name != metrics_filename
        or not isinstance(integrity_files, dict)
    ):
        raise ArtifactSaveError("loaded artifact metrics manifest entry is invalid")

    temp_directory = Path(tempfile.mkdtemp(prefix=f".{target.name}.metrics-", dir=target.parent))
    try:
        shutil.copytree(target, temp_directory, dirs_exist_ok=True)
        metrics_path = temp_directory / metrics_filename
        _write_json(metrics_path, metrics)
        integrity_files[metrics_filename] = {
            "sha256": sha256_file(metrics_path),
            "size_bytes": metrics_path.stat().st_size,
        }
        manifest["final_metrics_updated_at_utc"] = datetime.now(UTC).isoformat()
        _write_json(temp_directory / MANIFEST_FILENAME, manifest)
        load_artifacts(temp_directory, verify_integrity=True)
        _install_directory(temp_directory, target, overwrite=True)
    except Exception:
        if temp_directory.exists():
            shutil.rmtree(temp_directory)
        raise

    load_artifacts(target, verify_integrity=True)
    return target


__all__ = [
    "ARTIFACT_FILES",
    "ARTIFACT_FORMAT_VERSION",
    "MANIFEST_FILENAME",
    "ArtifactSaveError",
    "collect_runtime_versions",
    "fingerprint_files",
    "get_git_hash",
    "save_artifact_version",
    "save_artifacts",
    "sha256_file",
    "update_artifact_metrics",
]
