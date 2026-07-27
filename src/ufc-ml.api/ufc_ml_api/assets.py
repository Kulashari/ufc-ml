"""Secure runtime retrieval for private, read-only model assets.

The public repository intentionally excludes processed datasets and trained
artifacts.  A deployed API can opt into this module by supplying a read-only
GitHub token through its host's secret manager.  The token is sent only as an
HTTP authorization header, is never written to disk, and is never included in
an exception message.
"""

from __future__ import annotations

import os
import re
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

_ARCHIVE_API_URL = "https://api.github.com/repos/{repository}/tarball/{ref}"
_ARCHIVE_REDIRECT_HOSTS = frozenset({"codeload.github.com", "github.com"})
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
_MAX_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024
_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
_REPOSITORY_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}/[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$"
_TOKEN_PATTERN = r"^[A-Za-z0-9_-]+$"
_SHA_LENGTHS = frozenset({40, 64})
_REQUIRED_DATA_FILES = (
    Path("data/processed/ufc_model_ready.csv"),
    Path("data/processed/ufc_fighter_latest_features.csv"),
    Path("data/processed/ufc_fighter_profiles_clean.csv"),
    Path("data/processed/ufc_model_feature_dictionary.csv"),
)


class AssetBootstrapError(RuntimeError):
    """Raised when private runtime assets cannot be acquired safely."""


@dataclass(frozen=True)
class AssetCheckout:
    """Location and immutable source identity of a checked-out asset bundle."""

    root: Path
    repository: str
    ref: str


class _NoRedirect(HTTPRedirectHandler):
    """Expose GitHub's archive redirect so credentials never cross hosts."""

    def redirect_request(
        self,
        request: Request,
        file_pointer: Any,
        status_code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> None:
        return None


def _env_value(name: str) -> str | None:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else None


def _validate_repository(repository: str) -> None:
    if not re.fullmatch(_REPOSITORY_PATTERN, repository):
        raise AssetBootstrapError(
            "UFC_ML_ASSETS_REPOSITORY must use the exact owner/repository format."
        )


def _validate_ref(ref: str) -> None:
    if len(ref) not in _SHA_LENGTHS or any(
        character not in "0123456789abcdef" for character in ref
    ):
        raise AssetBootstrapError(
            "UFC_ML_ASSETS_REF must be a full lowercase Git commit SHA (40 or 64 characters)."
        )


def _validate_token(token: str) -> None:
    """Reject values that could be reflected by an invalid HTTP header error."""

    if not re.fullmatch(_TOKEN_PATTERN, token):
        raise AssetBootstrapError("GITHUB_ASSETS_TOKEN contains unsupported characters.")


def _asset_parent_directory() -> str | None:
    configured = _env_value("UFC_ML_ASSETS_DIR")
    if configured is None:
        return None
    path = Path(configured).expanduser().resolve()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise AssetBootstrapError(
            "Could not create the configured runtime asset directory."
        ) from error
    if not path.is_dir():
        raise AssetBootstrapError("UFC_ML_ASSETS_DIR must point to a directory.")
    return str(path)


def _archive_redirect(location: str) -> str:
    parsed = urlparse(location)
    if parsed.scheme != "https" or parsed.hostname not in _ARCHIVE_REDIRECT_HOSTS:
        raise AssetBootstrapError("GitHub returned an unexpected private-asset archive location.")
    return location


def _write_response(response: Any, destination: Path) -> None:
    written = 0
    with destination.open("wb") as output:
        while chunk := response.read(_DOWNLOAD_CHUNK_SIZE):
            written += len(chunk)
            if written > _MAX_ARCHIVE_BYTES:
                raise AssetBootstrapError(
                    "The private-asset archive exceeds the supported size limit."
                )
            output.write(chunk)


def _download_archive(*, repository: str, ref: str, token: str) -> Path:
    file_descriptor, archive_name = tempfile.mkstemp(prefix="ufc-ml-assets-", suffix=".tar.gz")
    os.close(file_descriptor)
    archive = Path(archive_name)
    try:
        request = Request(
            _ARCHIVE_API_URL.format(repository=repository, ref=ref),
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "User-Agent": "ufc-ml-api-assets-bootstrap",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            response = build_opener(_NoRedirect()).open(request, timeout=60)
        except HTTPError as error:
            if error.code not in {301, 302, 303, 307, 308}:
                raise
            location = error.headers.get("Location")
            if not location:
                raise AssetBootstrapError(
                    "GitHub did not provide a private-asset archive location."
                ) from error
            redirected_request = Request(
                _archive_redirect(location),
                headers={"User-Agent": "ufc-ml-api-assets-bootstrap"},
            )
            response = build_opener(_NoRedirect()).open(redirected_request, timeout=60)
        with response:
            _write_response(response, archive)
        return archive
    except AssetBootstrapError:
        archive.unlink(missing_ok=True)
        raise
    except (HTTPError, URLError, OSError, ValueError) as error:
        archive.unlink(missing_ok=True)
        raise AssetBootstrapError(
            "Could not download private assets. Verify the repository, pinned "
            "commit, and backend secret."
        ) from error


def _safe_archive_member_path(member: tarfile.TarInfo, root_name: str) -> PurePosixPath | None:
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != root_name:
        raise AssetBootstrapError("The private-asset archive contains an unsafe path.")
    if (
        member.issym()
        or member.islnk()
        or member.isdev()
        or not (member.isdir() or member.isfile())
    ):
        raise AssetBootstrapError("The private-asset archive contains an unsupported file type.")
    relative_parts = path.parts[1:]
    return PurePosixPath(*relative_parts) if relative_parts else None


def _extract_archive(archive_path: Path, destination: Path) -> None:
    try:
        with tarfile.open(archive_path, mode="r:*") as archive:
            members = archive.getmembers()
            root_names = {
                PurePosixPath(member.name).parts[0]
                for member in members
                if PurePosixPath(member.name).parts
            }
            if len(root_names) != 1:
                raise AssetBootstrapError("The private-asset archive has an unexpected layout.")
            root_name = next(iter(root_names))
            total_size = 0
            extracted: set[PurePosixPath] = set()
            destination = destination.resolve()
            for member in members:
                relative = _safe_archive_member_path(member, root_name)
                if relative is None:
                    continue
                if member.isfile():
                    total_size += member.size
                    if total_size > _MAX_ARCHIVE_BYTES:
                        raise AssetBootstrapError(
                            "The private-asset archive exceeds the supported extracted size limit."
                        )
                target = (destination / Path(*relative.parts)).resolve()
                try:
                    target.relative_to(destination)
                except ValueError as error:
                    raise AssetBootstrapError(
                        "The private-asset archive contains an unsafe path."
                    ) from error
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if relative in extracted:
                    raise AssetBootstrapError(
                        "The private-asset archive contains duplicate file paths."
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise AssetBootstrapError("Could not extract a private-asset file.")
                with source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=_DOWNLOAD_CHUNK_SIZE)
                extracted.add(relative)
    except (OSError, tarfile.TarError) as error:
        raise AssetBootstrapError("Could not safely unpack the private-asset archive.") from error


def _is_lfs_pointer(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX


def _validate_asset_layout(root: Path) -> None:
    missing = [str(path) for path in _REQUIRED_DATA_FILES if not (root / path).is_file()]
    if missing:
        raise AssetBootstrapError(
            "Private assets are missing required processed files: " + ", ".join(missing)
        )
    protected_files = [root / relative_path for relative_path in _REQUIRED_DATA_FILES]
    protected_files.extend(root.glob("artifacts/**/*.joblib"))
    for path in protected_files:
        if path.is_file() and _is_lfs_pointer(path):
            raise AssetBootstrapError(
                "Private assets contain a Git LFS pointer instead of file contents. "
                "Include LFS objects in GitHub source archives or store these assets "
                "outside Git LFS."
            )


def checkout_configured_assets() -> AssetCheckout | None:
    """Download a pinned private asset archive when deployment variables are set.

    Returning ``None`` preserves the local workflow: local data and artifacts
    remain usable when no repository is configured.
    """

    repository = _env_value("UFC_ML_ASSETS_REPOSITORY")
    if repository is None:
        return None
    token = _env_value("GITHUB_ASSETS_TOKEN")
    ref = _env_value("UFC_ML_ASSETS_REF")
    if token is None:
        raise AssetBootstrapError("GITHUB_ASSETS_TOKEN is required for private asset retrieval.")
    if ref is None:
        raise AssetBootstrapError("UFC_ML_ASSETS_REF is required for private asset retrieval.")
    _validate_repository(repository)
    _validate_token(token)
    _validate_ref(ref)

    try:
        root = Path(tempfile.mkdtemp(prefix="ufc-ml-assets-", dir=_asset_parent_directory()))
    except OSError as error:
        raise AssetBootstrapError("Could not create a private-asset checkout directory.") from error
    archive_path: Path | None = None
    try:
        archive_path = _download_archive(repository=repository, ref=ref, token=token)
        _extract_archive(archive_path, root)
        _validate_asset_layout(root)
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise
    finally:
        if archive_path is not None:
            archive_path.unlink(missing_ok=True)
    return AssetCheckout(root=root, repository=repository, ref=ref)


__all__ = ["AssetBootstrapError", "AssetCheckout", "checkout_configured_assets"]
