"""Challenge-aware browser fetching and immutable local HTML caching."""

from __future__ import annotations

import hashlib
import os
import random
import re
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self
from urllib.parse import urlparse

from ufc_ml_latestdatafetcher import __version__
from ufc_ml_latestdatafetcher.config import SourceConfig
from ufc_ml_latestdatafetcher.errors import SourceFetchError
from ufc_ml_latestdatafetcher.models import SourcePageRecord

PageKind = Literal[
    "completed_events",
    "fighter_directory",
    "event",
    "fight",
    "fighter",
]

_EXPECTED_SELECTORS: dict[PageKind, str] = {
    "completed_events": "a[href*='/event-details/']",
    "fighter_directory": "a[href*='/fighter-details/']",
    "event": "table.b-fight-details__table_type_event-details",
    "fight": ".b-fight-details__person-link",
    "fighter": ".b-content__title-record",
}
_CHALLENGE_MARKERS = (
    "Checking your browser",
    "<title>Loading…</title>",
    "<title>Loading…</title>",
    "<title>Loading...</title>",
    "cf-chl",
)


@dataclass(frozen=True, slots=True)
class FetchedPage:
    html: str
    record: SourcePageRecord


class HtmlCache:
    """Store immutable, content-addressed HTML with an atomic latest pointer."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()

    def _directory_for(self, kind: PageKind, source_id: str) -> Path:
        safe_id = "".join(
            character for character in source_id if character.isalnum() or character in "-_"
        )
        if not safe_id or safe_id != source_id:
            raise ValueError(f"unsafe source ID {source_id!r}")
        directory = {
            "completed_events": "completed-events",
            "fighter_directory": "fighter-directory",
            "event": "events",
            "fight": "fights",
            "fighter": "fighters",
        }[kind]
        return self.root / directory / safe_id

    def path_for(self, kind: PageKind, source_id: str, sha256: str) -> Path:
        if re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
            raise ValueError(f"invalid SHA-256 digest {sha256!r}")
        return self._directory_for(kind, source_id) / f"{sha256}.html"

    def _latest_pointer(self, kind: PageKind, source_id: str) -> Path:
        return self._directory_for(kind, source_id) / "latest.sha256"

    def read(self, kind: PageKind, source_id: str) -> str | None:
        pointer = self._latest_pointer(kind, source_id)
        if not pointer.is_file():
            # Migrate caches written by the pre-content-addressed prototype in place.
            legacy_path = self._directory_for(kind, source_id).with_suffix(".html")
            if not legacy_path.is_file():
                return None
            legacy_stat = legacy_path.stat()
            html = legacy_path.read_text(encoding="utf-8")
            migrated_path = self.write(kind, source_id, html)
            with suppress(OSError):
                os.utime(
                    migrated_path,
                    ns=(legacy_stat.st_atime_ns, legacy_stat.st_mtime_ns),
                )
            return html
        digest = pointer.read_text(encoding="ascii").strip()
        path = self.path_for(kind, source_id, digest)
        if not path.is_file():
            return None
        html = path.read_text(encoding="utf-8")
        actual = hashlib.sha256(html.encode("utf-8")).hexdigest()
        if actual != digest:
            raise SourceFetchError(f"cached HTML checksum mismatch: {path}")
        return html

    def write(self, kind: PageKind, source_id: str, html: str) -> Path:
        digest = hashlib.sha256(html.encode("utf-8")).hexdigest()
        path = self.path_for(kind, source_id, digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file():
            temporary = path.with_suffix(".html.tmp")
            temporary.write_text(html, encoding="utf-8")
            temporary.replace(path)
        pointer = self._latest_pointer(kind, source_id)
        pointer_temporary = pointer.with_suffix(".sha256.tmp")
        pointer_temporary.write_text(digest, encoding="ascii")
        pointer_temporary.replace(pointer)
        return path


def validate_source_html(html: str, *, kind: PageKind, url: str) -> None:
    """Reject challenge, error, and wrong-page responses before caching or parsing."""

    if any(marker.casefold() in html.casefold() for marker in _CHALLENGE_MARKERS):
        raise SourceFetchError(f"browser challenge did not complete for {url}")
    # This is intentionally only a cheap pre-parser guard. BrowserFetcher has
    # already waited for the exact CSS selector, while cached pages are fully
    # checked by their parser.
    marker = {
        "completed_events": "/event-details/",
        "fighter_directory": "/fighter-details/",
        "event": "b-fight-details__table_type_event-details",
        "fight": "b-fight-details__person-link",
        "fighter": "b-content__title-record",
    }[kind]
    if marker not in html:
        raise SourceFetchError(f"{kind} marker is absent from {url}")


class BrowserFetcher:
    """Reuse one Playwright browser context so the UFCStats challenge cookie persists."""

    def __init__(
        self,
        config: SourceConfig,
        *,
        headless: bool | None = None,
        delay_seconds: float | None = None,
    ) -> None:
        self.config = config
        self.headless = config.headless if headless is None else headless
        self.delay_seconds = config.delay_seconds if delay_seconds is None else delay_seconds
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None
        self._last_request_at: float | None = None

    def __enter__(self) -> Self:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise SourceFetchError(
                'Playwright is required. Install with `pip install -e ".[latestdata]"` '
                "and then run `python -m playwright install chromium`."
            ) from exc
        try:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=self.headless)
            self._new_context()
        except Exception as exc:
            self.close()
            raise SourceFetchError(
                "Could not launch Chromium. Run `python -m playwright install chromium`."
            ) from exc
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _new_context(self) -> None:
        if self._context is not None:
            self._context.close()
        assert self._browser is not None
        self._context = self._browser.new_context(user_agent=self.config.user_agent)
        self._page = self._context.new_page()

    def _throttle(self) -> None:
        if self._last_request_at is None:
            return
        jitter = random.uniform(0.0, self.config.jitter_seconds)
        remaining = self.delay_seconds + jitter - (time.monotonic() - self._last_request_at)
        if remaining > 0:
            time.sleep(remaining)

    def fetch(self, url: str, *, kind: PageKind) -> str:
        if self._page is None:
            raise SourceFetchError("BrowserFetcher must be used as a context manager")
        timeout_ms = self.config.timeout_seconds * 1000
        last_error: Exception | None = None
        for attempt in range(1, self.config.retry_attempts + 1):
            try:
                self._throttle()
                response = self._page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                self._last_request_at = time.monotonic()
                if response is not None and response.status >= 400:
                    raise SourceFetchError(f"HTTP {response.status} for {url}")
                self._page.wait_for_selector(_EXPECTED_SELECTORS[kind], timeout=timeout_ms)
                requested_path = urlparse(url).path.rstrip("/")
                final_path = urlparse(str(self._page.url)).path.rstrip("/")
                if final_path != requested_path:
                    raise SourceFetchError(
                        f"unexpected redirect while fetching {url}: {self._page.url}"
                    )
                html = str(self._page.content())
                validate_source_html(html, kind=kind, url=url)
                return html
            except Exception as exc:
                last_error = exc
                if attempt < self.config.retry_attempts:
                    self._new_context()
                    time.sleep(min(2**attempt, 8))
        raise SourceFetchError(
            f"Failed to fetch {kind} page after {self.config.retry_attempts} attempts: {url}: "
            f"{last_error}"
        ) from last_error

    def close(self) -> None:
        for value in (self._context, self._browser, self._playwright):
            if value is not None:
                with suppress(Exception):
                    value.close() if hasattr(value, "close") else value.stop()
        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None


class CachedBrowserClient:
    """Read valid cache entries first and record provenance for every page."""

    def __init__(self, fetcher: BrowserFetcher, cache: HtmlCache) -> None:
        self.fetcher = fetcher
        self.cache = cache
        self.records: list[SourcePageRecord] = []

    def get(
        self,
        url: str,
        *,
        kind: PageKind,
        source_id: str,
        refresh: bool = False,
    ) -> FetchedPage:
        if refresh:
            html = None
        else:
            try:
                html = self.cache.read(kind, source_id)
            except SourceFetchError:
                html = None
        from_cache = html is not None
        if html is not None:
            try:
                validate_source_html(html, kind=kind, url=url)
            except SourceFetchError:
                html = None
                from_cache = False
        if html is None:
            html = self.fetcher.fetch(url, kind=kind)
            cache_path = self.cache.write(kind, source_id, html)
            fetched_at = datetime.now(UTC).isoformat()
        else:
            digest = hashlib.sha256(html.encode("utf-8")).hexdigest()
            cache_path = self.cache.path_for(kind, source_id, digest)
            fetched_at = datetime.fromtimestamp(cache_path.stat().st_mtime, UTC).isoformat()
        record = SourcePageRecord(
            page_kind=kind,
            source_id=source_id,
            url=url,
            fetched_at_utc=fetched_at,
            sha256=hashlib.sha256(html.encode("utf-8")).hexdigest(),
            cache_path=str(cache_path),
            from_cache=int(from_cache),
            parser_version=__version__,
        )
        self.records.append(record)
        return FetchedPage(html=html, record=record)


__all__ = [
    "BrowserFetcher",
    "CachedBrowserClient",
    "FetchedPage",
    "HtmlCache",
    "PageKind",
    "validate_source_html",
]
