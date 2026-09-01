"""Shared strict parsing helpers."""

from __future__ import annotations

import re
from datetime import date, datetime
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from ufc_ml_latestdatafetcher.errors import SourceParseError

_SOURCE_ID = re.compile(r"^[0-9a-f]{16}$")


def soup_for(html: str, *, page_name: str) -> BeautifulSoup:
    if "checking your browser" in html.casefold() or "<title>loading" in html.casefold():
        raise SourceParseError(f"{page_name} HTML is a browser challenge, not source data")
    soup = BeautifulSoup(html, "lxml")
    if soup.title and "stats | ufc" not in soup.title.get_text(" ", strip=True).casefold():
        raise SourceParseError(f"unexpected {page_name} page title")
    return soup


def clean_text(value: str) -> str:
    return " ".join(value.replace("\xa0", " ").split())


def absolute_url(url: str, *, base_url: str) -> str:
    joined = urljoin(f"{base_url.rstrip('/')}/", url.strip())
    return joined.replace("https://ufcstats.com", base_url)


def source_id_from_url(url: str, *, resource: str) -> str:
    parts = [part for part in urlparse(url).path.split("/") if part]
    try:
        index = parts.index(resource)
        value = parts[index + 1]
    except (ValueError, IndexError) as exc:
        raise SourceParseError(f"could not extract {resource} ID from {url!r}") from exc
    if not _SOURCE_ID.fullmatch(value):
        raise SourceParseError(f"invalid UFCStats source ID {value!r} in {url!r}")
    return value


def parse_ufcstats_date(value: str, *, field_name: str = "date") -> date:
    normalized = clean_text(value)
    for pattern in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(normalized, pattern).date()
        except ValueError:
            continue
    raise SourceParseError(f"invalid {field_name}: {value!r}")


def optional_int(value: str) -> int | None:
    normalized = clean_text(value)
    if normalized in {"", "--", "---"}:
        return None
    try:
        return int(normalized)
    except ValueError as exc:
        raise SourceParseError(f"expected integer, got {value!r}") from exc


def optional_float(value: str) -> float | None:
    normalized = clean_text(value)
    if normalized in {"", "--", "---"}:
        return None
    try:
        return float(normalized)
    except ValueError as exc:
        raise SourceParseError(f"expected number, got {value!r}") from exc


def parse_landed_attempted(value: str) -> tuple[int | None, int | None]:
    normalized = clean_text(value)
    if normalized in {"", "--", "---"}:
        return None, None
    match = re.fullmatch(r"(\d+)\s+of\s+(\d+)", normalized, flags=re.IGNORECASE)
    if match is None:
        raise SourceParseError(f"expected 'landed of attempted', got {value!r}")
    landed, attempted = (int(match.group(1)), int(match.group(2)))
    if landed > attempted:
        raise SourceParseError(f"landed exceeds attempted in {value!r}")
    return landed, attempted


def parse_clock_seconds(value: str) -> int | None:
    normalized = clean_text(value)
    if normalized in {"", "--", "---"}:
        return None
    match = re.fullmatch(r"(\d+):(\d{2})", normalized)
    if match is None or int(match.group(2)) >= 60:
        raise SourceParseError(f"invalid clock value {value!r}")
    return int(match.group(1)) * 60 + int(match.group(2))


__all__ = [
    "absolute_url",
    "clean_text",
    "optional_float",
    "optional_int",
    "parse_clock_seconds",
    "parse_landed_attempted",
    "parse_ufcstats_date",
    "soup_for",
    "source_id_from_url",
]
