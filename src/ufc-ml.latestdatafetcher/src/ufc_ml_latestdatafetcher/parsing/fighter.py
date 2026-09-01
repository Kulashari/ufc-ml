"""Fighter directory and stable profile parsing."""

from __future__ import annotations

import re
from datetime import datetime

from ufc_ml_latestdatafetcher.errors import SourceParseError
from ufc_ml_latestdatafetcher.models import FighterProfile, FighterReference
from ufc_ml_latestdatafetcher.parsing.common import (
    absolute_url,
    clean_text,
    optional_float,
    soup_for,
    source_id_from_url,
)


def _height_inches(value: str) -> float | None:
    normalized = clean_text(value)
    if normalized in {"", "--", "---"}:
        return None
    match = re.fullmatch(r"(\d+)\s*'\s*(\d+(?:\.\d+)?)\s*\"", normalized)
    if match is None:
        raise SourceParseError(f"invalid fighter height {value!r}")
    return int(match.group(1)) * 12 + float(match.group(2))


def _inches(value: str) -> float | None:
    normalized = clean_text(value)
    if normalized in {"", "--", "---"}:
        return None
    return optional_float(normalized.rstrip('"'))


def _pounds(value: str) -> float | None:
    normalized = clean_text(value)
    if normalized in {"", "--", "---"}:
        return None
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*lbs?\.?", normalized, flags=re.IGNORECASE)
    if match is None:
        raise SourceParseError(f"invalid fighter weight {value!r}")
    return float(match.group(1))


def _percentage(value: str) -> float | None:
    normalized = clean_text(value)
    if normalized in {"", "--", "---"}:
        return None
    if not normalized.endswith("%"):
        raise SourceParseError(f"invalid percentage {value!r}")
    number = optional_float(normalized[:-1])
    return None if number is None else number / 100.0


def _record(value: str) -> tuple[int | None, int | None, int | None, int | None]:
    normalized = clean_text(value)
    match = re.search(r"Record:\s*(\d+)-(\d+)-(\d+)", normalized, flags=re.IGNORECASE)
    if match is None:
        return None, None, None, None
    nc_match = re.search(r"\((\d+)\s+NC\)", normalized, flags=re.IGNORECASE)
    return (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3)),
        int(nc_match.group(1)) if nc_match else 0,
    )


def parse_fighter_directory(
    html: str,
    *,
    base_url: str,
) -> tuple[FighterReference, ...]:
    soup = soup_for(html, page_name="fighter-directory")
    references: list[FighterReference] = []
    seen: set[str] = set()
    for row in soup.select("tr.b-statistics__table-row"):
        anchors = row.select("a[href*='/fighter-details/']")
        if not anchors:
            continue
        href = anchors[0].get("href")
        if not isinstance(href, str):
            continue
        fighter_url = absolute_url(href, base_url=base_url)
        fighter_id = source_id_from_url(fighter_url, resource="fighter-details")
        if fighter_id in seen:
            raise SourceParseError(f"duplicate fighter {fighter_id} in directory page")
        cells = row.find_all("td", recursive=False)
        if len(cells) < 2:
            raise SourceParseError(f"directory fighter {fighter_id} is missing name columns")
        first = clean_text(cells[0].get_text(" ", strip=True))
        last = clean_text(cells[1].get_text(" ", strip=True))
        name = clean_text(f"{first} {last}")
        if not name:
            raise SourceParseError(f"directory fighter {fighter_id} has no name")
        references.append(
            FighterReference(
                fighter_id=fighter_id,
                fighter_name=name,
                fighter_url=fighter_url,
            )
        )
        seen.add(fighter_id)
    if not references:
        raise SourceParseError("fighter-directory page contains no fighter links")
    return tuple(references)


def parse_fighter_profile(html: str, *, fighter_url: str, base_url: str) -> FighterProfile:
    soup = soup_for(html, page_name="fighter")
    fighter_url = absolute_url(fighter_url, base_url=base_url)
    fighter_id = source_id_from_url(fighter_url, resource="fighter-details")
    name_node = soup.select_one(".b-content__title-highlight")
    if name_node is None:
        raise SourceParseError(f"fighter {fighter_id} has no name")
    record_node = soup.select_one(".b-content__title-record")
    wins, losses, draws, no_contests = _record(
        record_node.get_text(" ", strip=True) if record_node else ""
    )
    nickname_node = soup.select_one(".b-content__Nickname")

    values: dict[str, str] = {}
    profile_lists = soup.select("ul.b-list__box-list")
    if not profile_lists:
        raise SourceParseError(f"fighter {fighter_id} has no bio list")
    for profile_list in profile_lists:
        for item in profile_list.select("li.b-list__box-list-item"):
            label = item.select_one(".b-list__box-item-title")
            if label is None:
                continue
            key = clean_text(label.get_text(" ", strip=True)).rstrip(":").casefold()
            if not key:
                continue
            label.extract()
            value = clean_text(item.get_text(" ", strip=True))
            previous = values.get(key)
            if previous is not None and previous != value:
                raise SourceParseError(
                    f"fighter {fighter_id} has conflicting profile values for {key!r}"
                )
            values[key] = value

    dob = None
    raw_dob = values.get("dob", "")
    if raw_dob not in {"", "--", "---"}:
        try:
            dob = datetime.strptime(raw_dob, "%b %d, %Y").date()
        except ValueError as exc:
            raise SourceParseError(f"fighter {fighter_id} has invalid DOB {raw_dob!r}") from exc

    return FighterProfile(
        fighter_id=fighter_id,
        fighter_name=clean_text(name_node.get_text(" ", strip=True)),
        nickname=(clean_text(nickname_node.get_text(" ", strip=True)) if nickname_node else ""),
        height_inches=_height_inches(values.get("height", "")),
        weight_lbs=_pounds(values.get("weight", "")),
        reach_inches=_inches(values.get("reach", "")),
        stance=(values.get("stance", "") or "Unknown").title(),
        dob=dob,
        wins=wins,
        losses=losses,
        draws=draws,
        no_contests=no_contests,
        # UFCStats career summaries are current as of fetch time. They are retained
        # for provenance/inspection, not for reconstructing historical snapshots.
        slpm_current=optional_float(values.get("slpm", "")),
        striking_accuracy_current=_percentage(values.get("str. acc.", "")),
        sapm_current=optional_float(values.get("sapm", "")),
        striking_defense_current=_percentage(values.get("str. def", "")),
        takedown_average_current=optional_float(values.get("td avg.", "")),
        takedown_accuracy_current=_percentage(values.get("td acc.", "")),
        takedown_defense_current=_percentage(values.get("td def.", "")),
        submission_average_current=optional_float(values.get("sub. avg.", "")),
        fighter_url=fighter_url,
    )


__all__ = ["parse_fighter_directory", "parse_fighter_profile"]
