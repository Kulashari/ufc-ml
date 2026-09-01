"""Completed-event index parsing."""

from __future__ import annotations

from ufc_ml_latestdatafetcher.errors import SourceParseError
from ufc_ml_latestdatafetcher.models import EventRecord
from ufc_ml_latestdatafetcher.parsing.common import (
    absolute_url,
    clean_text,
    parse_ufcstats_date,
    soup_for,
    source_id_from_url,
)


def parse_completed_events(html: str, *, base_url: str) -> tuple[EventRecord, ...]:
    """Parse every dated event link; callers decide the inclusive date window."""

    soup = soup_for(html, page_name="completed-events")
    records: list[EventRecord] = []
    seen: set[str] = set()
    for row in soup.select("tr.b-statistics__table-row, tr.b-statistics__table-row_type_first"):
        anchor = row.select_one("a[href*='/event-details/']")
        if anchor is None:
            continue
        href = anchor.get("href")
        if not isinstance(href, str):
            continue
        url = absolute_url(href, base_url=base_url)
        event_id = source_id_from_url(url, resource="event-details")
        if event_id in seen:
            raise SourceParseError(f"duplicate event ID on completed-events page: {event_id}")
        date_node = row.select_one(".b-statistics__date")
        cells = row.find_all("td", recursive=False)
        if date_node is None or len(cells) < 2:
            raise SourceParseError(f"event row {event_id} is missing date or location")
        records.append(
            EventRecord(
                event_id=event_id,
                event_name=clean_text(anchor.get_text(" ", strip=True)),
                event_date=parse_ufcstats_date(date_node.get_text(" ", strip=True)),
                location=clean_text(cells[1].get_text(" ", strip=True)),
                event_url=url,
            )
        )
        seen.add(event_id)
    if not records:
        raise SourceParseError("completed-events page contains no valid event rows")
    return tuple(records)


__all__ = ["parse_completed_events"]
