"""Event detail parsing and fight discovery."""

from __future__ import annotations

from ufc_ml_latestdatafetcher.errors import SourceParseError
from ufc_ml_latestdatafetcher.models import (
    EventRecord,
    FighterReference,
    FightReference,
    ParsedEvent,
)
from ufc_ml_latestdatafetcher.parsing.common import (
    absolute_url,
    clean_text,
    parse_ufcstats_date,
    soup_for,
    source_id_from_url,
)


def _metadata(soup: object) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in soup.select("ul.b-list__box-list li.b-list__box-list-item"):  # type: ignore[attr-defined]
        label = item.select_one(".b-list__box-item-title")
        if label is None:
            continue
        key = clean_text(label.get_text(" ", strip=True)).rstrip(":").casefold()
        label.extract()
        result[key] = clean_text(item.get_text(" ", strip=True))
    return result


def parse_event(
    html: str,
    *,
    event_url: str,
    base_url: str,
    fallback: EventRecord | None = None,
) -> ParsedEvent:
    soup = soup_for(html, page_name="event")
    event_id = source_id_from_url(event_url, resource="event-details")
    title_node = soup.select_one("h2.b-content__title .b-content__title-highlight")
    if title_node is None:
        raise SourceParseError(f"event {event_id} has no title")
    metadata = _metadata(soup)
    date_text = metadata.get("date")
    if date_text is None and fallback is None:
        raise SourceParseError(f"event {event_id} has no date")
    if date_text is not None:
        event_date = parse_ufcstats_date(date_text, field_name="event date")
    else:
        assert fallback is not None
        event_date = fallback.event_date
    location = metadata.get("location", fallback.location if fallback else "")
    event = EventRecord(
        event_id=event_id,
        event_name=clean_text(title_node.get_text(" ", strip=True)),
        event_date=event_date,
        location=location,
        event_url=absolute_url(event_url, base_url=base_url),
    )

    fights: list[FightReference] = []
    seen: set[str] = set()
    rows = soup.select(
        "table.b-fight-details__table_type_event-details tr[data-link*='/fight-details/']"
    )
    for bout_order, row in enumerate(rows, start=1):
        raw_url = row.get("data-link")
        if not isinstance(raw_url, str):
            continue
        fight_url = absolute_url(raw_url, base_url=base_url)
        fight_id = source_id_from_url(fight_url, resource="fight-details")
        if fight_id in seen:
            raise SourceParseError(f"event {event_id} contains duplicate fight {fight_id}")
        fighter_anchors = row.select("a[href*='/fighter-details/']")
        unique: list[FighterReference] = []
        for anchor in fighter_anchors:
            href = anchor.get("href")
            if not isinstance(href, str):
                continue
            fighter_url = absolute_url(href, base_url=base_url)
            reference = FighterReference(
                fighter_id=source_id_from_url(fighter_url, resource="fighter-details"),
                fighter_name=clean_text(anchor.get_text(" ", strip=True)),
                fighter_url=fighter_url,
            )
            if reference.fighter_id not in {item.fighter_id for item in unique}:
                unique.append(reference)
        if len(unique) != 2:
            raise SourceParseError(f"fight {fight_id} does not have exactly two fighter IDs")
        fights.append(
            FightReference(
                fight_id=fight_id,
                fight_url=fight_url,
                bout_order=bout_order,
                fighters=(unique[0], unique[1]),
            )
        )
        seen.add(fight_id)
    if not fights:
        raise SourceParseError(f"event {event_id} contains no fight links")
    return ParsedEvent(event=event, fights=tuple(fights))


__all__ = ["parse_event"]
