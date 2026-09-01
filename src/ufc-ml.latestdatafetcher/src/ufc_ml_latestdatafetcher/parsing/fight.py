"""Detailed fight parsing, including lossless totals and optional round statistics."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from bs4.element import Tag

from ufc_ml_latestdatafetcher.errors import SourceParseError
from ufc_ml_latestdatafetcher.models import (
    EventRecord,
    FighterFightStats,
    FighterReference,
    FightRecord,
    FightReference,
    ParsedFight,
    RoundStats,
)
from ufc_ml_latestdatafetcher.parsing.common import (
    absolute_url,
    clean_text,
    optional_int,
    parse_clock_seconds,
    parse_landed_attempted,
    soup_for,
    source_id_from_url,
)

_TERMINAL_STATUS_PAIRS = frozenset({("W", "L"), ("L", "W"), ("D", "D"), ("NC", "NC")})
_REGULAR_FIELDS = (
    "kd",
    "sig_str",
    "sig_str_percentage",
    "total_str",
    "td",
    "td_percentage",
    "sub_attempts",
    "reversals",
    "control_seconds",
)
_SIGNIFICANT_FIELDS = (
    "sig_str",
    "sig_str_percentage",
    "head",
    "body",
    "leg",
    "distance",
    "clinch",
    "ground",
)


def _paired_text(cell: Tag, *, fight_id: str) -> tuple[str, str]:
    paragraphs = cell.find_all("p", recursive=False)
    if len(paragraphs) != 2:
        paragraphs = cell.select("p.b-fight-details__table-text")
    values = tuple(clean_text(item.get_text(" ", strip=True)) for item in paragraphs)
    if len(values) != 2:
        raise SourceParseError(
            f"fight {fight_id} stats cell has {len(values)} fighter values instead of two"
        )
    return values[0], values[1]


def _metadata(soup: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    selector = ".b-fight-details__text-item_first, .b-fight-details__text-item"
    for item in soup.select(selector):
        label = item.select_one(".b-fight-details__text-item-label, .b-fight-details__label")
        if label is None:
            continue
        key = clean_text(label.get_text(" ", strip=True)).rstrip(":").casefold()
        if key == "details":
            details_parent = item.find_parent("p", class_="b-fight-details__text")
            if details_parent is not None:
                details = clean_text(details_parent.get_text(" ", strip=True))
                result[key] = details.removeprefix("Details:").strip()
            continue
        label.extract()
        result[key] = clean_text(item.get_text(" ", strip=True))
    return result


def _schedule(time_format: str) -> tuple[int | None, tuple[int, ...]]:
    match = re.search(r"(\d+)\s*Rnd", time_format, flags=re.IGNORECASE)
    scheduled_rounds = int(match.group(1)) if match else None
    overtime_match = re.search(
        r"\+\s*(?:(\d+)\s*)?OT\b",
        time_format,
        flags=re.IGNORECASE,
    )
    if scheduled_rounds is not None and overtime_match is not None:
        scheduled_rounds += int(overtime_match.group(1) or "1")
    durations_match = re.search(r"\(([^)]+)\)", time_format)
    durations: tuple[int, ...] = ()
    if durations_match:
        raw = [item.strip() for item in durations_match.group(1).split("-")]
        if raw and all(item.isdigit() for item in raw):
            durations = tuple(int(item) * 60 for item in raw)
    if scheduled_rounds is not None and durations and len(durations) != scheduled_rounds:
        raise SourceParseError(
            f"time format period count disagrees with its durations: {time_format!r}"
        )
    return scheduled_rounds, durations


def _total_fight_seconds(end_round: int, end_time: str, time_format: str) -> int:
    elapsed = parse_clock_seconds(end_time)
    if elapsed is None:
        raise SourceParseError("completed fight has no ending clock")
    _, durations = _schedule(time_format)
    if durations and end_round <= len(durations):
        return sum(durations[: end_round - 1]) + elapsed
    if end_round == 1:
        return elapsed
    return (end_round - 1) * 300 + elapsed


def _table_rows(
    soup: Any,
    *,
    fight_id: str,
    expected_fighter_names: tuple[str, str],
) -> tuple[dict[tuple[int | None, int], dict[str, int | None]], bool]:
    partial: dict[tuple[int | None, int], dict[str, int | None]] = defaultdict(dict)
    regular_total_count = 0
    significant_total_count = 0
    regular_round_numbers: set[int] = set()
    significant_round_numbers: set[int] = set()

    # UFCStats places the overall significant-strikes table outside the
    # ``js-fight-section`` section, while the other three tables are inside it.
    # Classless tables are therefore valid and the header signature is authoritative.
    for table in soup.select("table"):
        headers = [clean_text(node.get_text(" ", strip=True)) for node in table.select("thead th")]
        rows = table.select("tbody tr")
        if not headers or not rows:
            continue
        is_regular = len(headers) >= 10 and headers[1].casefold() == "kd"
        is_significant = len(headers) >= 9 and any(
            header.casefold() == "head" for header in headers
        )
        if not is_regular and not is_significant:
            continue
        fields = _REGULAR_FIELDS if is_regular else _SIGNIFICANT_FIELDS
        expected_cells = 10 if is_regular else 9
        round_headers: list[int] = []
        for header in headers:
            match = re.fullmatch(r"Round\s+(\d+)", header, flags=re.IGNORECASE)
            if match is not None:
                round_headers.append(int(match.group(1)))
        is_per_round = bool(round_headers)

        if not is_per_round:
            if len(rows) != 1:
                raise SourceParseError(f"fight {fight_id} overall stats table has multiple rows")
            if is_regular:
                regular_total_count += 1
            else:
                significant_total_count += 1
        else:
            expected_rounds = list(range(1, len(rows) + 1))
            if round_headers != expected_rounds:
                raise SourceParseError(
                    f"fight {fight_id} per-round table headers are {round_headers}, "
                    f"expected {expected_rounds}"
                )

        for row_index, row in enumerate(rows, start=1):
            cells = row.find_all("td", recursive=False)
            if len(cells) != expected_cells:
                raise SourceParseError(
                    f"fight {fight_id} stats table expected {expected_cells} cells, "
                    f"got {len(cells)}"
                )
            values_by_cell = [_paired_text(cell, fight_id=fight_id) for cell in cells]
            parsed_names = tuple(name.casefold() for name in values_by_cell[0])
            expected_names = tuple(name.casefold() for name in expected_fighter_names)
            if parsed_names != expected_names:
                raise SourceParseError(
                    f"fight {fight_id} stats fighter order {values_by_cell[0]} disagrees "
                    f"with page participants {expected_fighter_names}"
                )
            round_number: int | None = None
            if is_per_round:
                round_number = round_headers[row_index - 1]
                explicit_rounds: list[int] = []
                for heading in row.find_all("th", recursive=False):
                    match = re.fullmatch(
                        r"Round\s+(\d+)",
                        clean_text(heading.get_text(" ", strip=True)),
                        flags=re.IGNORECASE,
                    )
                    if match is not None:
                        explicit_rounds.append(int(match.group(1)))
                if explicit_rounds and explicit_rounds != [round_number]:
                    raise SourceParseError(
                        f"fight {fight_id} round row label {explicit_rounds} disagrees "
                        f"with header round {round_number}"
                    )
                target_rounds = regular_round_numbers if is_regular else significant_round_numbers
                if round_number in target_rounds:
                    raise SourceParseError(
                        f"fight {fight_id} contains duplicate round {round_number} "
                        f"in {'regular' if is_regular else 'significant'} stats"
                    )
                target_rounds.add(round_number)
            for corner in (0, 1):
                target = partial[(round_number, corner)]
                for field_name, pair in zip(fields, values_by_cell[1:], strict=True):
                    raw = pair[corner]
                    if field_name in {
                        "sig_str",
                        "total_str",
                        "td",
                        "head",
                        "body",
                        "leg",
                        "distance",
                        "clinch",
                        "ground",
                    }:
                        landed, attempted = parse_landed_attempted(raw)
                        parsed_values = {
                            f"{field_name}_landed": landed,
                            f"{field_name}_attempted": attempted,
                        }
                    elif field_name == "control_seconds":
                        parsed_values = {field_name: parse_clock_seconds(raw)}
                    elif field_name.endswith("percentage"):
                        # Percentages are intentionally not persisted; exact landed/attempted
                        # counts are sufficient and avoid UFCStats rounding artifacts.
                        continue
                    else:
                        parsed_values = {field_name: optional_int(raw)}
                    for name, value in parsed_values.items():
                        if name in target and target[name] != value:
                            raise SourceParseError(
                                f"fight {fight_id} has conflicting {name} values for "
                                f"round {round_number}: {target[name]} != {value}"
                            )
                        target[name] = value

    if regular_total_count != 1 or significant_total_count != 1:
        raise SourceParseError(
            f"fight {fight_id} requires exactly one recognized regular and significant "
            f"overall stats table; found regular={regular_total_count}, "
            f"significant={significant_total_count}"
        )
    if (
        regular_round_numbers
        and significant_round_numbers
        and regular_round_numbers != significant_round_numbers
    ):
        raise SourceParseError(
            f"fight {fight_id} regular/significant round sets disagree: "
            f"{sorted(regular_round_numbers)} != {sorted(significant_round_numbers)}"
        )
    return partial, True


def _value(values: dict[str, int | None], name: str) -> int | None:
    return values.get(name)


def parse_fight(
    html: str,
    *,
    event: EventRecord,
    reference: FightReference,
    base_url: str,
) -> ParsedFight:
    soup = soup_for(html, page_name="fight")
    fight_id = source_id_from_url(reference.fight_url, resource="fight-details")
    person_nodes = soup.select("div.b-fight-details__person")
    fighters: list[FighterReference] = []
    statuses: list[str] = []
    for person in person_nodes:
        anchor = person.select_one("a.b-fight-details__person-link[href*='/fighter-details/']")
        status = person.select_one(".b-fight-details__person-status")
        if anchor is None or status is None:
            continue
        href = anchor.get("href")
        if not isinstance(href, str):
            continue
        fighter_url = absolute_url(href, base_url=base_url)
        fighters.append(
            FighterReference(
                fighter_id=source_id_from_url(fighter_url, resource="fighter-details"),
                fighter_name=clean_text(anchor.get_text(" ", strip=True)),
                fighter_url=fighter_url,
            )
        )
        statuses.append(clean_text(status.get_text(" ", strip=True)).upper())
    if len(fighters) != 2 or len(statuses) != 2:
        raise SourceParseError(f"fight {fight_id} does not have exactly two fighters and statuses")
    status_pair = (statuses[0], statuses[1])
    if status_pair not in _TERMINAL_STATUS_PAIRS:
        raise SourceParseError(f"fight {fight_id} is not terminal: statuses={status_pair}")
    if {item.fighter_id for item in fighters} != {item.fighter_id for item in reference.fighters}:
        raise SourceParseError(f"fight {fight_id} fighter IDs disagree with its event page")

    title_node = soup.select_one(".b-fight-details__fight-title")
    if title_node is None:
        raise SourceParseError(f"fight {fight_id} has no bout descriptor")
    raw_weight_class = clean_text(title_node.get_text(" ", strip=True))
    metadata = _metadata(soup)
    required = ("method", "round", "time", "time format")
    missing = [name for name in required if not metadata.get(name)]
    if missing:
        raise SourceParseError(f"fight {fight_id} is missing metadata: {missing}")
    try:
        end_round = int(metadata["round"])
    except ValueError as exc:
        raise SourceParseError(f"fight {fight_id} has invalid ending round") from exc
    time_format = metadata["time format"]
    scheduled_rounds, round_durations = _schedule(time_format)
    scheduled_duration = (
        sum(round_durations)
        if round_durations
        else (scheduled_rounds * 300 if scheduled_rounds is not None else None)
    )
    winner_index = statuses.index("W") if "W" in statuses else None
    winner = fighters[winner_index] if winner_index is not None else None

    partial, stats_available = _table_rows(
        soup,
        fight_id=fight_id,
        expected_fighter_names=(fighters[0].fighter_name, fighters[1].fighter_name),
    )
    totals: list[FighterFightStats] = []
    rounds: list[RoundStats] = []
    if stats_available:
        for corner, fighter in enumerate(fighters):
            opponent = fighters[1 - corner]
            values = partial[(None, corner)]
            totals.append(
                FighterFightStats(
                    fight_id=fight_id,
                    event_id=event.event_id,
                    event_date=event.event_date,
                    fighter_id=fighter.fighter_id,
                    opponent_id=opponent.fighter_id,
                    fighter_name=fighter.fighter_name,
                    corner=corner + 1,
                    result=statuses[corner],
                    kd=_value(values, "kd"),
                    sig_str_landed=_value(values, "sig_str_landed"),
                    sig_str_attempted=_value(values, "sig_str_attempted"),
                    total_str_landed=_value(values, "total_str_landed"),
                    total_str_attempted=_value(values, "total_str_attempted"),
                    td_landed=_value(values, "td_landed"),
                    td_attempted=_value(values, "td_attempted"),
                    sub_attempts=_value(values, "sub_attempts"),
                    reversals=_value(values, "reversals"),
                    control_seconds=_value(values, "control_seconds"),
                    head_landed=_value(values, "head_landed"),
                    head_attempted=_value(values, "head_attempted"),
                    body_landed=_value(values, "body_landed"),
                    body_attempted=_value(values, "body_attempted"),
                    leg_landed=_value(values, "leg_landed"),
                    leg_attempted=_value(values, "leg_attempted"),
                    distance_landed=_value(values, "distance_landed"),
                    distance_attempted=_value(values, "distance_attempted"),
                    clinch_landed=_value(values, "clinch_landed"),
                    clinch_attempted=_value(values, "clinch_attempted"),
                    ground_landed=_value(values, "ground_landed"),
                    ground_attempted=_value(values, "ground_attempted"),
                )
            )
    round_numbers = sorted(round_number for round_number, _ in partial if round_number is not None)
    for round_number in sorted(set(round_numbers)):
        assert round_number is not None
        for corner, fighter in enumerate(fighters):
            opponent = fighters[1 - corner]
            values = partial[(round_number, corner)]
            rounds.append(
                RoundStats(
                    fight_id=fight_id,
                    event_id=event.event_id,
                    event_date=event.event_date,
                    round_number=round_number,
                    fighter_id=fighter.fighter_id,
                    opponent_id=opponent.fighter_id,
                    fighter_name=fighter.fighter_name,
                    corner=corner + 1,
                    kd=_value(values, "kd"),
                    sig_str_landed=_value(values, "sig_str_landed"),
                    sig_str_attempted=_value(values, "sig_str_attempted"),
                    total_str_landed=_value(values, "total_str_landed"),
                    total_str_attempted=_value(values, "total_str_attempted"),
                    td_landed=_value(values, "td_landed"),
                    td_attempted=_value(values, "td_attempted"),
                    sub_attempts=_value(values, "sub_attempts"),
                    reversals=_value(values, "reversals"),
                    control_seconds=_value(values, "control_seconds"),
                    head_landed=_value(values, "head_landed"),
                    head_attempted=_value(values, "head_attempted"),
                    body_landed=_value(values, "body_landed"),
                    body_attempted=_value(values, "body_attempted"),
                    leg_landed=_value(values, "leg_landed"),
                    leg_attempted=_value(values, "leg_attempted"),
                    distance_landed=_value(values, "distance_landed"),
                    distance_attempted=_value(values, "distance_attempted"),
                    clinch_landed=_value(values, "clinch_landed"),
                    clinch_attempted=_value(values, "clinch_attempted"),
                    ground_landed=_value(values, "ground_landed"),
                    ground_attempted=_value(values, "ground_attempted"),
                )
            )

    lower_title = raw_weight_class.casefold()
    fight = FightRecord(
        fight_id=fight_id,
        event_id=event.event_id,
        event_date=event.event_date,
        bout_order=reference.bout_order,
        fight_url=absolute_url(reference.fight_url, base_url=base_url),
        fighter_1_id=fighters[0].fighter_id,
        fighter_1_name=fighters[0].fighter_name,
        fighter_2_id=fighters[1].fighter_id,
        fighter_2_name=fighters[1].fighter_name,
        fighter_1_status=statuses[0],
        fighter_2_status=statuses[1],
        winner_id=winner.fighter_id if winner else None,
        winner_name=winner.fighter_name if winner else None,
        raw_weight_class=raw_weight_class,
        method=metadata["method"],
        method_details=metadata.get("details", ""),
        end_round=end_round,
        end_time=metadata["time"],
        total_fight_time_sec=_total_fight_seconds(end_round, metadata["time"], time_format),
        time_format=time_format,
        scheduled_rounds=scheduled_rounds,
        scheduled_duration_sec=scheduled_duration,
        referee=metadata.get("referee", ""),
        is_title_bout=int("title bout" in lower_title),
        is_interim_title=int("interim" in lower_title),
        is_tournament_final=int("tournament" in lower_title and "title bout" in lower_title),
        is_superfight=int("superfight" in lower_title),
        stats_available=int(stats_available),
    )
    return ParsedFight(
        fight=fight,
        totals=tuple(totals),
        rounds=tuple(rounds),
        fighters=(fighters[0], fighters[1]),
    )


__all__ = ["parse_fight"]
