"""Integrity validation for normalized crawler data.

Feature-to-source mapping and crawl-data validity are deliberately reported as
separate contracts. A complete static mapping does not prove that a particular
repository contains usable source rows.
"""

from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import fields
from datetime import date, datetime
from pathlib import Path
from typing import Any

from ufc_ml_latestdatafetcher.errors import DatasetValidationError
from ufc_ml_latestdatafetcher.feature_coverage import build_feature_coverage_report
from ufc_ml_latestdatafetcher.models import (
    EventRecord,
    FighterFightStats,
    FighterProfile,
    FighterReference,
    FightRecord,
    RoundStats,
    SourcePageRecord,
)
from ufc_ml_latestdatafetcher.storage import LocalRepository, read_csv_rows
from ufc_ml_latestdatafetcher.training_contract import is_legacy_label_eligible

_LANDED_ATTEMPTED = (
    ("sig_str_landed", "sig_str_attempted"),
    ("total_str_landed", "total_str_attempted"),
    ("td_landed", "td_attempted"),
    ("head_landed", "head_attempted"),
    ("body_landed", "body_attempted"),
    ("leg_landed", "leg_attempted"),
    ("distance_landed", "distance_attempted"),
    ("clinch_landed", "clinch_attempted"),
    ("ground_landed", "ground_attempted"),
)
_COUNT_FIELDS = ("kd", "sub_attempts", "reversals", "control_seconds")
_SUM_FIELDS = tuple(
    dict.fromkeys(
        [
            name
            for landed_name, attempted_name in _LANDED_ATTEMPTED
            for name in (landed_name, attempted_name)
        ]
        + list(_COUNT_FIELDS)
    )
)
_TERMINAL_STATUS_PAIRS = frozenset({("W", "L"), ("L", "W"), ("D", "D"), ("NC", "NC")})
_FIGHTER_DIRECTORY_LETTERS = frozenset("abcdefghijklmnopqrstuvwxyz")
_SOURCE_ID = re.compile(r"^[0-9a-f]{16}$")
_SOURCE_PAGE_KINDS = frozenset(
    {"completed_events", "fighter_directory", "event", "fight", "fighter"}
)


def _duplicates(rows: list[dict[str, str]], keys: tuple[str, ...]) -> list[tuple[str, ...]]:
    counts = Counter(tuple(row.get(key, "") for key in keys) for row in rows)
    return [key for key, count in counts.items() if count > 1]


def _validate_exact_schema(
    path: Path,
    record_type: type[Any],
    *,
    table_name: str,
    required: bool,
) -> None:
    if not path.is_file():
        if required:
            raise DatasetValidationError(f"required normalized table is absent: {path}")
        return
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            actual = tuple(csv.DictReader(handle).fieldnames or ())
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise DatasetValidationError(f"could not read schema for {table_name}: {exc}") from exc
    expected = tuple(field.name for field in fields(record_type))
    if actual != expected:
        missing = [name for name in expected if name not in actual]
        unexpected = [name for name in actual if name not in expected]
        raise DatasetValidationError(
            f"{table_name} schema differs from the normalized contract; "
            f"missing={missing}, unexpected={unexpected}, order_matches={actual == expected}"
        )


def _required_values(
    row: dict[str, str],
    names: Iterable[str],
    *,
    table_name: str,
    identity: str,
) -> None:
    missing = [name for name in names if not str(row.get(name, "")).strip()]
    if missing:
        raise DatasetValidationError(
            f"{table_name} row {identity} has blank required values: {missing}"
        )


def _nonnegative_int(value: str, *, table_name: str, identity: str, field: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise DatasetValidationError(
            f"{table_name} row {identity} has invalid integer {field}={value!r}"
        ) from exc
    if parsed < 0:
        raise DatasetValidationError(f"{table_name} row {identity} has negative {field}={parsed}")
    return parsed


def _validate_stats(rows: list[dict[str, str]], *, table_name: str) -> None:
    for row_number, row in enumerate(rows, start=2):
        identity = f"line {row_number}"
        for landed_name, attempted_name in _LANDED_ATTEMPTED:
            landed_raw = row.get(landed_name, "")
            attempted_raw = row.get(attempted_name, "")
            if landed_raw == "" and attempted_raw == "":
                continue
            if landed_raw == "" or attempted_raw == "":
                raise DatasetValidationError(
                    f"{table_name}:{row_number} has only one of {landed_name}/{attempted_name}"
                )
            landed = _nonnegative_int(
                landed_raw, table_name=table_name, identity=identity, field=landed_name
            )
            attempted = _nonnegative_int(
                attempted_raw, table_name=table_name, identity=identity, field=attempted_name
            )
            if landed > attempted:
                raise DatasetValidationError(
                    f"{table_name}:{row_number} has invalid {landed_name}/{attempted_name}"
                )
        for name in _COUNT_FIELDS:
            raw = row.get(name, "")
            if raw:
                _nonnegative_int(raw, table_name=table_name, identity=identity, field=name)


def _validate_source_pages(rows: list[dict[str, str]]) -> None:
    for row_number, row in enumerate(rows, start=2):
        identity = f"line {row_number}"
        _required_values(
            row,
            (
                "page_kind",
                "source_id",
                "url",
                "fetched_at_utc",
                "sha256",
                "cache_path",
                "parser_version",
            ),
            table_name="source_pages.csv",
            identity=identity,
        )
        expected_sha256 = row["sha256"].strip().casefold()
        page_kind = row["page_kind"]
        source_id = row["source_id"]
        if page_kind not in _SOURCE_PAGE_KINDS:
            raise DatasetValidationError(
                f"source_pages.csv:{row_number} has unknown page_kind={page_kind!r}"
            )
        if page_kind == "completed_events" and source_id != "completed_all":
            raise DatasetValidationError(
                f"source_pages.csv:{row_number} has invalid completed index ID {source_id!r}"
            )
        if page_kind == "fighter_directory" and source_id not in _FIGHTER_DIRECTORY_LETTERS:
            raise DatasetValidationError(
                f"source_pages.csv:{row_number} has invalid directory ID {source_id!r}"
            )
        if page_kind in {"event", "fight", "fighter"} and _SOURCE_ID.fullmatch(source_id) is None:
            raise DatasetValidationError(
                f"source_pages.csv:{row_number} has invalid source ID {source_id!r}"
            )
        try:
            fetched_at = datetime.fromisoformat(row["fetched_at_utc"])
        except ValueError as exc:
            raise DatasetValidationError(
                f"source_pages.csv:{row_number} has invalid fetched_at_utc"
            ) from exc
        if fetched_at.tzinfo is None:
            raise DatasetValidationError(
                f"source_pages.csv:{row_number} fetched_at_utc has no timezone"
            )
        if len(expected_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha256
        ):
            raise DatasetValidationError(
                f"source_pages.csv:{row_number} has invalid sha256={row['sha256']!r}"
            )
        cache_path = Path(row["cache_path"]).expanduser()
        if not cache_path.is_file():
            raise DatasetValidationError(
                f"source_pages.csv:{row_number} cache file is absent: {cache_path}"
            )
        try:
            # CachedBrowserClient fingerprints the decoded HTML content. Reading as
            # text also avoids platform newline translation changing the digest.
            html = cache_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise DatasetValidationError(
                f"source_pages.csv:{row_number} could not read {cache_path}: {exc}"
            ) from exc
        actual_sha256 = hashlib.sha256(html.encode("utf-8")).hexdigest()
        if actual_sha256 != expected_sha256:
            raise DatasetValidationError(
                f"source_pages.csv:{row_number} SHA-256 mismatch for {cache_path}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        if row.get("from_cache", "") not in {"0", "1"}:
            raise DatasetValidationError(
                f"source_pages.csv:{row_number} has invalid from_cache={row.get('from_cache')!r}"
            )


def _validate_status_and_winner(fight: dict[str, str]) -> None:
    fight_id = fight["fight_id"]
    statuses = (fight["fighter_1_status"], fight["fighter_2_status"])
    if statuses not in _TERMINAL_STATUS_PAIRS:
        raise DatasetValidationError(f"fight {fight_id} has non-terminal statuses {statuses}")
    expected_id = ""
    expected_name = ""
    if statuses[0] == "W":
        expected_id = fight["fighter_1_id"]
        expected_name = fight["fighter_1_name"]
    elif statuses[1] == "W":
        expected_id = fight["fighter_2_id"]
        expected_name = fight["fighter_2_name"]
    if fight.get("winner_id", "") != expected_id or fight.get("winner_name", "") != expected_name:
        raise DatasetValidationError(
            f"fight {fight_id} winner fields disagree with statuses {statuses}"
        )


def _validate_participant_row(
    row: dict[str, str],
    *,
    fight: dict[str, str],
    table_name: str,
) -> None:
    fight_id = fight["fight_id"]
    fighter_id = row["fighter_id"]
    first_id = fight["fighter_1_id"]
    second_id = fight["fighter_2_id"]
    if fighter_id == first_id:
        expected = {
            "opponent_id": second_id,
            "fighter_name": fight["fighter_1_name"],
            "corner": "1",
        }
        if "result" in row:
            expected["result"] = fight["fighter_1_status"]
    elif fighter_id == second_id:
        expected = {
            "opponent_id": first_id,
            "fighter_name": fight["fighter_2_name"],
            "corner": "2",
        }
        if "result" in row:
            expected["result"] = fight["fighter_2_status"]
    else:
        raise DatasetValidationError(
            f"{table_name} row references non-participant {fighter_id} in fight {fight_id}"
        )
    expected.update({"event_id": fight["event_id"], "event_date": fight["event_date"]})
    disagreements = {
        name: (row.get(name, ""), value)
        for name, value in expected.items()
        if row.get(name, "") != value
    }
    if disagreements:
        raise DatasetValidationError(
            f"{table_name} row for fight {fight_id}, fighter {fighter_id} disagrees with "
            f"fight data: {disagreements}"
        )


def _reconcile_totals_and_rounds(
    fight: dict[str, str],
    totals: list[dict[str, str]],
    rounds: list[dict[str, str]],
) -> bool:
    """Reconcile totals when every completed round has both participant rows."""

    if not rounds:
        return False
    fight_id = fight["fight_id"]
    end_round = _nonnegative_int(
        fight["end_round"],
        table_name="fights.csv",
        identity=fight_id,
        field="end_round",
    )
    rounds_by_number: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rounds:
        round_number = _nonnegative_int(
            row["round_number"],
            table_name="round_stats.csv",
            identity=f"{fight_id}/{row.get('fighter_id', '')}",
            field="round_number",
        )
        if round_number < 1 or round_number > end_round:
            raise DatasetValidationError(
                f"fight {fight_id} has out-of-range round {round_number}; end_round={end_round}"
            )
        rounds_by_number[round_number].append(row)
    expected_rounds = set(range(1, end_round + 1))
    if set(rounds_by_number) != expected_rounds or any(
        len(pair) != 2 for pair in rounds_by_number.values()
    ):
        return False

    round_by_fighter: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rounds:
        round_by_fighter[row["fighter_id"]].append(row)
    for total in totals:
        fighter_id = total["fighter_id"]
        for field in _SUM_FIELDS:
            total_raw = total.get(field, "")
            round_values = [row.get(field, "") for row in round_by_fighter[fighter_id]]
            # Older cards can expose significant-strike rounds without regular
            # per-round fields such as KD/TD/CTRL. Overall totals are authoritative;
            # compare a metric only when it is present in every completed round.
            if total_raw == "" or any(value == "" for value in round_values):
                continue
            total_value = _nonnegative_int(
                total_raw,
                table_name="bout_fighter_totals.csv",
                identity=f"{fight_id}/{fighter_id}",
                field=field,
            )
            round_sum = sum(
                _nonnegative_int(
                    value,
                    table_name="round_stats.csv",
                    identity=f"{fight_id}/{fighter_id}",
                    field=field,
                )
                for value in round_values
            )
            if total_value != round_sum:
                raise DatasetValidationError(
                    f"fight {fight_id}, fighter {fighter_id} has {field} total "
                    f"{total_value}, but rounds sum to {round_sum}"
                )
    return True


def validate_local_repository(
    repository: LocalRepository,
    *,
    feature_dictionary_path: Path,
    expected_feature_count: int,
    expected_event_ids: Iterable[str] | None = None,
    require_fighter_directory_complete: bool = False,
    minimum_fighter_directory_count: int = 1,
) -> dict[str, Any]:
    """Validate normalized data separately from the static feature-source mapping.

    ``expected_event_ids`` is a required subset, allowing an incremental repository
    to retain successful events from earlier runs. Directory completeness is optional
    because bounded smoke crawls intentionally skip or limit that graph.
    """

    repository.materialize_csvs()
    schemas = (
        (repository.events_path, EventRecord, "events.csv", True),
        (repository.fights_path, FightRecord, "fights.csv", True),
        (repository.totals_path, FighterFightStats, "bout_fighter_totals.csv", True),
        (repository.rounds_path, RoundStats, "round_stats.csv", False),
        (repository.fighters_path, FighterProfile, "fighters.csv", True),
        (
            repository.fighter_index_path,
            FighterReference,
            "fighter_index.csv",
            require_fighter_directory_complete,
        ),
        (repository.source_pages_path, SourcePageRecord, "source_pages.csv", True),
    )
    for path, record_type, table_name, required in schemas:
        _validate_exact_schema(path, record_type, table_name=table_name, required=required)

    events = read_csv_rows(repository.events_path)
    fights = read_csv_rows(repository.fights_path)
    totals = read_csv_rows(repository.totals_path)
    rounds = read_csv_rows(repository.rounds_path)
    fighters = read_csv_rows(repository.fighters_path)
    fighter_index = read_csv_rows(repository.fighter_index_path)
    sources = read_csv_rows(repository.source_pages_path)
    for table_name, rows in (
        ("events.csv", events),
        ("fights.csv", fights),
        ("bout_fighter_totals.csv", totals),
    ):
        if not rows:
            raise DatasetValidationError(f"{table_name} is empty")

    for name, rows, keys in (
        ("events", events, ("event_id",)),
        ("fights", fights, ("fight_id",)),
        ("fights", fights, ("event_id", "bout_order")),
        ("bout_fighter_totals", totals, ("fight_id", "fighter_id")),
        ("round_stats", rounds, ("fight_id", "round_number", "fighter_id")),
        ("fighters", fighters, ("fighter_id",)),
        ("fighter_index", fighter_index, ("fighter_id",)),
        ("source_pages", sources, ("page_kind", "source_id", "sha256")),
    ):
        duplicates = _duplicates(rows, keys)
        if duplicates:
            raise DatasetValidationError(f"{name} contains duplicate keys: {duplicates[:5]}")

    event_by_id = {row["event_id"]: row for row in events}
    fighter_ids = {row["fighter_id"] for row in fighters}
    indexed_fighter_ids = {row["fighter_id"] for row in fighter_index}
    fight_by_id = {row["fight_id"]: row for row in fights}
    fights_by_event: dict[str, list[dict[str, str]]] = defaultdict(list)
    totals_by_fight: dict[str, list[dict[str, str]]] = defaultdict(list)
    rounds_by_fight: dict[str, list[dict[str, str]]] = defaultdict(list)

    for event_id, event_row in event_by_id.items():
        if _SOURCE_ID.fullmatch(event_id) is None:
            raise DatasetValidationError(f"events.csv has invalid event_id={event_id!r}")
        _required_values(
            event_row,
            ("event_id", "event_name", "event_date", "location", "event_url"),
            table_name="events.csv",
            identity=event_id,
        )
        try:
            date.fromisoformat(event_row["event_date"])
        except ValueError as exc:
            raise DatasetValidationError(
                f"events.csv row {event_id} has invalid event_date"
            ) from exc
    for fighter in fighters:
        if _SOURCE_ID.fullmatch(fighter.get("fighter_id", "")) is None:
            raise DatasetValidationError(
                f"fighters.csv has invalid fighter_id={fighter.get('fighter_id')!r}"
            )
        _required_values(
            fighter,
            ("fighter_id", "fighter_name", "fighter_url", "profile_origin"),
            table_name="fighters.csv",
            identity=fighter.get("fighter_id", ""),
        )
    for reference in fighter_index:
        fighter_id = reference.get("fighter_id", "")
        if _SOURCE_ID.fullmatch(fighter_id) is None:
            raise DatasetValidationError(f"fighter_index.csv has invalid fighter_id={fighter_id!r}")
        _required_values(
            reference,
            ("fighter_id", "fighter_name", "fighter_url"),
            table_name="fighter_index.csv",
            identity=fighter_id,
        )

    for row in totals:
        if row["fight_id"] not in fight_by_id:
            raise DatasetValidationError(f"totals row references unknown fight {row['fight_id']}")
        totals_by_fight[row["fight_id"]].append(row)
    for row in rounds:
        if row["fight_id"] not in fight_by_id:
            raise DatasetValidationError(f"round row references unknown fight {row['fight_id']}")
        rounds_by_fight[row["fight_id"]].append(row)

    reconciled_fights = 0
    for fight_id, fight in fight_by_id.items():
        if _SOURCE_ID.fullmatch(fight_id) is None:
            raise DatasetValidationError(f"fights.csv has invalid fight_id={fight_id!r}")
        try:
            date.fromisoformat(fight["event_date"])
        except ValueError as exc:
            raise DatasetValidationError(
                f"fights.csv row {fight_id} has invalid event_date"
            ) from exc
        _required_values(
            fight,
            (
                "fight_id",
                "event_id",
                "event_date",
                "fight_url",
                "fighter_1_id",
                "fighter_1_name",
                "fighter_2_id",
                "fighter_2_name",
                "fighter_1_status",
                "fighter_2_status",
                "raw_weight_class",
                "method",
                "end_round",
                "end_time",
                "total_fight_time_sec",
                "time_format",
            ),
            table_name="fights.csv",
            identity=fight_id,
        )
        referenced_event = event_by_id.get(fight["event_id"])
        if referenced_event is None:
            raise DatasetValidationError(f"fight {fight_id} references an unknown event")
        if fight["event_date"] != referenced_event["event_date"]:
            raise DatasetValidationError(f"fight {fight_id} date disagrees with its event")
        fights_by_event[fight["event_id"]].append(fight)
        participants = {fight["fighter_1_id"], fight["fighter_2_id"]}
        if len(participants) != 2:
            raise DatasetValidationError(f"fight {fight_id} does not have two distinct fighters")
        missing_profiles = participants - fighter_ids
        if missing_profiles:
            raise DatasetValidationError(
                f"fight {fight_id} has participants without profiles: {sorted(missing_profiles)}"
            )
        _validate_status_and_winner(fight)

        stats_available = fight.get("stats_available", "")
        if stats_available not in {"0", "1"}:
            raise DatasetValidationError(
                f"fight {fight_id} has invalid stats_available={stats_available!r}"
            )
        fight_totals = totals_by_fight.get(fight_id, [])
        if stats_available == "1":
            totals_fighter_ids = {row["fighter_id"] for row in fight_totals}
            if len(fight_totals) != 2 or totals_fighter_ids != participants:
                raise DatasetValidationError(
                    f"fight {fight_id} does not have exactly two participant totals rows"
                )
        elif fight_totals:
            raise DatasetValidationError(f"fight {fight_id} has totals despite stats_available=0")
        for row in fight_totals:
            _validate_participant_row(row, fight=fight, table_name="bout_fighter_totals.csv")

        fight_rounds = rounds_by_fight.get(fight_id, [])
        rounds_by_number: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in fight_rounds:
            _validate_participant_row(row, fight=fight, table_name="round_stats.csv")
            rounds_by_number[row["round_number"]].append(row)
        for round_number, pair in rounds_by_number.items():
            if len(pair) != 2 or {row["fighter_id"] for row in pair} != participants:
                raise DatasetValidationError(
                    f"fight {fight_id} round {round_number} does not contain both fighters"
                )
        if fight_totals and _reconcile_totals_and_rounds(fight, fight_totals, fight_rounds):
            reconciled_fights += 1

    empty_events = sorted(set(event_by_id) - set(fights_by_event))
    if empty_events:
        raise DatasetValidationError(f"events contain no fights: {empty_events[:5]}")
    requested_event_ids = {
        str(value).strip() for value in expected_event_ids or () if str(value).strip()
    }
    missing_expected_events = sorted(requested_event_ids - set(event_by_id))
    if missing_expected_events:
        raise DatasetValidationError(
            f"repository is missing expected events: {missing_expected_events[:10]}"
        )

    missing_index_profiles = indexed_fighter_ids - fighter_ids
    directory_page_ids = {
        row["source_id"] for row in sources if row.get("page_kind") == "fighter_directory"
    }
    missing_directory_pages = sorted(_FIGHTER_DIRECTORY_LETTERS - directory_page_ids)
    if require_fighter_directory_complete:
        if not fighter_index:
            raise DatasetValidationError("fighter directory completeness requires a nonempty index")
        if len(fighter_index) < minimum_fighter_directory_count:
            raise DatasetValidationError(
                f"fighter directory has {len(fighter_index)} rows; expected at least "
                f"{minimum_fighter_directory_count}"
            )
        if missing_directory_pages:
            raise DatasetValidationError(
                f"fighter directory source pages are incomplete: {missing_directory_pages}"
            )
        if missing_index_profiles:
            raise DatasetValidationError(
                "fighter directory contains IDs without profiles: "
                f"{sorted(missing_index_profiles)[:10]}"
            )

    _validate_stats(totals, table_name="bout_fighter_totals.csv")
    _validate_stats(rounds, table_name="round_stats.csv")
    _validate_source_pages(sources)
    source_pairs = {(row["page_kind"], row["source_id"]) for row in sources}
    source_triples = {(row["page_kind"], row["source_id"], row["sha256"]) for row in sources}
    if not any(kind == "completed_events" for kind, _ in source_pairs):
        raise DatasetValidationError("source provenance has no completed-events index")
    missing_event_sources = sorted(
        event_id for event_id in event_by_id if ("event", event_id) not in source_pairs
    )
    if missing_event_sources:
        raise DatasetValidationError(
            f"events have no cached source provenance: {missing_event_sources[:10]}"
        )
    missing_fight_sources = sorted(
        fight_id for fight_id in fight_by_id if ("fight", fight_id) not in source_pairs
    )
    if missing_fight_sources:
        raise DatasetValidationError(
            f"fights have no cached source provenance: {missing_fight_sources[:10]}"
        )
    missing_profile_sources = sorted(
        fighter["fighter_id"]
        for fighter in fighters
        if fighter.get("profile_origin") == "ufcstats"
        and (
            not fighter.get("profile_as_of_utc")
            or not fighter.get("profile_source_sha256")
            or (
                "fighter",
                fighter["fighter_id"],
                fighter.get("profile_source_sha256", ""),
            )
            not in source_triples
        )
    )
    if missing_profile_sources:
        raise DatasetValidationError(
            "scraped fighter profiles have no cached source provenance: "
            f"{missing_profile_sources[:10]}"
        )
    mapping = build_feature_coverage_report(
        feature_dictionary_path,
        expected_feature_count=expected_feature_count,
    )
    dates = sorted(row["event_date"] for row in events)
    eligible_label_count = sum(
        is_legacy_label_eligible(
            event_date=date.fromisoformat(fight["event_date"]),
            fighter_1_status=fight["fighter_1_status"],
            fighter_2_status=fight["fighter_2_status"],
            time_format=fight["time_format"],
        )
        for fight in fights
    )
    return {
        "status": "valid",
        "validation_scope": "actual_normalized_crawl_data",
        "counts": repository.counts(),
        "event_date_min": dates[0],
        "event_date_max": dates[-1],
        "expected_events_checked": len(requested_event_ids),
        "totals_round_reconciled_fights": reconciled_fights,
        "legacy_label_eligible_fights": eligible_label_count,
        "feature_source_mapping": {
            "status": mapping["status"],
            "scope": mapping["scope"],
            "feature_count": mapping["feature_count"],
            "page_types": mapping["page_types"],
        },
        "fighter_directory_coverage": {
            "required_complete": require_fighter_directory_complete,
            "directory_pages_present": len(directory_page_ids),
            "directory_pages_missing": missing_directory_pages,
            "indexed": len(indexed_fighter_ids),
            "profiles_present": len(indexed_fighter_ids & fighter_ids),
            "profiles_missing": len(missing_index_profiles),
        },
        "processed_training_assets_modified": False,
    }


__all__ = ["validate_local_repository"]
