"""Transactional local storage with normalized CSV materializations."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from contextlib import closing
from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import date
from pathlib import Path
from typing import Any, cast

from ufc_ml_latestdatafetcher.config import StorageConfig
from ufc_ml_latestdatafetcher.models import (
    EventRecord,
    FighterFightStats,
    FighterProfile,
    FighterReference,
    FightRecord,
    RoundStats,
    SourcePageRecord,
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_ready(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _row(record: Any) -> dict[str, Any]:
    if not is_dataclass(record):
        raise TypeError(f"expected dataclass record, got {type(record).__name__}")
    data = asdict(cast(Any, record))
    return {key: _json_ready(value) for key, value in data.items()}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _json_ready(row.get(name)) for name in fieldnames})
    temporary.replace(path)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(_json_ready(dict(payload)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@dataclass(frozen=True, slots=True)
class _TableSpec:
    name: str
    record_type: type[Any]
    key_fields: tuple[str, ...]
    sort_fields: tuple[str, ...]
    path_attribute: str

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(field.name for field in fields(self.record_type))


_TABLE_SPECS = (
    _TableSpec("events", EventRecord, ("event_id",), ("event_date", "event_id"), "events_path"),
    _TableSpec(
        "fights",
        FightRecord,
        ("fight_id",),
        ("event_date", "event_id", "bout_order"),
        "fights_path",
    ),
    _TableSpec(
        "bout_fighter_totals",
        FighterFightStats,
        ("fight_id", "fighter_id"),
        ("event_date", "fight_id", "corner"),
        "totals_path",
    ),
    _TableSpec(
        "round_stats",
        RoundStats,
        ("fight_id", "round_number", "fighter_id"),
        ("event_date", "fight_id", "round_number", "corner"),
        "rounds_path",
    ),
    _TableSpec(
        "fighters",
        FighterProfile,
        ("fighter_id",),
        ("fighter_name", "fighter_id"),
        "fighters_path",
    ),
    _TableSpec(
        "fighter_index",
        FighterReference,
        ("fighter_id",),
        ("fighter_name", "fighter_id"),
        "fighter_index_path",
    ),
    _TableSpec(
        "source_pages",
        SourcePageRecord,
        ("page_kind", "source_id", "sha256"),
        ("page_kind", "source_id", "fetched_at_utc", "sha256"),
        "source_pages_path",
    ),
)
_SPEC_BY_NAME = {spec.name: spec for spec in _TABLE_SPECS}
_NUMERIC_SORT_FIELDS = frozenset({"bout_order", "round_number", "corner"})
_LEGACY_UNKNOWN_PARSER_VERSION = "legacy-unknown"


def _identifier(value: str) -> str:
    """Quote one internal SQLite identifier."""

    return '"' + value.replace('"', '""') + '"'


class LocalRepository:
    """Use SQLite for atomic updates and expose deterministic normalized CSV views."""

    def __init__(self, config: StorageConfig) -> None:
        self.config = config
        root = config.normalized_dir
        root.mkdir(parents=True, exist_ok=True)
        self.database_path = root / "ufcstats.sqlite3"
        self.events_path = root / "events.csv"
        self.fights_path = root / "fights.csv"
        self.totals_path = root / "bout_fighter_totals.csv"
        self.rounds_path = root / "round_stats.csv"
        self.fighters_path = root / "fighters.csv"
        self.fighter_index_path = root / "fighter_index.csv"
        self.source_pages_path = root / "source_pages.csv"
        if self._initialize() or self._source_pages_csv_has_blank_parser_versions():
            self.materialize_table("source_pages")

    def _source_pages_csv_has_blank_parser_versions(self) -> bool:
        """Detect a migration committed to SQLite but not yet materialized to CSV."""

        return any(
            not row.get("parser_version", "").strip()
            for row in read_csv_rows(self.source_pages_path)
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> bool:
        source_pages_changed = False
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS repository_metadata "
                "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            for spec in _TABLE_SPECS:
                column_sql = ", ".join(f"{_identifier(column)} TEXT" for column in spec.columns)
                primary_key = ", ".join(_identifier(name) for name in spec.key_fields)
                connection.execute(
                    f"CREATE TABLE IF NOT EXISTS {_identifier(spec.name)} "
                    f"({column_sql}, PRIMARY KEY ({primary_key}))"
                )
                actual_columns = {
                    str(row["name"])
                    for row in connection.execute(f"PRAGMA table_info({_identifier(spec.name)})")
                }
                for missing in set(spec.columns) - actual_columns:
                    connection.execute(
                        f"ALTER TABLE {_identifier(spec.name)} "
                        f"ADD COLUMN {_identifier(missing)} TEXT"
                    )
            bootstrapped = connection.execute(
                "SELECT value FROM repository_metadata WHERE key = ?",
                ("csv_bootstrap_complete",),
            ).fetchone()
            if bootstrapped is None:
                self._bootstrap_csvs(connection)
                connection.execute(
                    "INSERT INTO repository_metadata(key, value) VALUES (?, ?)",
                    ("csv_bootstrap_complete", "1"),
                )
            cache_migrated = connection.execute(
                "SELECT value FROM repository_metadata WHERE key = ?",
                ("immutable_cache_migration_complete",),
            ).fetchone()
            if cache_migrated is None:
                source_pages_changed = bool(self._prune_unverifiable_legacy_sources(connection))
                connection.execute(
                    "INSERT INTO repository_metadata(key, value) VALUES (?, ?)",
                    ("immutable_cache_migration_complete", "1"),
                )
            migrated_parser_versions = connection.execute(
                "UPDATE source_pages SET parser_version = ? "
                "WHERE COALESCE(TRIM(parser_version), '') = ''",
                (_LEGACY_UNKNOWN_PARSER_VERSION,),
            ).rowcount
            source_pages_changed = bool(migrated_parser_versions) or source_pages_changed
        return source_pages_changed

    def _bootstrap_csvs(self, connection: sqlite3.Connection) -> None:
        """Import CSVs written by the pre-SQLite prototype once, without deleting them."""

        for spec in _TABLE_SPECS:
            path = cast(Path, getattr(self, spec.path_attribute))
            rows = read_csv_rows(path)
            if rows:
                if spec.name == "fighters":
                    for row in rows:
                        row["profile_origin"] = row.get("profile_origin") or "baseline"
                self._upsert_rows(connection, spec, rows)

    @staticmethod
    def _prune_unverifiable_legacy_sources(connection: sqlite3.Connection) -> int:
        """Drop prototype provenance whose mutable cache no longer matches its hash."""

        deleted = 0
        rows = connection.execute(
            "SELECT page_kind, source_id, sha256, cache_path FROM source_pages"
        ).fetchall()
        for row in rows:
            path = Path(str(row["cache_path"])).expanduser()
            try:
                html = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                matches = False
            else:
                matches = hashlib.sha256(html.encode("utf-8")).hexdigest() == str(row["sha256"])
            if not matches:
                connection.execute(
                    "DELETE FROM source_pages WHERE page_kind = ? AND source_id = ? AND sha256 = ?",
                    (row["page_kind"], row["source_id"], row["sha256"]),
                )
                deleted += 1
        return deleted

    @staticmethod
    def _upsert_rows(
        connection: sqlite3.Connection,
        spec: _TableSpec,
        rows: Iterable[Mapping[str, Any]],
    ) -> int:
        materialized = [
            tuple(_json_ready(row.get(column)) for column in spec.columns) for row in rows
        ]
        if not materialized:
            return 0
        columns_sql = ", ".join(_identifier(name) for name in spec.columns)
        placeholders = ", ".join("?" for _ in spec.columns)
        keys_sql = ", ".join(_identifier(name) for name in spec.key_fields)
        update_fields = [name for name in spec.columns if name not in spec.key_fields]
        if update_fields:
            update_sql = ", ".join(
                f"{_identifier(name)} = excluded.{_identifier(name)}" for name in update_fields
            )
            conflict_sql = f"DO UPDATE SET {update_sql}"
        else:
            conflict_sql = "DO NOTHING"
        statement = (
            f"INSERT INTO {_identifier(spec.name)} ({columns_sql}) "
            f"VALUES ({placeholders}) ON CONFLICT ({keys_sql}) {conflict_sql}"
        )
        connection.executemany(statement, materialized)
        return len(materialized)

    @staticmethod
    def _record_rows(records: Iterable[Any]) -> list[dict[str, Any]]:
        return [_row(record) for record in records]

    def _merge_table(self, table_name: str, records: Iterable[Any]) -> int:
        spec = _SPEC_BY_NAME[table_name]
        rows = self._record_rows(records)
        with closing(self._connect()) as connection, connection:
            count = self._upsert_rows(connection, spec, rows)
        self.materialize_table(table_name)
        return count

    def publish_event(
        self,
        event: EventRecord,
        fights: Iterable[FightRecord],
        totals: Iterable[FighterFightStats],
        rounds: Iterable[RoundStats],
        profiles: Iterable[FighterProfile],
        source_pages: Iterable[SourcePageRecord] = (),
    ) -> None:
        """Replace an event and all descendants in one all-or-nothing transaction."""

        fight_rows = tuple(fights)
        total_rows = tuple(totals)
        round_rows = tuple(rounds)
        profile_rows = tuple(profiles)
        source_rows = tuple(source_pages)
        if not fight_rows:
            raise ValueError(f"event {event.event_id} cannot be published without fights")
        if (
            any(row.event_id != event.event_id for row in fight_rows)
            or any(row.event_id != event.event_id for row in total_rows)
            or any(row.event_id != event.event_id for row in round_rows)
        ):
            raise ValueError(f"event {event.event_id} publication contains another event ID")
        fight_ids = {row.fight_id for row in fight_rows}
        if any(row.fight_id not in fight_ids for row in total_rows) or any(
            row.fight_id not in fight_ids for row in round_rows
        ):
            raise ValueError(f"event {event.event_id} publication contains orphan statistics")

        with closing(self._connect()) as connection, connection:
            # Delete by event, not only by incoming IDs, so source corrections cannot
            # leave ghost fights or stale extra rounds behind.
            for table_name in ("round_stats", "bout_fighter_totals", "fights", "events"):
                connection.execute(
                    f"DELETE FROM {_identifier(table_name)} WHERE event_id = ?",
                    (event.event_id,),
                )
            self._upsert_rows(
                connection,
                _SPEC_BY_NAME["fighters"],
                self._record_rows(profile_rows),
            )
            self._upsert_rows(connection, _SPEC_BY_NAME["events"], (_row(event),))
            self._upsert_rows(
                connection,
                _SPEC_BY_NAME["fights"],
                self._record_rows(fight_rows),
            )
            self._upsert_rows(
                connection,
                _SPEC_BY_NAME["bout_fighter_totals"],
                self._record_rows(total_rows),
            )
            self._upsert_rows(
                connection,
                _SPEC_BY_NAME["round_stats"],
                self._record_rows(round_rows),
            )
            self._upsert_rows(
                connection,
                _SPEC_BY_NAME["source_pages"],
                self._record_rows(source_rows),
            )

    def publish_fighter_discovery(
        self,
        references: Iterable[FighterReference],
        profiles: Iterable[FighterProfile],
        source_pages: Iterable[SourcePageRecord] = (),
    ) -> None:
        """Replace the complete A-Z index snapshot and upsert fetched profiles atomically."""

        reference_rows = tuple(references)
        source_rows = tuple(source_pages)
        if not reference_rows:
            raise ValueError("fighter directory snapshot cannot be empty")
        with closing(self._connect()) as connection, connection:
            connection.execute(f"DELETE FROM {_identifier('fighter_index')}")
            self._upsert_rows(
                connection,
                _SPEC_BY_NAME["fighter_index"],
                self._record_rows(reference_rows),
            )
            self._upsert_rows(
                connection,
                _SPEC_BY_NAME["fighters"],
                self._record_rows(profiles),
            )
            self._upsert_rows(
                connection,
                _SPEC_BY_NAME["source_pages"],
                self._record_rows(source_rows),
            )

    def merge_events(self, records: Iterable[EventRecord]) -> int:
        return self._merge_table("events", records)

    def merge_fights(self, records: Iterable[FightRecord]) -> int:
        return self._merge_table("fights", records)

    def merge_totals(self, records: Iterable[FighterFightStats]) -> int:
        return self._merge_table("bout_fighter_totals", records)

    def merge_rounds(self, records: Iterable[RoundStats]) -> int:
        return self._merge_table("round_stats", records)

    def merge_fighters(self, records: Iterable[FighterProfile]) -> int:
        return self._merge_table("fighters", records)

    def merge_fighter_index(self, records: Iterable[FighterReference]) -> int:
        return self._merge_table("fighter_index", records)

    def merge_source_pages(self, records: Iterable[SourcePageRecord]) -> int:
        return self._merge_table("source_pages", records)

    def known_event_ids(self) -> set[str]:
        """Return event IDs with at least one transactionally stored fight."""

        with closing(self._connect()) as connection:
            return {
                str(row[0])
                for row in connection.execute(
                    "SELECT e.event_id FROM events AS e "
                    "WHERE COALESCE(TRIM(e.event_id), '') != '' "
                    "AND EXISTS ("
                    "SELECT 1 FROM fights AS f WHERE f.event_id = e.event_id"
                    ")"
                )
            }

    def latest_stored_fight_date(self) -> date | None:
        """Return the newest fight date in SQLite, or ``None`` when no fights exist."""

        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT MAX(NULLIF(TRIM(event_date), '')) FROM fights"
            ).fetchone()
        value = row[0] if row is not None else None
        return date.fromisoformat(str(value)) if value is not None else None

    def known_fighter_ids(self) -> set[str]:
        with closing(self._connect()) as connection:
            return {
                str(row[0])
                for row in connection.execute(f"SELECT fighter_id FROM {_identifier('fighters')}")
            }

    def fighter_ids_needing_bio_refresh(self) -> set[str]:
        """Return baseline-seeded profiles whose model bio primitives are incomplete."""

        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT fighter_id FROM fighters "
                "WHERE COALESCE(profile_origin, '') != 'ufcstats' AND ("
                "COALESCE(height_inches, '') = '' OR "
                "COALESCE(reach_inches, '') = '' OR "
                "COALESCE(dob, '') = '' OR "
                "LOWER(COALESCE(stance, '')) IN ('', 'unknown'))"
            )
            return {str(row[0]) for row in rows}

    def materialize_table(self, table_name: str) -> None:
        spec = _SPEC_BY_NAME[table_name]
        with closing(self._connect()) as connection:
            selected = connection.execute(f"SELECT * FROM {_identifier(spec.name)}").fetchall()
        rows = [{column: row[column] for column in spec.columns} for row in selected]

        def sort_key(row: Mapping[str, Any]) -> tuple[str | int, ...]:
            ordered: list[str | int] = []
            for name in spec.sort_fields:
                value = row.get(name)
                if name in _NUMERIC_SORT_FIELDS:
                    ordered.append(int(value or 0))
                else:
                    ordered.append(str(value or ""))
            return tuple(ordered)

        rows.sort(key=sort_key)
        output_path = cast(Path, getattr(self, spec.path_attribute))
        atomic_write_csv(output_path, rows, fieldnames=spec.columns)

    def materialize_csvs(self) -> None:
        """Refresh every human-readable CSV view from committed SQLite state."""

        for spec in _TABLE_SPECS:
            self.materialize_table(spec.name)

    @property
    def normalized_csv_paths(self) -> tuple[Path, ...]:
        return tuple(cast(Path, getattr(self, spec.path_attribute)) for spec in _TABLE_SPECS)

    def backup_database(self, destination: Path) -> None:
        """Create a transactionally consistent SQLite snapshot for a run bundle."""

        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"database backup destination already exists: {destination}")
        with (
            closing(self._connect()) as source,
            closing(sqlite3.connect(destination, timeout=30.0)) as target,
        ):
            source.backup(target)
            target.commit()

    def counts(self) -> dict[str, int]:
        with closing(self._connect()) as connection:
            return {
                spec.name: int(
                    connection.execute(f"SELECT COUNT(*) FROM {_identifier(spec.name)}").fetchone()[
                        0
                    ]
                )
                for spec in _TABLE_SPECS
            }


__all__ = [
    "LocalRepository",
    "atomic_write_csv",
    "atomic_write_json",
    "read_csv_rows",
]
