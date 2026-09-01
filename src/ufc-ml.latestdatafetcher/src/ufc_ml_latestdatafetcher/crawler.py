"""Incremental UFCStats crawl orchestration with event-level publication boundaries."""

from __future__ import annotations

from collections.abc import Callable, Collection
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from ufc_ml_latestdatafetcher import __version__
from ufc_ml_latestdatafetcher.config import FetcherConfig
from ufc_ml_latestdatafetcher.errors import LatestDataFetcherError, SourceParseError
from ufc_ml_latestdatafetcher.feature_coverage import build_feature_coverage_report
from ufc_ml_latestdatafetcher.fetching import CachedBrowserClient
from ufc_ml_latestdatafetcher.legacy import export_legacy_candidates, seed_fighters_from_baseline
from ufc_ml_latestdatafetcher.models import (
    EventRecord,
    FighterFightStats,
    FighterProfile,
    FighterReference,
    RoundStats,
)
from ufc_ml_latestdatafetcher.parsing import (
    parse_completed_events,
    parse_event,
    parse_fight,
    parse_fighter_directory,
    parse_fighter_profile,
)
from ufc_ml_latestdatafetcher.storage import LocalRepository, atomic_write_json
from ufc_ml_latestdatafetcher.validation import validate_local_repository

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class EventRefreshPlan:
    """SQLite-aware event selection for one refresh invocation."""

    eligible: tuple[EventRecord, ...]
    already_stored: tuple[EventRecord, ...]
    missing: tuple[EventRecord, ...]
    candidates: tuple[EventRecord, ...]
    selected: tuple[EventRecord, ...]
    gaps_at_or_before_watermark: tuple[EventRecord, ...]
    latest_stored_fight_date: date | None
    refresh_existing: bool


class UFCStatsCrawler:
    def __init__(
        self,
        config: FetcherConfig,
        repository: LocalRepository,
        client: CachedBrowserClient,
        *,
        feature_dictionary_path: Path,
        expected_feature_count: int,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.config = config
        self.repository = repository
        self.client = client
        self.feature_dictionary_path = feature_dictionary_path
        self.expected_feature_count = expected_feature_count
        self.progress = progress or (lambda _: None)

    def _url(self, path: str) -> str:
        return f"{self.config.source.base_url}{path if path.startswith('/') else f'/{path}'}"

    def discover_events(self, *, refresh: bool = True) -> tuple[EventRecord, ...]:
        url = self._url(self.config.source.completed_events_path)
        page = self.client.get(
            url,
            kind="completed_events",
            source_id="completed_all",
            refresh=refresh,
        )
        events = parse_completed_events(page.html, base_url=self.config.source.base_url)
        if len(events) < self.config.source.minimum_completed_event_count:
            raise SourceParseError(
                "completed-events index appears paginated or truncated: "
                f"expected at least {self.config.source.minimum_completed_event_count}, "
                f"parsed {len(events)}"
            )
        oldest = min(event.event_date for event in events)
        if oldest > self.config.source.earliest_completed_event_date:
            raise SourceParseError(
                "completed-events index does not reach the configured historical boundary: "
                f"oldest parsed date is {oldest}, expected "
                f"{self.config.source.earliest_completed_event_date} or earlier"
            )
        return events

    @staticmethod
    def select_events(
        events: tuple[EventRecord, ...],
        *,
        since: date | None,
        through: date,
        max_events: int | None,
    ) -> tuple[EventRecord, ...]:
        eligible = [
            event
            for event in events
            if event.event_date <= through and (since is None or event.event_date >= since)
        ]
        eligible.sort(key=lambda event: (event.event_date, event.event_id), reverse=True)
        if max_events is not None:
            eligible = eligible[:max_events]
        eligible.sort(key=lambda event: (event.event_date, event.event_id))
        return tuple(eligible)

    @classmethod
    def build_event_refresh_plan(
        cls,
        events: tuple[EventRecord, ...],
        *,
        since: date | None,
        through: date,
        max_events: int | None,
        known_event_ids: Collection[str],
        latest_stored_fight_date: date | None,
        refresh_existing: bool,
    ) -> EventRefreshPlan:
        """Plan missing events while retaining holes at/before the SQLite watermark.

        A date watermark alone is unsafe: a later event may have committed after an
        earlier event failed. Event IDs are therefore authoritative for skipping work;
        the latest fight date is retained for diagnostics and gap classification.
        """

        eligible = cls.select_events(
            events,
            since=since,
            through=through,
            max_events=None,
        )
        normalized_known_ids = {value.strip() for value in known_event_ids if value.strip()}
        already_stored = tuple(
            event for event in eligible if event.event_id in normalized_known_ids
        )
        missing = tuple(event for event in eligible if event.event_id not in normalized_known_ids)
        candidates = eligible if refresh_existing else missing
        selected = cls.select_events(
            candidates,
            since=since,
            through=through,
            max_events=max_events,
        )
        gaps = (
            tuple(event for event in missing if event.event_date <= latest_stored_fight_date)
            if latest_stored_fight_date is not None
            else ()
        )
        return EventRefreshPlan(
            eligible=eligible,
            already_stored=already_stored,
            missing=missing,
            candidates=candidates,
            selected=selected,
            gaps_at_or_before_watermark=gaps,
            latest_stored_fight_date=latest_stored_fight_date,
            refresh_existing=refresh_existing,
        )

    def plan_event_refresh(
        self,
        events: tuple[EventRecord, ...],
        *,
        since: date | None,
        through: date,
        max_events: int | None,
        refresh_existing: bool,
    ) -> EventRefreshPlan:
        """Build an incremental plan from the completed index and local SQLite state."""

        return self.build_event_refresh_plan(
            events,
            since=since,
            through=through,
            max_events=max_events,
            known_event_ids=self.repository.known_event_ids(),
            latest_stored_fight_date=self.repository.latest_stored_fight_date(),
            refresh_existing=refresh_existing,
        )

    def _fetch_profile(
        self,
        reference: FighterReference,
        *,
        refresh: bool,
    ) -> FighterProfile:
        page = self.client.get(
            reference.fighter_url,
            kind="fighter",
            source_id=reference.fighter_id,
            refresh=refresh,
        )
        try:
            profile = parse_fighter_profile(
                page.html,
                fighter_url=reference.fighter_url,
                base_url=self.config.source.base_url,
            )
        except SourceParseError:
            if refresh or not page.record.from_cache:
                raise
            page = self.client.get(
                reference.fighter_url,
                kind="fighter",
                source_id=reference.fighter_id,
                refresh=True,
            )
            profile = parse_fighter_profile(
                page.html,
                fighter_url=reference.fighter_url,
                base_url=self.config.source.base_url,
            )
        if profile.fighter_id != reference.fighter_id:
            raise LatestDataFetcherError(
                f"fighter profile ID mismatch: {profile.fighter_id} != {reference.fighter_id}"
            )
        return replace(
            profile,
            profile_as_of_utc=page.record.fetched_at_utc,
            profile_source_sha256=page.record.sha256,
            profile_origin="ufcstats",
        )

    def _crawl_event(
        self,
        listing: EventRecord,
        *,
        refresh_pages: bool,
        refresh_known_fighters: bool,
    ) -> dict[str, int]:
        source_record_start = len(self.client.records)
        event_page = self.client.get(
            listing.event_url,
            kind="event",
            source_id=listing.event_id,
            refresh=refresh_pages,
        )
        try:
            parsed_event = parse_event(
                event_page.html,
                event_url=listing.event_url,
                base_url=self.config.source.base_url,
                fallback=listing,
            )
        except SourceParseError:
            if refresh_pages or not event_page.record.from_cache:
                raise
            event_page = self.client.get(
                listing.event_url,
                kind="event",
                source_id=listing.event_id,
                refresh=True,
            )
            parsed_event = parse_event(
                event_page.html,
                event_url=listing.event_url,
                base_url=self.config.source.base_url,
                fallback=listing,
            )
        if parsed_event.event.event_date != listing.event_date:
            raise SourceParseError(
                f"event {listing.event_id} date differs between index "
                f"({listing.event_date}) and detail page ({parsed_event.event.event_date})"
            )
        fights = []
        totals: list[FighterFightStats] = []
        rounds: list[RoundStats] = []
        references: dict[str, FighterReference] = {}
        for reference in parsed_event.fights:
            self.progress(
                f"fight {reference.bout_order}/{len(parsed_event.fights)} {reference.fight_id}"
            )
            fight_page = self.client.get(
                reference.fight_url,
                kind="fight",
                source_id=reference.fight_id,
                refresh=refresh_pages,
            )
            try:
                parsed_fight = parse_fight(
                    fight_page.html,
                    event=parsed_event.event,
                    reference=reference,
                    base_url=self.config.source.base_url,
                )
            except SourceParseError:
                if refresh_pages or not fight_page.record.from_cache:
                    raise
                fight_page = self.client.get(
                    reference.fight_url,
                    kind="fight",
                    source_id=reference.fight_id,
                    refresh=True,
                )
                parsed_fight = parse_fight(
                    fight_page.html,
                    event=parsed_event.event,
                    reference=reference,
                    base_url=self.config.source.base_url,
                )
            fights.append(parsed_fight.fight)
            totals.extend(parsed_fight.totals)
            rounds.extend(parsed_fight.rounds)
            references.update({fighter.fighter_id: fighter for fighter in parsed_fight.fighters})

        profiles: list[FighterProfile] = []
        for fighter_id, fighter_reference in sorted(references.items()):
            self.progress(f"fighter profile {fighter_reference.fighter_name} ({fighter_id})")
            profiles.append(self._fetch_profile(fighter_reference, refresh=refresh_known_fighters))

        # Replace the event and every descendant in one SQLite transaction only
        # after every page in the event has parsed successfully.
        self.repository.publish_event(
            parsed_event.event,
            fights,
            totals,
            rounds,
            profiles,
            self.client.records[source_record_start:],
        )
        return {
            "fights": len(fights),
            "totals": len(totals),
            "rounds": len(rounds),
            "profiles": len(profiles),
        }

    def _crawl_fighter_directory(
        self,
        *,
        refresh_known_fighters: bool,
        max_new_fighters: int | None,
    ) -> dict[str, Any]:
        source_record_start = len(self.client.records)
        references: dict[str, FighterReference] = {}
        failures: list[dict[str, str]] = []
        directory_page_failures = 0
        directory_snapshot_valid = True
        for letter in "abcdefghijklmnopqrstuvwxyz":
            self.progress(f"fighter directory {letter.upper()}")
            url = self._url(self.config.source.fighter_directory_path.format(letter=letter))
            try:
                page = self.client.get(
                    url,
                    kind="fighter_directory",
                    source_id=letter,
                    refresh=True,
                )
                for reference in parse_fighter_directory(
                    page.html,
                    base_url=self.config.source.base_url,
                ):
                    existing = references.get(reference.fighter_id)
                    if existing is not None and existing != reference:
                        raise SourceParseError(
                            f"fighter {reference.fighter_id} has conflicting directory rows"
                        )
                    references[reference.fighter_id] = reference
            except Exception as exc:
                directory_page_failures += 1
                directory_snapshot_valid = False
                failures.append({"letter": letter, "error": str(exc)})

        if (
            directory_page_failures == 0
            and len(references) < self.config.source.minimum_fighter_directory_count
        ):
            directory_snapshot_valid = False
            failures.append(
                {
                    "stage": "fighter_directory_completeness",
                    "error": (
                        f"parsed {len(references)} fighters; expected at least "
                        f"{self.config.source.minimum_fighter_directory_count}"
                    ),
                }
            )

        known = self.repository.known_fighter_ids()
        incomplete_baseline_bios = self.repository.fighter_ids_needing_bio_refresh()
        pending = [
            reference
            for fighter_id, reference in sorted(references.items())
            if (
                refresh_known_fighters
                or fighter_id not in known
                or fighter_id in incomplete_baseline_bios
            )
        ]
        profiles_missing_before_fetch = len(pending)
        if max_new_fighters is not None:
            pending = pending[:max_new_fighters]
        profiles: list[FighterProfile] = []
        for index, reference in enumerate(pending, start=1):
            self.progress(f"directory profile {index}/{len(pending)} {reference.fighter_name}")
            try:
                profiles.append(self._fetch_profile(reference, refresh=refresh_known_fighters))
            except Exception as exc:
                failures.append({"fighter_id": reference.fighter_id, "error": str(exc)})
        if directory_page_failures == 0 and directory_snapshot_valid:
            self.repository.publish_fighter_discovery(
                references.values(),
                profiles,
                self.client.records[source_record_start:],
            )
        elif profiles:
            # Keep successfully parsed profiles, but never replace a complete index
            # with a partial A-Z discovery snapshot.
            self.repository.merge_fighters(profiles)
        return {
            "discovered_fighters": len(references),
            "directory_pages_succeeded": 26 - directory_page_failures,
            "directory_snapshot_valid": directory_snapshot_valid,
            "profiles_missing_before_fetch": profiles_missing_before_fetch,
            "profiles_fetched": len(profiles),
            "profiles_deferred_by_limit": profiles_missing_before_fetch - len(pending),
            "failures": failures,
        }

    def refresh(
        self,
        *,
        since: date | None,
        through: date,
        max_events: int | None = None,
        include_fighter_directory: bool = True,
        refresh_pages: bool = False,
        refresh_known_fighters: bool = False,
        max_new_fighters: int | None = None,
    ) -> dict[str, Any]:
        started_at = datetime.now(UTC)
        run_id = started_at.strftime("%Y%m%dT%H%M%S%fZ")
        manifest_path = self.config.storage.manifest_dir / f"run-{run_id}.json"
        source_manifest_path = self.config.storage.manifest_dir / f"source-pages-{run_id}.json"
        initial_manifest: dict[str, Any] = {
            "run_id": run_id,
            "fetcher_version": __version__,
            "status": "running",
            "started_at_utc": started_at.isoformat(),
            "since": since,
            "through": through,
            "max_events": max_events,
            "refresh_existing_events": refresh_pages,
            "include_fighter_directory": include_fighter_directory,
            "processed_training_assets_modified": False,
        }
        atomic_write_json(manifest_path, initial_manifest)
        try:
            seed_count = seed_fighters_from_baseline(
                self.repository,
                self.config.storage.baseline_fighters_path,
            )
            discovered = self.discover_events(refresh=True)
            # Persist the discovery page before crawling descendants so an
            # interrupted run still retains its exact index provenance.
            self.repository.merge_source_pages(self.client.records)
            event_plan = self.plan_event_refresh(
                discovered,
                since=since,
                through=through,
                max_events=max_events,
                refresh_existing=refresh_pages,
            )
            eligible = event_plan.eligible
            selected = event_plan.selected
            watermark = (
                event_plan.latest_stored_fight_date.isoformat()
                if event_plan.latest_stored_fight_date is not None
                else "none"
            )
            self.progress(
                f"SQLite latest fight date {watermark}; "
                f"{len(event_plan.already_stored)} stored, "
                f"{len(event_plan.missing)} missing, "
                f"{len(event_plan.gaps_at_or_before_watermark)} earlier gaps, "
                f"{len(selected)} selected"
            )
            event_failures: list[dict[str, str]] = []
            totals = {
                "events": 0,
                "fights": 0,
                "totals": 0,
                "rounds": 0,
                "profiles": 0,
            }
            for index, event in enumerate(selected, start=1):
                self.progress(
                    f"event {index}/{len(selected)} "
                    f"{event.event_date.isoformat()} {event.event_name}"
                )
                try:
                    counts = self._crawl_event(
                        event,
                        refresh_pages=refresh_pages,
                        refresh_known_fighters=refresh_known_fighters,
                    )
                    totals["events"] += 1
                    for key in ("fights", "totals", "rounds", "profiles"):
                        totals[key] += counts[key]
                except Exception as exc:
                    event_failures.append(
                        {
                            "event_id": event.event_id,
                            "event_name": event.event_name,
                            "error": str(exc),
                        }
                    )

            directory_result: dict[str, Any] = {
                "discovered_fighters": 0,
                "directory_pages_succeeded": 0,
                "directory_snapshot_valid": False,
                "profiles_missing_before_fetch": 0,
                "profiles_fetched": 0,
                "profiles_deferred_by_limit": 0,
                "failures": [],
            }
            if include_fighter_directory:
                directory_result = self._crawl_fighter_directory(
                    refresh_known_fighters=refresh_known_fighters,
                    max_new_fighters=max_new_fighters,
                )

            self.repository.merge_source_pages(self.client.records)
            self.repository.materialize_csvs()
            coverage = build_feature_coverage_report(
                self.feature_dictionary_path,
                expected_feature_count=self.expected_feature_count,
            )
            coverage_path = self.config.storage.manifest_dir / f"feature-source-{run_id}.json"
            atomic_write_json(coverage_path, coverage)

            failures = [*event_failures, *directory_result["failures"]]
            completion_blockers: list[str] = []
            if len(selected) < len(event_plan.candidates):
                completion_blockers.append(
                    f"--max-events selected {len(selected)} of "
                    f"{len(event_plan.candidates)} refresh candidates"
                )
            if not include_fighter_directory:
                completion_blockers.append("fighter A-Z directory crawl was skipped")
            deferred_profiles = int(directory_result["profiles_deferred_by_limit"])
            if deferred_profiles:
                completion_blockers.append(
                    f"--max-new-fighters deferred {deferred_profiles} profiles"
                )

            validation: dict[str, Any] | None = None
            try:
                expected_covered_events = {
                    event.event_id for event in (*event_plan.already_stored, *event_plan.selected)
                }
                validation = validate_local_repository(
                    self.repository,
                    feature_dictionary_path=self.feature_dictionary_path,
                    expected_feature_count=self.expected_feature_count,
                    expected_event_ids=expected_covered_events,
                    require_fighter_directory_complete=include_fighter_directory,
                    minimum_fighter_directory_count=(
                        self.config.source.minimum_fighter_directory_count
                    ),
                )
            except LatestDataFetcherError as exc:
                failures.append({"stage": "repository_validation", "error": str(exc)})

            exports: dict[str, Any]
            if failures or completion_blockers or validation is None:
                exports = {
                    "status": "withheld",
                    "reason": "candidate exports require a complete, validated run",
                }
            else:
                exports = {
                    "status": "exported",
                    **export_legacy_candidates(
                        self.repository,
                        output_dir=self.config.storage.candidate_dir / f"run-{run_id}",
                    ),
                }

            if validation is None:
                status = "invalid"
            elif failures:
                status = "partial"
            elif completion_blockers:
                status = "bounded"
            else:
                status = "complete"
            completed_at = datetime.now(UTC)
            atomic_write_json(
                source_manifest_path,
                {
                    "run_id": run_id,
                    "page_count": len(self.client.records),
                    "pages": [asdict(record) for record in self.client.records],
                },
            )
            result: dict[str, Any] = {
                "run_id": run_id,
                "fetcher_version": __version__,
                "status": status,
                "started_at_utc": started_at.isoformat(),
                "completed_at_utc": completed_at.isoformat(),
                "since": since,
                "through": through,
                "events_discovered": len(discovered),
                "events_eligible": len(eligible),
                "events_already_stored": len(event_plan.already_stored),
                "events_missing_before_run": len(event_plan.missing),
                "events_gaps_at_or_before_watermark": len(event_plan.gaps_at_or_before_watermark),
                "events_refresh_candidates": len(event_plan.candidates),
                "events_selected": len(selected),
                "events_deferred_by_limit": len(event_plan.candidates) - len(selected),
                "latest_stored_fight_date_before_run": event_plan.latest_stored_fight_date,
                "latest_stored_fight_date_after_run": (self.repository.latest_stored_fight_date()),
                "refresh_existing_events": refresh_pages,
                "selected_event_records": [asdict(event) for event in selected],
                "seeded_fighter_profiles": seed_count,
                "scraped": totals,
                "fighter_directory": directory_result,
                "completion_blockers": completion_blockers,
                "failures": failures,
                "repository": validation,
                "feature_source_mapping_path": str(coverage_path),
                "source_page_count": len(self.client.records),
                "source_page_manifest_path": str(source_manifest_path),
                "exports": exports,
                "manifest_path": str(manifest_path),
                "processed_training_assets_modified": False,
                "model_retrained": False,
            }
            atomic_write_json(manifest_path, result)
            return result
        except Exception as exc:
            # Preserve diagnostic provenance even when discovery/storage fails before
            # a normal result can be assembled.
            with suppress(Exception):
                self.repository.merge_source_pages(self.client.records)
                self.repository.materialize_csvs()
            with suppress(Exception):
                atomic_write_json(
                    source_manifest_path,
                    {
                        "run_id": run_id,
                        "page_count": len(self.client.records),
                        "pages": [asdict(record) for record in self.client.records],
                    },
                )
            failed_result = {
                **initial_manifest,
                "status": "failed",
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "fatal_error": str(exc),
                "source_page_count": len(self.client.records),
                "source_page_manifest_path": str(source_manifest_path),
                "manifest_path": str(manifest_path),
            }
            atomic_write_json(manifest_path, failed_result)
            raise


__all__ = ["EventRefreshPlan", "UFCStatsCrawler"]
