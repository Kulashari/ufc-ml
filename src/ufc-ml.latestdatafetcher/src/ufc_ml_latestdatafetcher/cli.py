"""Standalone CLI for current UFCStats acquisition and local candidate exports."""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from ufc_ml_core.config import AppConfig, load_config
from ufc_ml_core.exceptions import UFCPredictorError
from ufc_ml_latestdatafetcher import __version__
from ufc_ml_latestdatafetcher.config import FetcherConfig, load_fetcher_config
from ufc_ml_latestdatafetcher.errors import LatestDataFetcherError
from ufc_ml_latestdatafetcher.fetching import BrowserFetcher, CachedBrowserClient, HtmlCache
from ufc_ml_latestdatafetcher.storage import LocalRepository
from ufc_ml_latestdatafetcher.validation import validate_local_repository

if TYPE_CHECKING:
    from ufc_ml_latestdatafetcher.crawler import UFCStatsCrawler

app = typer.Typer(
    name="ufc-latest-data",
    help="Cache and normalize UFCStats data without training or publishing a model.",
    no_args_is_help=True,
    invoke_without_command=True,
    pretty_exceptions_show_locals=False,
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _emit(payload: dict[str, Any]) -> None:
    typer.echo(json.dumps(_json_ready(payload), indent=2, sort_keys=True))


def _progress(message: str) -> None:
    typer.echo(f"[ufc-latest-data] {message}", err=True)


def _abort(exc: Exception) -> None:
    typer.echo(f"Error: {exc}", err=True)
    raise typer.Exit(code=2) from exc


def _parse_date(value: str | None, *, field_name: str) -> date | None:
    if value is None:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"{field_name} must be YYYY-MM-DD, got {value!r}") from exc


def _load(
    fetcher_config_path: Path,
    model_config_override: Path | None,
) -> tuple[FetcherConfig, AppConfig]:
    fetcher_config = load_fetcher_config(fetcher_config_path)
    model_config_path = model_config_override or fetcher_config.model_config_path
    model_config = load_config(model_config_path)
    return fetcher_config, model_config


def _crawler(
    fetcher_config: FetcherConfig,
    model_config: AppConfig,
    *,
    headed: bool,
    delay_seconds: float | None,
) -> tuple[BrowserFetcher, UFCStatsCrawler]:
    try:
        from ufc_ml_latestdatafetcher.crawler import UFCStatsCrawler
    except ImportError as exc:
        raise LatestDataFetcherError(
            'Latest-data dependencies are required. Install with `pip install -e ".[latestdata]"`.'
        ) from exc
    browser = BrowserFetcher(
        fetcher_config.source,
        headless=False if headed else None,
        delay_seconds=delay_seconds,
    )
    repository = LocalRepository(fetcher_config.storage)
    client = CachedBrowserClient(browser, HtmlCache(fetcher_config.storage.raw_html_dir))
    crawler = UFCStatsCrawler(
        fetcher_config,
        repository,
        client,
        feature_dictionary_path=model_config.data.feature_dictionary_path,
        expected_feature_count=model_config.data.expected_feature_count,
        progress=_progress,
    )
    return browser, crawler


@app.callback()
def root(
    version: bool = typer.Option(
        False,
        "--version",
        help="Show the installed package version and exit.",
        is_eager=True,
    ),
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()


@app.command("discover")
def discover_command(
    fetcher_config_path: Path = typer.Option(
        Path("configs/latestdatafetcher.yaml"),
        "--fetcher-config",
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    model_config_path: Path | None = typer.Option(
        None,
        "--model-config",
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    since: str | None = typer.Option(
        None,
        "--since",
        help="Inclusive YYYY-MM-DD; defaults to the day after the model cutoff.",
    ),
    through: str | None = typer.Option(
        None,
        "--through",
        help="Inclusive YYYY-MM-DD; defaults to today.",
    ),
    headed: bool = typer.Option(False, "--headed", help="Show the Chromium window."),
    delay_seconds: float | None = typer.Option(None, "--delay-seconds", min=0.0),
) -> None:
    """Fetch only the completed-events index and report the incremental plan."""

    try:
        fetcher_config, model_config = _load(fetcher_config_path, model_config_path)
        since_date = _parse_date(since, field_name="since") or (
            model_config.data.dataset_cutoff + timedelta(days=1)
        )
        through_date = _parse_date(through, field_name="through") or date.today()
        if since_date > through_date:
            raise ValueError("since cannot be later than through")
        if through_date > date.today():
            raise ValueError("through cannot be later than today")
        browser, crawler = _crawler(
            fetcher_config,
            model_config,
            headed=headed,
            delay_seconds=delay_seconds,
        )
        with browser:
            events = crawler.discover_events(refresh=True)
        event_plan = crawler.plan_event_refresh(
            events,
            since=since_date,
            through=through_date,
            max_events=None,
            refresh_existing=False,
        )
        selected = event_plan.selected
        _emit(
            {
                "status": "planned",
                "model_cutoff": model_config.data.dataset_cutoff,
                "since": since_date,
                "through": through_date,
                "events_on_index": len(events),
                "events_eligible": len(event_plan.eligible),
                "events_already_stored": len(event_plan.already_stored),
                "events_missing": len(event_plan.missing),
                "events_gaps_at_or_before_watermark": len(event_plan.gaps_at_or_before_watermark),
                "events_selected": len(selected),
                "latest_stored_fight_date": event_plan.latest_stored_fight_date,
                "future_events_ignored": sum(event.event_date > through_date for event in events),
                "events": [
                    {
                        "event_id": event.event_id,
                        "event_date": event.event_date,
                        "event_name": event.event_name,
                        "event_url": event.event_url,
                    }
                    for event in selected
                ],
                "raw_index_cached": True,
                "normalized_data_written": False,
            }
        )
    except (LatestDataFetcherError, UFCPredictorError, OSError, ValueError, sqlite3.Error) as exc:
        _abort(exc)


def _run_refresh(
    *,
    fetcher_config_path: Path,
    model_config_path: Path | None,
    since_date: date | None,
    through_date: date,
    max_events: int | None,
    fighter_directory: bool,
    refresh_pages: bool,
    refresh_known_fighters: bool,
    max_new_fighters: int | None,
    headed: bool,
    delay_seconds: float | None,
    allow_partial: bool,
) -> None:
    fetcher_config, model_config = _load(fetcher_config_path, model_config_path)
    if since_date is not None and since_date > through_date:
        raise ValueError("since cannot be later than through")
    if through_date > date.today():
        raise ValueError("through cannot be later than today")
    browser, crawler = _crawler(
        fetcher_config,
        model_config,
        headed=headed,
        delay_seconds=delay_seconds,
    )
    with browser:
        result = crawler.refresh(
            since=since_date,
            through=through_date,
            max_events=max_events,
            include_fighter_directory=fighter_directory,
            refresh_pages=refresh_pages,
            refresh_known_fighters=refresh_known_fighters,
            max_new_fighters=max_new_fighters,
        )
    _emit(result)
    if result["status"] != "complete" and not allow_partial:
        typer.echo("Refresh was not complete; inspect its blockers and run manifest.", err=True)
        raise typer.Exit(code=2)


@app.command("refresh")
def refresh_command(
    fetcher_config_path: Path = typer.Option(
        Path("configs/latestdatafetcher.yaml"),
        "--fetcher-config",
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    model_config_path: Path | None = typer.Option(
        None,
        "--model-config",
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    since: str | None = typer.Option(
        None,
        "--since",
        help="Inclusive YYYY-MM-DD; defaults to the day after the model cutoff.",
    ),
    through: str | None = typer.Option(None, "--through", help="Defaults to today."),
    max_events: int | None = typer.Option(None, "--max-events", min=1),
    fighter_directory: bool = typer.Option(
        True,
        "--fighter-directory/--skip-fighter-directory",
        help="Crawl A-Z indexes and fetch absent or incomplete baseline profiles.",
    ),
    refresh_pages: bool = typer.Option(
        False,
        "--refresh-detail-pages/--reuse-detail-cache",
        help=(
            "Revisit stored events and refetch event/fight pages; by default SQLite event IDs "
            "are skipped. The completed-event index is always refreshed."
        ),
    ),
    refresh_known_fighters: bool = typer.Option(
        False,
        "--refresh-known-fighters/--new-fighters-only",
        help="Refetch all discovered profiles; this can make thousands of requests.",
    ),
    max_new_fighters: int | None = typer.Option(None, "--max-new-fighters", min=1),
    headed: bool = typer.Option(False, "--headed", help="Show the Chromium window."),
    delay_seconds: float | None = typer.Option(None, "--delay-seconds", min=0.0),
    allow_partial: bool = typer.Option(False, "--allow-partial/--require-complete"),
) -> None:
    """Fetch missing completed events newer than the configured model snapshot."""

    try:
        _, model_config = _load(fetcher_config_path, model_config_path)
        since_date = _parse_date(since, field_name="since") or (
            model_config.data.dataset_cutoff + timedelta(days=1)
        )
        through_date = _parse_date(through, field_name="through") or date.today()
        _run_refresh(
            fetcher_config_path=fetcher_config_path,
            model_config_path=model_config_path,
            since_date=since_date,
            through_date=through_date,
            max_events=max_events,
            fighter_directory=fighter_directory,
            refresh_pages=refresh_pages,
            refresh_known_fighters=refresh_known_fighters,
            max_new_fighters=max_new_fighters,
            headed=headed,
            delay_seconds=delay_seconds,
            allow_partial=allow_partial,
        )
    except (LatestDataFetcherError, UFCPredictorError, OSError, ValueError, sqlite3.Error) as exc:
        _abort(exc)


@app.command("backfill")
def backfill_command(
    fetcher_config_path: Path = typer.Option(
        Path("configs/latestdatafetcher.yaml"),
        "--fetcher-config",
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    model_config_path: Path | None = typer.Option(
        None,
        "--model-config",
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    through: str | None = typer.Option(
        None,
        "--through",
        help="Defaults to the configured model cutoff.",
    ),
    max_events: int | None = typer.Option(None, "--max-events", min=1),
    fighter_directory: bool = typer.Option(True, "--fighter-directory/--skip-fighter-directory"),
    refresh_pages: bool = typer.Option(
        False,
        "--refresh-detail-pages/--reuse-detail-cache",
        help=(
            "Revisit stored events and refetch event/fight pages; by default SQLite event IDs "
            "are skipped. The completed-event index is always refreshed."
        ),
    ),
    headed: bool = typer.Option(False, "--headed", help="Show the Chromium window."),
    delay_seconds: float | None = typer.Option(None, "--delay-seconds", min=0.0),
    allow_partial: bool = typer.Option(False, "--allow-partial/--require-complete"),
) -> None:
    """Crawl completed history through the current cutoff, preserving non-label bouts."""

    try:
        _, model_config = _load(fetcher_config_path, model_config_path)
        through_date = (
            _parse_date(through, field_name="through") or model_config.data.dataset_cutoff
        )
        _run_refresh(
            fetcher_config_path=fetcher_config_path,
            model_config_path=model_config_path,
            since_date=None,
            through_date=through_date,
            max_events=max_events,
            fighter_directory=fighter_directory,
            refresh_pages=refresh_pages,
            refresh_known_fighters=False,
            max_new_fighters=None,
            headed=headed,
            delay_seconds=delay_seconds,
            allow_partial=allow_partial,
        )
    except (LatestDataFetcherError, UFCPredictorError, OSError, ValueError, sqlite3.Error) as exc:
        _abort(exc)


@app.command("validate")
def validate_command(
    fetcher_config_path: Path = typer.Option(
        Path("configs/latestdatafetcher.yaml"),
        "--fetcher-config",
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    model_config_path: Path | None = typer.Option(
        None,
        "--model-config",
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    require_fighter_directory: bool = typer.Option(
        True,
        "--require-fighter-directory/--allow-missing-fighter-directory",
        help="Require all A-Z source pages and a profile for every indexed fighter.",
    ),
) -> None:
    """Validate normalized data and separately report the 71-feature source mapping."""

    try:
        fetcher_config, model_config = _load(fetcher_config_path, model_config_path)
        result = validate_local_repository(
            LocalRepository(fetcher_config.storage),
            feature_dictionary_path=model_config.data.feature_dictionary_path,
            expected_feature_count=model_config.data.expected_feature_count,
            require_fighter_directory_complete=require_fighter_directory,
            minimum_fighter_directory_count=(fetcher_config.source.minimum_fighter_directory_count),
        )
        _emit(result)
    except (LatestDataFetcherError, UFCPredictorError, OSError, ValueError, sqlite3.Error) as exc:
        _abort(exc)


@app.command("status")
def status_command(
    fetcher_config_path: Path = typer.Option(
        Path("configs/latestdatafetcher.yaml"),
        "--fetcher-config",
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
) -> None:
    """Report local normalized record counts without network access."""

    try:
        fetcher_config = load_fetcher_config(fetcher_config_path)
        repository = LocalRepository(fetcher_config.storage)
        manifests = sorted(
            fetcher_config.storage.manifest_dir.glob("run-*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        latest_run: dict[str, Any] | None = None
        if manifests:
            manifest_payload = json.loads(manifests[0].read_text(encoding="utf-8"))
            if isinstance(manifest_payload, dict):
                latest_run = {
                    key: manifest_payload.get(key)
                    for key in (
                        "run_id",
                        "status",
                        "started_at_utc",
                        "completed_at_utc",
                        "through",
                        "fatal_error",
                    )
                }
                latest_run["manifest_path"] = str(manifests[0])
        _emit(
            {
                "status": "available",
                "database_path": str(repository.database_path),
                "normalized_dir": str(fetcher_config.storage.normalized_dir),
                "counts": repository.counts(),
                "latest_stored_fight_date": repository.latest_stored_fight_date(),
                "latest_run": latest_run,
                "processed_training_assets_modified": False,
            }
        )
    except (LatestDataFetcherError, OSError, ValueError, sqlite3.Error) as exc:
        _abort(exc)


__all__ = ["app"]
