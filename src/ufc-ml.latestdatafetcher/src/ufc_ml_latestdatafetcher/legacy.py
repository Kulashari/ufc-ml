"""Legacy-shaped import/export for candidate review, not processed training data."""

from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from ufc_ml_latestdatafetcher.errors import DatasetValidationError
from ufc_ml_latestdatafetcher.models import FighterProfile
from ufc_ml_latestdatafetcher.storage import (
    LocalRepository,
    atomic_write_csv,
    atomic_write_json,
    read_csv_rows,
)

LEGACY_FIGHT_COLUMNS = (
    "Fight_URL",
    "Fighter_1",
    "Fighter_2",
    "Winner",
    "Weight_Class",
    "Method",
    "End_Round",
    "End_Time",
    "Total_Fight_Time_Sec",
    "Time_Format",
    "F1_KD",
    "F2_KD",
    "F1_Sig_Landed",
    "F1_Sig_Att",
    "F2_Sig_Landed",
    "F2_Sig_Att",
    "F1_TD_Landed",
    "F2_TD_Landed",
    "F1_TD_Att",
    "F2_TD_Att",
    "F1_Sub_Att",
    "F2_Sub_Att",
    "F1_Ctrl_Sec",
    "F2_Ctrl_Sec",
    "F1_Head",
    "F2_Head",
    "F1_Body",
    "F2_Body",
    "F1_Leg",
    "F2_Leg",
    "F1_Distance",
    "F2_Distance",
    "F1_Clinch",
    "F2_Clinch",
    "F1_Ground",
    "F2_Ground",
    "Event_Date",
)
LEGACY_FIGHTER_COLUMNS = (
    "Fighter_Name",
    "Height",
    "Weight",
    "Reach",
    "Stance",
    "DOB",
    "Wins",
    "Losses",
    "Draws",
    "SLpM",
    "Str_Acc",
    "SApM",
    "Str_Def",
    "TD_Avg",
    "TD_Acc",
    "TD_Def",
    "Sub_Avg",
    "Fighter_URL",
)


def _fighter_id_from_url(url: str) -> str:
    value = urlparse(url).path.rstrip("/").rsplit("/", maxsplit=1)[-1]
    if re.fullmatch(r"[0-9a-f]{16}", value) is None:
        raise DatasetValidationError(f"invalid fighter URL in baseline data: {url!r}")
    return value


def _optional_float(value: str | None) -> float | None:
    normalized = str(value or "").strip().rstrip('"')
    if not normalized or normalized in {"--", "---"}:
        return None
    if normalized.endswith("%"):
        normalized = normalized[:-1]
        return float(normalized) / 100.0
    return float(normalized)


def _height(value: str | None) -> float | None:
    normalized = str(value or "").strip()
    if not normalized:
        return None
    match = re.fullmatch(r"(\d+)\s*'\s*(\d+(?:\.\d+)?)\s*\"", normalized)
    if match is None:
        return None
    return int(match.group(1)) * 12 + float(match.group(2))


def _weight(value: str | None) -> float | None:
    match = re.search(r"\d+(?:\.\d+)?", str(value or ""))
    return float(match.group()) if match else None


def profiles_from_legacy(path: Path) -> tuple[FighterProfile, ...]:
    """Seed stable bios so an incremental run need only fetch newly discovered fighters."""

    profiles: list[FighterProfile] = []
    for row in read_csv_rows(path):
        url = row.get("Fighter_URL", "").strip()
        if not url:
            continue
        fighter_id = _fighter_id_from_url(url)
        raw_dob = row.get("DOB", "").strip()
        dob = datetime.strptime(raw_dob, "%Y-%m-%d").date() if raw_dob else None
        profiles.append(
            FighterProfile(
                fighter_id=fighter_id,
                fighter_name=row.get("Fighter_Name", "").strip(),
                nickname="",
                height_inches=_height(row.get("Height")),
                weight_lbs=_weight(row.get("Weight")),
                reach_inches=_optional_float(row.get("Reach")),
                stance=row.get("Stance", "").strip() or "Unknown",
                dob=dob,
                wins=int(row["Wins"]) if row.get("Wins", "").strip() else None,
                losses=int(row["Losses"]) if row.get("Losses", "").strip() else None,
                draws=int(row["Draws"]) if row.get("Draws", "").strip() else None,
                no_contests=None,
                slpm_current=_optional_float(row.get("SLpM")),
                striking_accuracy_current=_optional_float(row.get("Str_Acc")),
                sapm_current=_optional_float(row.get("SApM")),
                striking_defense_current=_optional_float(row.get("Str_Def")),
                takedown_average_current=_optional_float(row.get("TD_Avg")),
                takedown_accuracy_current=_optional_float(row.get("TD_Acc")),
                takedown_defense_current=_optional_float(row.get("TD_Def")),
                submission_average_current=_optional_float(row.get("Sub_Avg")),
                fighter_url=url.replace("https://ufcstats.com", "http://ufcstats.com"),
                profile_as_of_utc=None,
                profile_source_sha256=None,
                profile_origin="baseline",
            )
        )
    return tuple(profiles)


def seed_fighters_from_baseline(repository: LocalRepository, path: Path) -> int:
    if not path.is_file():
        return 0
    known = repository.known_fighter_ids()
    missing_profiles = tuple(
        profile for profile in profiles_from_legacy(path) if profile.fighter_id not in known
    )
    if missing_profiles:
        repository.merge_fighters(missing_profiles)
    return len(missing_profiles)


def _blank_none(value: str | None) -> str:
    return "" if value in {None, ""} else str(value)


def _legacy_fight_rows(repository: LocalRepository) -> list[dict[str, Any]]:
    fights = {row["fight_id"]: row for row in read_csv_rows(repository.fights_path)}
    totals_by_fight: dict[str, dict[str, dict[str, str]]] = {}
    for row in read_csv_rows(repository.totals_path):
        totals_by_fight.setdefault(row["fight_id"], {})[row["fighter_id"]] = row
    output: list[dict[str, Any]] = []
    for fight in fights.values():
        totals = totals_by_fight.get(fight["fight_id"], {})
        first = totals.get(fight["fighter_1_id"], {})
        second = totals.get(fight["fighter_2_id"], {})
        winner = fight.get("winner_name", "")
        if not winner and (
            (fight.get("fighter_1_status"), fight.get("fighter_2_status"))
            in {("D", "D"), ("NC", "NC")}
        ):
            winner = "Draw/NC"
        output.append(
            {
                "Fight_URL": fight["fight_url"],
                "Fighter_1": fight["fighter_1_name"],
                "Fighter_2": fight["fighter_2_name"],
                "Winner": winner,
                "Weight_Class": fight["raw_weight_class"],
                "Method": fight["method"],
                "End_Round": fight["end_round"],
                "End_Time": fight["end_time"],
                "Total_Fight_Time_Sec": fight["total_fight_time_sec"],
                "Time_Format": fight["time_format"],
                "F1_KD": first.get("kd", ""),
                "F2_KD": second.get("kd", ""),
                "F1_Sig_Landed": first.get("sig_str_landed", ""),
                "F1_Sig_Att": first.get("sig_str_attempted", ""),
                "F2_Sig_Landed": second.get("sig_str_landed", ""),
                "F2_Sig_Att": second.get("sig_str_attempted", ""),
                "F1_TD_Landed": first.get("td_landed", ""),
                "F2_TD_Landed": second.get("td_landed", ""),
                "F1_TD_Att": first.get("td_attempted", ""),
                "F2_TD_Att": second.get("td_attempted", ""),
                "F1_Sub_Att": first.get("sub_attempts", ""),
                "F2_Sub_Att": second.get("sub_attempts", ""),
                "F1_Ctrl_Sec": first.get("control_seconds", ""),
                "F2_Ctrl_Sec": second.get("control_seconds", ""),
                "F1_Head": first.get("head_landed", ""),
                "F2_Head": second.get("head_landed", ""),
                "F1_Body": first.get("body_landed", ""),
                "F2_Body": second.get("body_landed", ""),
                "F1_Leg": first.get("leg_landed", ""),
                "F2_Leg": second.get("leg_landed", ""),
                "F1_Distance": first.get("distance_landed", ""),
                "F2_Distance": second.get("distance_landed", ""),
                "F1_Clinch": first.get("clinch_landed", ""),
                "F2_Clinch": second.get("clinch_landed", ""),
                "F1_Ground": first.get("ground_landed", ""),
                "F2_Ground": second.get("ground_landed", ""),
                "Event_Date": fight["event_date"],
            }
        )
    return output


def _format_height(value: str) -> str:
    if not value:
        return ""
    inches = float(value)
    feet, remainder = divmod(inches, 12)
    return f"{int(feet)}' {remainder:g}\""


def _format_percent(value: str) -> str:
    return f"{float(value) * 100:g}%" if value else ""


def _legacy_fighter_rows(repository: LocalRepository) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for fighter in read_csv_rows(repository.fighters_path):
        if fighter.get("profile_origin") == "baseline":
            # Baseline rows are already present in the legacy file. Re-exporting
            # their normalized representation would change missing-value spelling
            # and formatting despite no new source observation.
            continue
        output.append(
            {
                "Fighter_Name": fighter["fighter_name"],
                "Height": _format_height(fighter.get("height_inches", "")),
                "Weight": (
                    f"{float(fighter['weight_lbs']):g} lbs."
                    if fighter.get("weight_lbs", "")
                    else ""
                ),
                "Reach": (
                    f'{float(fighter["reach_inches"]):.1f}"'
                    if fighter.get("reach_inches", "")
                    else ""
                ),
                "Stance": fighter.get("stance", ""),
                "DOB": fighter.get("dob", ""),
                "Wins": fighter.get("wins", ""),
                "Losses": fighter.get("losses", ""),
                "Draws": fighter.get("draws", ""),
                "SLpM": fighter.get("slpm_current", ""),
                "Str_Acc": _format_percent(fighter.get("striking_accuracy_current", "")),
                "SApM": fighter.get("sapm_current", ""),
                "Str_Def": _format_percent(fighter.get("striking_defense_current", "")),
                "TD_Avg": fighter.get("takedown_average_current", ""),
                "TD_Acc": _format_percent(fighter.get("takedown_accuracy_current", "")),
                "TD_Def": _format_percent(fighter.get("takedown_defense_current", "")),
                "Sub_Avg": fighter.get("submission_average_current", ""),
                "Fighter_URL": fighter["fighter_url"],
            }
        )
    return output


def _merged_rows(
    baseline_path: Path,
    new_rows: list[dict[str, Any]],
    *,
    key: str,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {
        str(row[key]): dict(row) for row in read_csv_rows(baseline_path) if row.get(key)
    }
    for row in new_rows:
        merged[str(row[key])] = row
    return list(merged.values())


def export_legacy_candidates(
    repository: LocalRepository,
    *,
    output_dir: Path | None = None,
) -> dict[str, str | int]:
    """Atomically publish legacy-shaped candidates into a new directory.

    The normalized ID-rich tables remain the authoritative acquisition output.
    These files preserve the old column shape only and are not proof that the
    missing raw-to-feature transformation is reproducible.
    """

    finalized_dir = (output_dir or repository.config.candidate_dir).expanduser().resolve()
    finalized_dir.parent.mkdir(parents=True, exist_ok=True)
    if finalized_dir.exists():
        raise DatasetValidationError(
            f"candidate output directory already exists and will not be overwritten: "
            f"{finalized_dir}"
        )
    staging_dir = finalized_dir.with_name(f".{finalized_dir.name}.staging-{uuid4().hex}")
    staging_dir.mkdir(parents=False, exist_ok=False)
    fight_rows = _legacy_fight_rows(repository)
    fighter_rows = _legacy_fighter_rows(repository)
    staged_latest_fights = staging_dir / "ufc_gold_dataset_latest.csv"
    staged_latest_fighters = staging_dir / "ufc_fighters_latest.csv"
    staged_refreshed_fights = staging_dir / "ufc_gold_dataset_refreshed.csv"
    staged_refreshed_fighters = staging_dir / "ufc_fighters_refreshed.csv"
    staged_normalized_dir = staging_dir / "normalized"
    merged_fights = _merged_rows(
        repository.config.baseline_fights_path,
        fight_rows,
        key="Fight_URL",
    )
    merged_fighters = _merged_rows(
        repository.config.baseline_fighters_path,
        fighter_rows,
        key="Fighter_URL",
    )
    try:
        atomic_write_csv(
            staged_latest_fights,
            fight_rows,
            fieldnames=LEGACY_FIGHT_COLUMNS,
        )
        atomic_write_csv(
            staged_latest_fighters,
            fighter_rows,
            fieldnames=LEGACY_FIGHTER_COLUMNS,
        )
        atomic_write_csv(
            staged_refreshed_fights,
            merged_fights,
            fieldnames=LEGACY_FIGHT_COLUMNS,
        )
        atomic_write_csv(
            staged_refreshed_fighters,
            merged_fighters,
            fieldnames=LEGACY_FIGHTER_COLUMNS,
        )
        staged_normalized_dir.mkdir(parents=False, exist_ok=False)
        repository.backup_database(staged_normalized_dir / "ufcstats.sqlite3")
        for normalized_path in repository.normalized_csv_paths:
            shutil.copy2(normalized_path, staged_normalized_dir / normalized_path.name)
        atomic_write_json(
            staging_dir / "bundle_metadata.json",
            {
                "authoritative_data": "normalized/ufcstats.sqlite3",
                "normalized_csvs": "derived human-readable snapshots",
                "legacy_exports": "column-shape compatibility only; not training-ready",
                "historical_feature_rule": (
                    "emit pre-fight state before applying each bout; never use *_current "
                    "fighter summaries for historical features"
                ),
                "processed_training_assets_modified": False,
                "model_retrained": False,
            },
        )
        if finalized_dir.exists():
            raise DatasetValidationError(
                f"candidate output directory appeared during export and will not be "
                f"overwritten: {finalized_dir}"
            )
        staging_dir.rename(finalized_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    latest_fights = finalized_dir / staged_latest_fights.name
    latest_fighters = finalized_dir / staged_latest_fighters.name
    refreshed_fights = finalized_dir / staged_refreshed_fights.name
    refreshed_fighters = finalized_dir / staged_refreshed_fighters.name
    normalized_dir = finalized_dir / staged_normalized_dir.name
    return {
        "output_dir": str(finalized_dir),
        "compatibility_scope": "legacy_column_shape_only_not_training_ready",
        "authoritative_database_path": str(normalized_dir / "ufcstats.sqlite3"),
        "normalized_snapshot_dir": str(normalized_dir),
        "bundle_metadata_path": str(finalized_dir / "bundle_metadata.json"),
        "latest_fights_path": str(latest_fights),
        "latest_fight_count": len(fight_rows),
        "latest_fighters_path": str(latest_fighters),
        "latest_fighter_count": len(fighter_rows),
        "refreshed_fights_path": str(refreshed_fights),
        "refreshed_fight_count": len(merged_fights),
        "refreshed_fighters_path": str(refreshed_fighters),
        "refreshed_fighter_count": len(merged_fighters),
    }


__all__ = [
    "LEGACY_FIGHTER_COLUMNS",
    "LEGACY_FIGHT_COLUMNS",
    "export_legacy_candidates",
    "profiles_from_legacy",
    "seed_fighters_from_baseline",
]
