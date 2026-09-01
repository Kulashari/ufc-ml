"""Pure UFCStats HTML parsers."""

from ufc_ml_latestdatafetcher.parsing.completed_events import parse_completed_events
from ufc_ml_latestdatafetcher.parsing.event import parse_event
from ufc_ml_latestdatafetcher.parsing.fight import parse_fight
from ufc_ml_latestdatafetcher.parsing.fighter import (
    parse_fighter_directory,
    parse_fighter_profile,
)

__all__ = [
    "parse_completed_events",
    "parse_event",
    "parse_fight",
    "parse_fighter_directory",
    "parse_fighter_profile",
]
