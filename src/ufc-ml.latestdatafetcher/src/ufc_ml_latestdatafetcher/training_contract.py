"""Documented compatibility rules inferred from the checked-in training snapshot."""

from __future__ import annotations

import re
from datetime import date

LABEL_START_DATE = date(2001, 2, 23)
STANDARD_LABEL_TIME_FORMATS = frozenset({"3 Rnd (5-5-5)", "5 Rnd (5-5-5-5-5)"})
_DECISIVE_STATUS_PAIRS = frozenset({("W", "L"), ("L", "W")})
_FIGHT_ID = re.compile(r"^[0-9a-f]{16}$")


def fighter_1_is_model_a(fight_id: str) -> bool:
    """Return the deterministic corner orientation used by the current model rows."""

    normalized = fight_id.strip().casefold()
    if _FIGHT_ID.fullmatch(normalized) is None:
        raise ValueError(f"invalid UFCStats fight ID: {fight_id!r}")
    return int(normalized[-1], 16) % 2 == 0


def is_legacy_label_eligible(
    *,
    event_date: date,
    fighter_1_status: str,
    fighter_2_status: str,
    time_format: str,
) -> bool:
    """Return whether a bout matches the current 8,116-row binary-label policy.

    Ineligible bouts still belong in chronological fighter history; this predicate
    controls label emission only.
    """

    return (
        event_date >= LABEL_START_DATE
        and (fighter_1_status, fighter_2_status) in _DECISIVE_STATUS_PAIRS
        and time_format in STANDARD_LABEL_TIME_FORMATS
    )


__all__ = [
    "LABEL_START_DATE",
    "STANDARD_LABEL_TIME_FORMATS",
    "fighter_1_is_model_a",
    "is_legacy_label_eligible",
]
