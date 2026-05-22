"""Shared parsing helpers for rules and API mappers."""

from __future__ import annotations

from datetime import date, datetime

_DATE_FORMATS = ("%Y%m%d", "%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d")


def safe_float(value, default=None) -> float | None:
    """Parse a numeric value; return default for None/empty/invalid."""
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_date(value) -> date | None:
    """Parse DOB/lab dates from datetime, date, or common string formats."""
    if not value:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    raw = str(value).strip()
    if not raw:
        return None

    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def format_yyyymmdd_display(date_value) -> str:
    """
    Normalize date strings for display.
    Databricks YYYYMMDD values render as YYYY/MM/DD; other strings pass through.
    """
    if not date_value:
        return ""
    normalized = str(date_value).strip()
    if len(normalized) == 8 and normalized.isdigit():
        return f"{normalized[:4]}/{normalized[4:6]}/{normalized[6:]}"
    return normalized
