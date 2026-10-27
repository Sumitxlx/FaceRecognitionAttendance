from datetime import datetime, timezone


def utc_now_naive() -> datetime:
    """Return UTC time for legacy SQLite DATETIME columns without timezone support."""
    return datetime.now(timezone.utc).replace(tzinfo=None)
