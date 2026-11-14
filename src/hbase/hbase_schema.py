"""Row-key helpers shared by the HBase loader and query scripts."""

from datetime import datetime, timezone

LONG_MAX = 2**63 - 1
SESSIONS_TABLE = "user_sessions"
METRICS_TABLE = "product_metrics"


def epoch_ms(iso):
    return int(datetime.fromisoformat(iso).replace(tzinfo=timezone.utc).timestamp() * 1000)


def reverse_ts(iso):
    """Newest first: larger timestamps produce smaller keys."""
    return f"{LONG_MAX - epoch_ms(iso):019d}"


def session_row_key(user_id, start_time, session_id):
    return f"{user_id}|{reverse_ts(start_time)}|{session_id}".encode()


def user_prefix(user_id):
    return f"{user_id}|".encode()


def user_time_range(user_id, start_iso, end_iso):
    """(row_start, row_stop) covering sessions that started in [start_iso, end_iso].
    Because keys are reversed, the END of the window gives the START row."""
    row_start = f"{user_id}|{reverse_ts(end_iso)}".encode()
    row_stop = f"{user_id}|{reverse_ts(start_iso)}~".encode()  # '~' sorts after '|<session_id>'
    return row_start, row_stop


def metrics_row_key(product_id, day):
    """day: 'YYYYMMDD'"""
    return f"{product_id}|{day}".encode()
