"""
query_hbase.py - Example HBase access patterns enabled by the row-key design.

  1. latest   : the N most recent sessions of a user      -> prefix scan + LIMIT (newest first)
  2. range    : a user's sessions inside a time window     -> bounded scan (row_start/row_stop)
  3. journey  : full page-view sequence of one session     -> single-row GET of family 'pv'
  4. product  : daily views / cart adds / purchases and view->purchase conversion for a product
                over a date range                           -> bounded scan on product_metrics

Usage:
    python src/hbase/query_hbase.py                         # demo with a sample user/product
    python src/hbase/query_hbase.py --user user_000042 --product prod_00123 \
        --start 2026-09-01 --end 2026-09-30
"""

import argparse
import json
import os
import sys
import time

import happybase

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import HBASE_HOST, HBASE_PORT  # noqa: E402
from hbase_schema import METRICS_TABLE, SESSIONS_TABLE, user_prefix, user_time_range  # noqa: E402


def d(b):
    return b.decode() if isinstance(b, bytes) else b


def latest_sessions(tbl, user_id, n=5):
    """Most recent N sessions; reads only the small 's' family."""
    return [(d(k), {d(c)[2:]: d(v) for c, v in row.items()})
            for k, row in tbl.scan(row_prefix=user_prefix(user_id), columns=[b"s"], limit=n)]


def sessions_in_range(tbl, user_id, start_iso, end_iso):
    row_start, row_stop = user_time_range(user_id, start_iso, end_iso)
    cols = [b"s:start_time", b"s:status", b"s:device", b"s:n_pages", b"s:referrer"]
    return [(d(k), {d(c)[2:]: d(v) for c, v in row.items()})
            for k, row in tbl.scan(row_start=row_start, row_stop=row_stop, columns=cols)]


def session_journey(tbl, row_key):
    row = tbl.row(row_key.encode(), columns=[b"pv", b"c"])
    pages = [json.loads(v) for c, v in sorted(row.items()) if c.startswith(b"pv:")]
    cart = {d(c)[2:]: d(v) for c, v in row.items() if c.startswith(b"c:")}
    return pages, cart


def product_metrics(tbl, product_id, start_day, end_day):
    rows = tbl.scan(row_start=f"{product_id}|{start_day}".encode(),
                    row_stop=f"{product_id}|{end_day}~".encode())
    out = []
    for k, row in rows:
        r = {d(c)[2:]: float(v) for c, v in row.items()}
        r["day"] = d(k).split("|")[1]
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default=None, help="user_id (default: first user found in the table)")
    ap.add_argument("--product", default="prod_00123")
    ap.add_argument("--start", default="2026-09-01")
    ap.add_argument("--end", default="2026-10-31")
    args = ap.parse_args()

    conn = happybase.Connection(HBASE_HOST, HBASE_PORT, timeout=60_000)
    sessions, metrics = conn.table(SESSIONS_TABLE), conn.table(METRICS_TABLE)
    user = args.user or d(next(sessions.scan(limit=1, columns=[b"s:user_id"]))[0]).split("|")[0]

    print(f"\n=== 1. Latest 5 sessions of {user} (prefix scan '{user}|', newest first) ===")
    t = time.perf_counter()
    latest = latest_sessions(sessions, user, 5)
    ms = (time.perf_counter() - t) * 1000
    for k, s in latest:
        print(f"  {s['start_time']}  {s['status']:<9} {s['device']:<7} pages={s['n_pages']:<3}"
              f" ref={s['referrer']:<13} key={k}")
    print(f"  ({len(latest)} rows in {ms:.1f} ms)")

    print(f"\n=== 2. {user}'s sessions between {args.start} and {args.end} (bounded scan) ===")
    t = time.perf_counter()
    rng = sessions_in_range(sessions, user, f"{args.start}T00:00:00", f"{args.end}T23:59:59")
    ms = (time.perf_counter() - t) * 1000
    for k, s in rng[:10]:
        print(f"  {s['start_time']}  {s['status']:<9} {s['device']:<7} pages={s['n_pages']}")
    if len(rng) > 10:
        print(f"  ... {len(rng) - 10} more")
    print(f"  ({len(rng)} rows in {ms:.1f} ms)")

    if latest:
        key = latest[0][0]
        print(f"\n=== 3. Journey of the most recent session ({key.split('|')[2]}) ===")
        pages, cart = session_journey(sessions, key)
        for p in pages:
            print(f"  {p['t'][11:]}  {p['type']:<16} product={p['p'] or '-':<11} {p['d']:>5}s")
        print(f"  cart: {cart or '(empty)'}")

    s_day, e_day = args.start.replace("-", ""), args.end.replace("-", "")
    print(f"\n=== 4. Daily metrics for {args.product}, {args.start} .. {args.end} (bounded scan) ===")
    t = time.perf_counter()
    m = product_metrics(metrics, args.product, s_day, e_day)
    ms = (time.perf_counter() - t) * 1000
    for r in m[:12]:
        print(f"  {r['day']}  views={r['views']:>3.0f}  cart_adds={r['cart_adds']:>3.0f}"
              f"  purchases={r['purchases']:>3.0f}  revenue={r['revenue']:>9.2f}")
    if len(m) > 12:
        print(f"  ... {len(m) - 12} more days")
    views = sum(r["views"] for r in m)
    purchases = sum(r["purchases"] for r in m)
    print(f"  ({len(m)} days in {ms:.1f} ms)  total views={views:.0f}  purchases={purchases:.0f}  "
          f"view->purchase rate={purchases / views:.1%}" if views else f"  ({len(m)} days) no views in range")
    conn.close()


if __name__ == "__main__":
    main()
