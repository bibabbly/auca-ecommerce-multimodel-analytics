"""
load_hbase.py - Load session clickstream into HBase and build daily product metrics.

  user_sessions   : one row per session, key <user_id>|<reverse_ts>|<session_id>
                    families s (metadata), pv (page views, one cell each), c (cart lines)
  product_metrics : one row per product per day, key <product_id>|<yyyymmdd>
                    family m: views, cart_adds, purchases, units_sold, revenue
                    (views/cart adds come from sessions, purchases from transactions)

Tables should first be created with the HBase shell script (gives pre-split regions):
    docker exec -i ecom-hbase hbase shell -n < src/hbase/create_tables.hbase
If they are missing, this script creates them through Thrift (without pre-splits).

Usage:
    python src/hbase/load_hbase.py                    # all generated sessions
    python src/hbase/load_hbase.py --limit 100000     # a representative subset
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import happybase

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import DEFAULT_DATA_DIR, HBASE_HOST, HBASE_PORT, iter_records  # noqa: E402
from hbase_schema import METRICS_TABLE, SESSIONS_TABLE, metrics_row_key, session_row_key  # noqa: E402

BATCH_SIZE = 2_000


def connect(retries=10):
    for attempt in range(1, retries + 1):
        try:
            conn = happybase.Connection(HBASE_HOST, HBASE_PORT, timeout=60_000, autoconnect=True)
            conn.tables()  # forces a round-trip
            return conn
        except Exception as e:  # thrift not up yet right after `docker compose up`
            print(f"  HBase Thrift not ready ({type(e).__name__}); retry {attempt}/{retries} in 5s")
            time.sleep(5)
    sys.exit(f"Cannot reach HBase Thrift at {HBASE_HOST}:{HBASE_PORT}. Is the container running?")


def ensure_tables(conn):
    existing = {t.decode() for t in conn.tables()}
    if SESSIONS_TABLE not in existing:
        print(f"  creating {SESSIONS_TABLE} via Thrift (no pre-splits; prefer create_tables.hbase)")
        conn.create_table(SESSIONS_TABLE, {"s": dict(max_versions=1, bloom_filter_type="ROW"),
                                           "pv": dict(max_versions=1, bloom_filter_type="ROW"),
                                           "c": dict(max_versions=1, bloom_filter_type="ROW")})
    if METRICS_TABLE not in existing:
        print(f"  creating {METRICS_TABLE} via Thrift (no pre-splits; prefer create_tables.hbase)")
        conn.create_table(METRICS_TABLE, {"m": dict(max_versions=1, bloom_filter_type="ROW")})


def session_cells(s):
    g, d = s["geo_data"], s["device_profile"]
    cells = {
        b"s:session_id": s["session_id"], b"s:user_id": s["user_id"],
        b"s:start_time": s["start_time"], b"s:end_time": s["end_time"],
        b"s:duration_s": str(s["duration_seconds"]),
        b"s:device": d["type"], b"s:os": d["os"], b"s:browser": d["browser"],
        b"s:city": g["city"], b"s:state": g["state"], b"s:country": g["country"], b"s:ip": g["ip_address"],
        b"s:referrer": s["referrer"], b"s:status": s["conversion_status"],
        b"s:n_pages": str(len(s["page_views"])),
        b"s:viewed_products": ",".join(s["viewed_products"]),
    }
    for i, pv in enumerate(s["page_views"]):
        cells[f"pv:{i:03d}".encode()] = json.dumps(
            {"t": pv["timestamp"], "type": pv["page_type"], "p": pv["product_id"],
             "c": pv["category_id"], "d": pv["view_duration"]}, separators=(",", ":"))
    for pid, line in s["cart_contents"].items():
        cells[f"c:{pid}".encode()] = f"{line['quantity']}|{line['price']}"
    return {k: v.encode() for k, v in cells.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA_DIR)
    ap.add_argument("--limit", type=int, default=None, help="load only the first N sessions")
    args = ap.parse_args()

    conn = connect()
    ensure_tables(conn)
    sessions_tbl = conn.table(SESSIONS_TABLE)
    metrics = defaultdict(lambda: {"views": 0, "cart_adds": 0, "purchases": 0, "units_sold": 0, "revenue": 0.0})
    loaded_users = set()
    t0 = time.time()

    print(f"Loading sessions from {args.data} into '{SESSIONS_TABLE}' ...")
    n = cells = 0
    with sessions_tbl.batch(batch_size=BATCH_SIZE) as b:
        for s in iter_records(args.data, "sessions", args.limit):
            data = session_cells(s)
            b.put(session_row_key(s["user_id"], s["start_time"], s["session_id"]), data)
            cells += len(data)
            loaded_users.add(s["user_id"])
            day = s["start_time"][:10].replace("-", "")
            for pv in s["page_views"]:
                if pv["page_type"] == "product_detail" and pv["product_id"]:
                    metrics[(pv["product_id"], pv["timestamp"][:10].replace("-", ""))]["views"] += 1
            for pid in s["cart_contents"]:
                metrics[(pid, day)]["cart_adds"] += 1
            n += 1
            if n % 50_000 == 0:
                print(f"  {n:,} sessions ({cells:,} cells) {n / (time.time() - t0):,.0f} rows/s")
    print(f"  {n:,} sessions, {cells:,} cells in {time.time() - t0:.0f}s")

    # purchases come from transactions (both session-linked and direct orders)
    for t in iter_records(args.data, "transactions"):
        day = t["timestamp"][:10].replace("-", "")
        for it in t["items"]:
            m = metrics[(it["product_id"], day)]
            m["purchases"] += 1
            m["units_sold"] += it["quantity"]
            m["revenue"] += it["subtotal"]

    print(f"Writing {len(metrics):,} product-day rows into '{METRICS_TABLE}' ...")
    with conn.table(METRICS_TABLE).batch(batch_size=BATCH_SIZE) as b:
        for (pid, day), m in metrics.items():
            b.put(metrics_row_key(pid, day), {
                b"m:views": str(m["views"]).encode(), b"m:cart_adds": str(m["cart_adds"]).encode(),
                b"m:purchases": str(m["purchases"]).encode(), b"m:units_sold": str(m["units_sold"]).encode(),
                b"m:revenue": f"{m['revenue']:.2f}".encode()})
    print(f"Done in {time.time() - t0:.0f}s. Users with sessions loaded: {len(loaded_users):,}")
    conn.close()


if __name__ == "__main__":
    main()
