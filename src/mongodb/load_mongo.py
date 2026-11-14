"""
load_mongo.py - Load categories, products, users and transactions into MongoDB.

Document model (see docs/report for the full justification):

  categories   { _id: "cat_007", name, subcategories: [ {subcategory_id, name, profit_margin} ] }
               Subcategories are EMBEDDED: 3-5 per category, always read with their parent,
               never queried on their own -> bounded one-to-few relationship.

  products     { _id: "prod_00123", name, category: {id, name}, subcategory: {id, name, profit_margin},
                 base_price, current_stock, is_active, price_history: [ {price, date} ], creation_date }
               Category / subcategory names and margin are DENORMALISED (extended reference pattern):
               the catalog is read far more often than categories change, so product pages and
               revenue/profit pipelines need no $lookup. price_history is embedded (1-3 entries, bounded).

  users        { _id: "user_000042", geo: {city, state, country}, registration_date, last_active,
                 purchase_summary: { order_count, total_spent, avg_order_value, first_purchase,
                                     last_purchase, favourite_categories: [ ... ] } }
               The summary is a COMPUTED / SUBSET pattern: frequently-needed aggregates are embedded,
               while the unbounded list of orders stays in `transactions` (avoids 16 MB doc growth).

  transactions { _id: "txn_...", session_id, user_id, timestamp (Date), status, payment_method,
                 subtotal, discount, total,
                 items: [ {product_id, category_id, quantity, unit_price, subtotal} ] }
               Line items are EMBEDDED: they are always written and read together with the order
               (atomic single-document write). category_id is copied onto each item at load time so
               revenue-by-category needs only $unwind + $group.

Sessions are deliberately NOT stored here: the high-volume, append-only, time-ordered clickstream
goes to HBase (see src/hbase). MongoDB keeps the rich, entity-centric documents.

Usage:
    python src/mongodb/load_mongo.py                      # full generated dataset in data/raw
    python src/mongodb/load_mongo.py --limit-transactions 20000
"""

import argparse
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import BulkWriteError, ServerSelectionTimeoutError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import DEFAULT_DATA_DIR, MONGO_DB, MONGO_URI, batched, iter_records  # noqa: E402

BATCH = 5_000


def to_date(s):
    return datetime.fromisoformat(s) if s else None


def insert_batches(coll, docs, label):
    n = 0
    for batch in batched(docs, BATCH):
        try:
            coll.insert_many(batch, ordered=False)
        except BulkWriteError as e:  # e.g. duplicate keys on a re-run without --drop
            dup = sum(1 for err in e.details["writeErrors"] if err["code"] == 11000)
            print(f"  {label}: skipped {dup} duplicate(s)")
        n += len(batch)
    print(f"  {label}: {n:,} documents")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA_DIR)
    ap.add_argument("--limit-transactions", type=int, default=None, help="load only the first N transactions")
    ap.add_argument("--no-drop", action="store_true", help="keep existing collections")
    args = ap.parse_args()

    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
    except ServerSelectionTimeoutError:
        sys.exit(f"Cannot reach MongoDB at {MONGO_URI}. Is the container running? (docker compose up -d)")
    db = client[MONGO_DB]
    if not args.no_drop:
        for c in ("categories", "products", "users", "transactions"):
            db.drop_collection(c)
    t0 = time.time()
    print(f"Loading {args.data} into {MONGO_URI}/{MONGO_DB}")

    # ---- categories (embedded subcategories)
    categories = list(iter_records(args.data, "categories"))
    insert_batches(db.categories, ({"_id": c["category_id"], "name": c["name"],
                                    "subcategories": c["subcategories"]} for c in categories), "categories")
    cat_name = {c["category_id"]: c["name"] for c in categories}
    sub_info = {s["subcategory_id"]: s for c in categories for s in c["subcategories"]}

    # ---- products (extended reference to category / subcategory)
    product_cat = {}

    def product_docs():
        for p in iter_records(args.data, "products"):
            product_cat[p["product_id"]] = p["category_id"]
            sub = sub_info.get(p.get("subcategory_id"), {})
            yield {
                "_id": p["product_id"],
                "name": p["name"],
                "category": {"id": p["category_id"], "name": cat_name.get(p["category_id"])},
                "subcategory": {"id": p.get("subcategory_id"), "name": sub.get("name"),
                                "profit_margin": sub.get("profit_margin")},
                "base_price": p["base_price"],
                "current_stock": p["current_stock"],
                "is_active": p["is_active"],
                "price_history": [{"price": h["price"], "date": to_date(h["date"])} for h in p["price_history"]],
                "creation_date": to_date(p["creation_date"]),
            }
    insert_batches(db.products, product_docs(), "products")

    # ---- transactions (embedded line items) + build per-user summaries on the way
    summary = defaultdict(lambda: {"order_count": 0, "total_spent": 0.0, "first": None, "last": None,
                                   "cats": Counter()})

    def txn_docs():
        for t in iter_records(args.data, "transactions", args.limit_transactions):
            ts = to_date(t["timestamp"])
            s = summary[t["user_id"]]
            s["order_count"] += 1
            s["total_spent"] += t["total"]
            s["first"] = ts if s["first"] is None or ts < s["first"] else s["first"]
            s["last"] = ts if s["last"] is None or ts > s["last"] else s["last"]
            items = []
            for it in t["items"]:
                cid = product_cat.get(it["product_id"])
                s["cats"][cid] += it["subtotal"]
                items.append({**it, "category_id": cid})
            yield {
                "_id": t["transaction_id"],
                "session_id": t["session_id"],
                "user_id": t["user_id"],
                "timestamp": ts,
                "items": items,
                "subtotal": t["subtotal"],
                "discount": t["discount"],
                "total": t["total"],
                "payment_method": t["payment_method"],
                "status": t["status"],
            }
    insert_batches(db.transactions, txn_docs(), "transactions")

    # ---- users (embedded purchase summary)
    def user_docs():
        for u in iter_records(args.data, "users"):
            s = summary.get(u["user_id"])
            ps = {"order_count": 0, "total_spent": 0.0, "avg_order_value": 0.0,
                  "first_purchase": None, "last_purchase": None, "favourite_categories": []}
            if s:
                ps = {
                    "order_count": s["order_count"],
                    "total_spent": round(s["total_spent"], 2),
                    "avg_order_value": round(s["total_spent"] / s["order_count"], 2),
                    "first_purchase": s["first"],
                    "last_purchase": s["last"],
                    "favourite_categories": [{"category_id": c, "name": cat_name.get(c), "spent": round(v, 2)}
                                             for c, v in s["cats"].most_common(3)],
                }
            yield {"_id": u["user_id"], "geo": u["geo_data"],
                   "registration_date": to_date(u["registration_date"]),
                   "last_active": to_date(u["last_active"]), "purchase_summary": ps}
    insert_batches(db.users, user_docs(), "users")

    # ---- indexes that back the query patterns used in aggregations.py / integration jobs
    db.transactions.create_index([("user_id", ASCENDING), ("timestamp", DESCENDING)])  # a user's order history
    db.transactions.create_index([("timestamp", ASCENDING)])                           # time-range revenue
    db.transactions.create_index([("items.product_id", ASCENDING)])                   # multikey: orders with product X
    db.transactions.create_index([("session_id", ASCENDING)], sparse=True)            # session -> order (funnel)
    db.products.create_index([("category.id", ASCENDING), ("subcategory.id", ASCENDING)])
    db.users.create_index([("purchase_summary.total_spent", DESCENDING)])
    db.users.create_index([("geo.country", ASCENDING)])
    print(f"Indexes created. Done in {time.time() - t0:.1f}s")
    for c in ("categories", "products", "users", "transactions"):
        print(f"  {c:<13} {db[c].estimated_document_count():>8,}")


if __name__ == "__main__":
    main()
