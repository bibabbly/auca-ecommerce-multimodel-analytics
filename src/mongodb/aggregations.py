"""
aggregations.py - MongoDB aggregation pipelines (Part 1 analytics).

  1. revenue_by_category_month : net revenue + estimated gross profit per category per month.
     Uses $unwind on embedded line items, allocates the order-level discount to each line,
     $lookup's the product's subcategory profit margin, then $group / $sort.
  2. user_segments             : RFM-style customer segmentation on the embedded
     users.purchase_summary, with $facet producing three views in a single pass
     (frequency buckets, spend buckets, named segments).
  3. top_products              : top-selling products by units and revenue, joined back to the
     catalog for name / category / remaining stock (inventory-risk view).

Results are printed and saved as JSON in output/mongo/ (used by the visualisation step).

Usage:  python src/mongodb/aggregations.py
"""

import json
import os
import sys
from datetime import timedelta

from pymongo import MongoClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import MONGO_DB, MONGO_URI, PROJECT_ROOT  # noqa: E402

OUT_DIR = os.path.join(PROJECT_ROOT, "output", "mongo")


# --------------------------------------------------------------------------- pipeline 1
def revenue_by_category_month():
    return [
        {"$match": {"status": {"$in": ["completed", "shipped", "delivered", "processing"]}}},
        # share of the order-level discount carried by each line: net = line_subtotal * total/subtotal
        {"$addFields": {"net_ratio": {"$cond": [{"$gt": ["$subtotal", 0]},
                                                 {"$divide": ["$total", "$subtotal"]}, 1]}}},
        {"$unwind": "$items"},
        {"$lookup": {"from": "products", "localField": "items.product_id", "foreignField": "_id",
                     "pipeline": [{"$project": {"subcategory.profit_margin": 1, "category.name": 1}}],
                     "as": "product"}},
        {"$unwind": "$product"},
        {"$project": {
            "month": {"$dateToString": {"format": "%Y-%m", "date": "$timestamp"}},
            "category_id": "$items.category_id",
            "category_name": "$product.category.name",
            "units": "$items.quantity",
            "net_revenue": {"$multiply": ["$items.subtotal", "$net_ratio"]},
            "est_profit": {"$multiply": ["$items.subtotal", "$net_ratio",
                                         {"$ifNull": ["$product.subcategory.profit_margin", 0]}]},
            "order_id": "$_id",
        }},
        {"$group": {
            "_id": {"category_id": "$category_id", "month": "$month"},
            "category_name": {"$first": "$category_name"},
            "units": {"$sum": "$units"},
            "net_revenue": {"$sum": "$net_revenue"},
            "est_profit": {"$sum": "$est_profit"},
            "orders": {"$addToSet": "$order_id"},
        }},
        {"$project": {
            "_id": 0, "category_id": "$_id.category_id", "month": "$_id.month", "category_name": 1,
            "units": 1, "orders": {"$size": "$orders"},
            "net_revenue": {"$round": ["$net_revenue", 2]},
            "est_profit": {"$round": ["$est_profit", 2]},
            "margin_pct": {"$round": [{"$multiply": [{"$divide": ["$est_profit", "$net_revenue"]}, 100]}, 1]},
        }},
        {"$sort": {"month": 1, "net_revenue": -1}},
    ]


# --------------------------------------------------------------------------- pipeline 2
def user_segments(reference_date):
    """RFM-style segmentation. reference_date = last day covered by the data."""
    at_risk_cutoff = reference_date - timedelta(days=45)
    ps = "$purchase_summary"
    return [
        {"$addFields": {
            "segment": {"$switch": {"branches": [
                {"case": {"$eq": [f"{ps}.order_count", 0]}, "then": "1. Never purchased"},
                {"case": {"$lt": [f"{ps}.last_purchase", at_risk_cutoff]}, "then": "2. At risk (no order 45d+)"},
                {"case": {"$gte": [f"{ps}.order_count", 12]}, "then": "5. Champions (12+ orders)"},
                {"case": {"$gte": [f"{ps}.order_count", 7]}, "then": "4. Loyal (7-11 orders)"},
            ], "default": "3. Occasional (1-6 orders)"}},
        }},
        {"$facet": {
            "by_frequency": [
                {"$bucket": {"groupBy": f"{ps}.order_count",
                             "boundaries": [0, 1, 3, 5, 7, 9, 12, 16, 1000], "default": "other",
                             "output": {"users": {"$sum": 1},
                                        "avg_spent": {"$avg": f"{ps}.total_spent"}}}},
            ],
            "by_spend": [
                {"$bucket": {"groupBy": f"{ps}.total_spent",
                             "boundaries": [0, 0.01, 500, 1000, 2500, 5000, 10000, 1e9], "default": "other",
                             "output": {"users": {"$sum": 1},
                                        "avg_orders": {"$avg": f"{ps}.order_count"}}}},
            ],
            "segments": [
                {"$group": {"_id": "$segment", "users": {"$sum": 1},
                            "revenue": {"$sum": f"{ps}.total_spent"},
                            "avg_order_value": {"$avg": f"{ps}.avg_order_value"},
                            "avg_orders": {"$avg": f"{ps}.order_count"}}},
                {"$sort": {"_id": 1}},
            ],
        }},
    ]


# --------------------------------------------------------------------------- pipeline 3
def top_products(limit=15):
    return [
        {"$unwind": "$items"},
        {"$group": {"_id": "$items.product_id",
                    "units_sold": {"$sum": "$items.quantity"},
                    "revenue": {"$sum": "$items.subtotal"},
                    "orders": {"$sum": 1}}},
        {"$sort": {"revenue": -1}},
        {"$limit": limit},
        {"$lookup": {"from": "products", "localField": "_id", "foreignField": "_id", "as": "p"}},
        {"$unwind": "$p"},
        {"$project": {"_id": 0, "product_id": "$_id", "name": "$p.name", "category": "$p.category.name",
                      "units_sold": 1, "orders": 1, "revenue": {"$round": ["$revenue", 2]},
                      "current_stock": "$p.current_stock", "base_price": "$p.base_price"}},
    ]


# --------------------------------------------------------------------------- runner
def show(rows, cols, n=15):
    widths = {c: max(len(c), *(len(f"{r.get(c, '')}") for r in rows[:n])) for c in cols} if rows else {}
    print("  " + "  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows[:n]:
        print("  " + "  ".join(f"{r.get(c, '')}".ljust(widths[c]) for c in cols))


def save(name, data):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"{name}.json"), "w") as f:
        json.dump(data, f, indent=2, default=str)


def main():
    db = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)[MONGO_DB]
    if db.transactions.estimated_document_count() == 0:
        sys.exit("No transactions loaded. Run src/mongodb/load_mongo.py first.")

    print("\n=== 1. Net revenue & estimated profit by category and month ===")
    rev = list(db.transactions.aggregate(revenue_by_category_month(), allowDiskUse=True))
    save("revenue_by_category_month", rev)
    totals = {}
    for r in rev:
        t = totals.setdefault(r["category_name"], {"category": r["category_name"], "net_revenue": 0, "est_profit": 0})
        t["net_revenue"] += r["net_revenue"]
        t["est_profit"] += r["est_profit"]
    top = sorted(totals.values(), key=lambda x: -x["net_revenue"])
    for t in top:
        t["net_revenue"], t["est_profit"] = round(t["net_revenue"], 2), round(t["est_profit"], 2)
    print(f"  {len(rev)} (category, month) rows. Top categories over the whole period:")
    show(top, ["category", "net_revenue", "est_profit"], 10)

    print("\n=== 2. Customer segmentation (RFM-style, from embedded purchase_summary) ===")
    ref = db.transactions.find_one(sort=[("timestamp", -1)])["timestamp"]
    seg = list(db.users.aggregate(user_segments(ref)))[0]
    save("user_segments", seg)
    for s in seg["segments"]:
        s["segment"] = s.pop("_id")
        for k in ("revenue", "avg_order_value", "avg_orders"):
            s[k] = round(s[k] or 0, 2)
    print(f"  reference date: {ref.date()}")
    show(seg["segments"], ["segment", "users", "revenue", "avg_order_value", "avg_orders"])
    print("  Order-frequency buckets:")
    show([{"orders_from": b["_id"], "users": b["users"], "avg_spent": round(b["avg_spent"], 2)}
          for b in seg["by_frequency"]], ["orders_from", "users", "avg_spent"])

    print("\n=== 3. Top-selling products (with remaining stock) ===")
    tp = list(db.transactions.aggregate(top_products(), allowDiskUse=True))
    save("top_products", tp)
    show(tp, ["product_id", "name", "category", "units_sold", "revenue", "current_stock"])
    print(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
