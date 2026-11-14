"""
funnel_clv.py - Part 3: integrated analytics across HBase + MongoDB, processed in Spark.

Business questions
  Q1 Funnel: "Where do we lose customers between viewing a product and paying, and does it differ
     by device and traffic source?"
  Q2 CLV:    "What is each customer worth over the next 12 months, and how does on-site engagement
     relate to that value?"

Data sources (read from the live databases, not the JSON files)
  HBase   user_sessions  -> session metadata (s), page-view sequence (pv), cart lines (c)
  MongoDB transactions   -> orders, linked to sessions by session_id, with embedded line items
  MongoDB products       -> subcategory profit margin (to turn revenue into gross profit)
  MongoDB users          -> registration date, country

Workflow
  1. Extract: happybase scans user_sessions (only needed families); pymongo reads projections of
     transactions / products / users. Each becomes a Spark DataFrame.
     (At production scale the same step would use the MongoDB Spark Connector and an HBase
      Spark connector / TableInputFormat so each executor reads its own region / shard in parallel.)
  2. Transform in Spark: derive funnel stages per session, join sessions to orders on session_id
     (cross-store join), aggregate per device / referrer; compute RFM + engagement per user and a
     margin-based CLV estimate.
  3. Load: result tables to output/integration/*.csv for the visualisation step.

CLV model (simple, explainable, 90 days of data):
  monthly_orders   = orders_90d / 3
  avg_gross_profit = AOV * user's revenue-weighted profit margin
  churn_monthly    = share of buyers with no order in the last 30 days (cohort-level estimate)
  expected_months  = min(1 / churn_monthly, 12)            (capped 12-month horizon)
  CLV_12m          = monthly_orders * avg_gross_profit * expected_months

Usage:  python src/integration/funnel_clv.py [--hbase-limit 100000]
"""

import argparse
import json
import os
import sys
import time

import happybase
from pymongo import MongoClient
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.types import (ArrayType, BooleanType, DoubleType, IntegerType, StringType, StructField,
                               StructType, TimestampType)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "spark"))
from common import HBASE_HOST, HBASE_PORT, MONGO_DB, MONGO_URI, PROJECT_ROOT  # noqa: E402
from spark_common import get_spark  # noqa: E402

OUT_DIR = os.path.join(PROJECT_ROOT, "output", "integration")

SESSION_SCHEMA = StructType([
    StructField("session_id", StringType()), StructField("user_id", StringType()),
    StructField("start_time", StringType()), StructField("duration_s", IntegerType()),
    StructField("device", StringType()), StructField("referrer", StringType()),
    StructField("n_pages", IntegerType()), StructField("viewed_product", BooleanType()),
    StructField("cart_lines", IntegerType()), StructField("reached_checkout", BooleanType()),
    StructField("page_types", ArrayType(StringType()))])


# --------------------------------------------------------------------------- extract
def extract_hbase_sessions(limit=None):
    """Scan user_sessions. One Python tuple per session, page views reduced to the page-type list."""
    conn = happybase.Connection(HBASE_HOST, HBASE_PORT, timeout=120_000)
    tbl = conn.table("user_sessions")
    rows, t0 = [], time.time()
    for _, r in tbl.scan(batch_size=2000, limit=limit):
        s = {k.decode(): v.decode() for k, v in r.items()}
        pages = [json.loads(v)["type"] for k, v in sorted(s.items()) if k.startswith("pv:")]
        rows.append((s["s:session_id"], s["s:user_id"], s["s:start_time"], int(s["s:duration_s"]),
                     s["s:device"], s["s:referrer"], int(s["s:n_pages"]),
                     "product_detail" in pages,
                     sum(1 for k in s if k.startswith("c:")),
                     any(p in ("checkout", "confirmation") for p in pages),
                     pages))
        if len(rows) % 100_000 == 0:
            print(f"  HBase: {len(rows):,} sessions scanned ({time.time() - t0:.0f}s)")
    conn.close()
    print(f"  HBase: {len(rows):,} sessions in {time.time() - t0:.0f}s")
    return rows


def extract_mongo():
    db = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)[MONGO_DB]
    t0 = time.time()
    txns = [(d["_id"], d.get("session_id"), d["user_id"], d["timestamp"], float(d["total"]),
             [(i["product_id"], float(i["subtotal"])) for i in d["items"]])
            for d in db.transactions.find({}, {"session_id": 1, "user_id": 1, "timestamp": 1,
                                               "total": 1, "items.product_id": 1, "items.subtotal": 1})]
    margins = [(d["_id"], float(d["subcategory"].get("profit_margin") or 0))
               for d in db.products.find({}, {"subcategory.profit_margin": 1})]
    users = [(d["_id"], d["geo"].get("country"), d["registration_date"])
             for d in db.users.find({}, {"geo.country": 1, "registration_date": 1})]
    print(f"  MongoDB: {len(txns):,} transactions, {len(margins):,} products, {len(users):,} users "
          f"in {time.time() - t0:.0f}s")
    return txns, margins, users


def save(df, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    pdf = df.toPandas()
    pdf.to_csv(os.path.join(OUT_DIR, f"{name}.csv"), index=False)
    return pdf


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hbase-limit", type=int, default=None, help="scan only the first N sessions")
    args = ap.parse_args()
    t0 = time.time()

    print("1. Extract")
    session_rows = extract_hbase_sessions(args.hbase_limit)
    txn_rows, margin_rows, user_rows = extract_mongo()

    spark = get_spark("integration-funnel-clv")
    sessions = (spark.createDataFrame(session_rows, SESSION_SCHEMA)
                .withColumn("start_time", F.to_timestamp("start_time")).repartition(16).cache())
    txns = spark.createDataFrame(txn_rows, StructType([
        StructField("transaction_id", StringType()), StructField("session_id", StringType()),
        StructField("user_id", StringType()), StructField("timestamp", TimestampType()),
        StructField("total", DoubleType()),
        StructField("items", ArrayType(StructType([StructField("product_id", StringType()),
                                                   StructField("subtotal", DoubleType())])))])).cache()
    margins = spark.createDataFrame(margin_rows, "product_id string, margin double")
    users = spark.createDataFrame(user_rows, "user_id string, country string, registration_date timestamp")

    # ------------------------------------------------------------------ Q1 funnel
    print("\n2. Funnel conversion (HBase sessions JOIN MongoDB orders on session_id)")
    linked_orders = txns.filter(F.col("session_id").isNotNull()).select("session_id", F.lit(True).alias("purchased"))
    funnel_base = (sessions.join(linked_orders, "session_id", "left")      # cross-store join
                   .fillna({"purchased": False})
                   .withColumn("s1_session", F.lit(1))
                   .withColumn("s2_viewed_product", F.col("viewed_product").cast("int"))
                   .withColumn("s3_added_to_cart", (F.col("viewed_product") & (F.col("cart_lines") > 0)).cast("int"))
                   .withColumn("s4_reached_checkout",
                               (F.col("viewed_product") & (F.col("cart_lines") > 0) & F.col("reached_checkout")).cast("int"))
                   .withColumn("s5_purchased", F.col("purchased").cast("int")))
    stages = ["s1_session", "s2_viewed_product", "s3_added_to_cart", "s4_reached_checkout", "s5_purchased"]

    overall = funnel_base.agg(*[F.sum(s).alias(s) for s in stages]).first().asDict()
    print(f"  {'stage':<22}{'sessions':>10}{'% of start':>12}{'% of prev':>11}")
    prev = None
    for s in stages:
        v = overall[s]
        print(f"  {s:<22}{v:>10,}{100 * v / overall['s1_session']:>11.1f}%"
              f"{(100 * v / prev if prev else 100):>10.1f}%")
        prev = v
    save(spark.createDataFrame([(s, overall[s]) for s in stages], "stage string, sessions long"), "funnel_overall")

    def funnel_by(col):
        df = (funnel_base.groupBy(col).agg(*[F.sum(s).alias(s) for s in stages])
              .withColumn("view_rate", F.round(100 * F.col("s2_viewed_product") / F.col("s1_session"), 1))
              .withColumn("cart_rate", F.round(100 * F.col("s3_added_to_cart") / F.col("s2_viewed_product"), 1))
              .withColumn("checkout_rate", F.round(100 * F.col("s4_reached_checkout") / F.col("s3_added_to_cart"), 1))
              .withColumn("purchase_rate", F.round(100 * F.col("s5_purchased") / F.col("s4_reached_checkout"), 1))
              .withColumn("overall_conversion", F.round(100 * F.col("s5_purchased") / F.col("s1_session"), 2))
              .orderBy(F.desc("overall_conversion")))
        df.show(truncate=False)
        return save(df, f"funnel_by_{col}")
    funnel_by("device")
    funnel_by("referrer")

    # orders that came from sessions not present in HBase would show up here (data-consistency check)
    orphan = linked_orders.join(sessions.select("session_id"), "session_id", "left_anti").count()
    print(f"  session-linked orders whose session is not in HBase: {orphan:,}"
          + (" (expected when --hbase-limit is used)" if args.hbase_limit else ""))

    # ------------------------------------------------------------------ Q2 CLV
    print("\n3. Customer Lifetime Value (MongoDB orders + product margins + HBase engagement)")
    ref = txns.agg(F.max("timestamp")).first()[0]
    item_profit = (txns.select("transaction_id", "user_id", F.explode("items").alias("i"))
                   .select("user_id", F.col("i.product_id").alias("product_id"), F.col("i.subtotal").alias("subtotal"))
                   .join(F.broadcast(margins), "product_id", "left")
                   .groupBy("user_id")
                   .agg((F.sum(F.col("subtotal") * F.coalesce("margin", F.lit(0.0))) / F.sum("subtotal")).alias("margin")))
    rfm = (txns.groupBy("user_id")
           .agg(F.count("*").alias("orders"), F.sum("total").alias("revenue"),
                F.max("timestamp").alias("last_order"))
           .withColumn("aov", F.col("revenue") / F.col("orders"))
           .withColumn("recency_days", F.datediff(F.lit(ref), "last_order")))
    raw_churn = rfm.agg(F.avg((F.col("recency_days") > 30).cast("double"))).first()[0]
    churn = max(raw_churn, 1 / 12)  # floor so expected lifetime never exceeds the 12-month horizon
    expected_months = min(1 / churn, 12.0)
    print(f"  reference date {ref.date()}, observed 30-day inactivity {raw_churn:.1%} "
          f"-> monthly churn used {churn:.1%}, "
          f"expected lifetime {expected_months:.1f} months (capped at 12)")

    engagement = (sessions.groupBy("user_id")
                  .agg(F.count("*").alias("sessions"), F.avg("duration_s").alias("avg_session_s"),
                       F.sum("n_pages").alias("page_views"),
                       F.sum((F.col("cart_lines") > 0).cast("int")).alias("cart_sessions")))
    clv = (users.join(rfm, "user_id", "left").join(item_profit, "user_id", "left")
           .join(engagement, "user_id", "left")
           .fillna({"orders": 0, "revenue": 0.0, "aov": 0.0, "margin": 0.0, "sessions": 0, "page_views": 0})
           .withColumn("monthly_orders", F.col("orders") / 3.0)
           .withColumn("clv_12m", F.round(F.col("monthly_orders") * F.col("aov") * F.col("margin")
                                          * F.lit(expected_months), 2))
           .withColumn("session_to_order_pct",
                       F.round(100 * F.col("orders") / F.when(F.col("sessions") > 0, F.col("sessions")), 1))
           .withColumn("clv_tier", F.ntile(4).over(Window.orderBy("clv_12m"))))
    clv = clv.withColumn("clv_tier", F.concat(F.lit("Q"), F.col("clv_tier"))).cache()

    tiers = (clv.groupBy("clv_tier")
             .agg(F.count("*").alias("users"), F.round(F.avg("clv_12m"), 2).alias("avg_clv_12m"),
                  F.round(F.sum("clv_12m"), 0).alias("total_clv_12m"),
                  F.round(F.avg("orders"), 2).alias("avg_orders_90d"), F.round(F.avg("aov"), 2).alias("avg_aov"),
                  F.round(F.avg("margin") * 100, 1).alias("avg_margin_pct"),
                  F.round(F.avg("sessions"), 1).alias("avg_sessions"),
                  F.round(F.avg("avg_session_s") / 60, 1).alias("avg_session_min"),
                  F.round(F.avg("session_to_order_pct"), 1).alias("session_to_order_pct"))
             .orderBy("clv_tier"))
    tiers.show(truncate=False)
    save(tiers, "clv_tiers")
    corr = clv.select(F.corr("sessions", "clv_12m").alias("sessions"), F.corr("page_views", "clv_12m").alias("page_views"),
                      F.corr("avg_session_s", "clv_12m").alias("avg_session_s"),
                      F.corr("orders", "clv_12m").alias("orders"), F.corr("aov", "clv_12m").alias("aov")).first()
    print("  Pearson correlation with CLV: " + ", ".join(f"{k}={v:.2f}" for k, v in corr.asDict().items()))
    top_share = clv.filter(F.col("clv_tier") == "Q4").agg(F.sum("clv_12m")).first()[0] / clv.agg(F.sum("clv_12m")).first()[0]
    print(f"  top quartile holds {top_share:.1%} of total predicted 12-month value")
    save(clv.select("user_id", "country", "orders", "revenue", "aov", "margin", "recency_days", "sessions",
                    "page_views", "avg_session_s", "session_to_order_pct", "clv_12m", "clv_tier"), "clv_users")
    save(clv.groupBy("country").agg(F.count("*").alias("users"), F.round(F.sum("clv_12m"), 0).alias("total_clv"),
                                    F.round(F.avg("clv_12m"), 2).alias("avg_clv"))
         .orderBy(F.desc("total_clv")).limit(25), "clv_by_country")

    print(f"\nResults in {OUT_DIR}. Done in {time.time() - t0:.0f}s")
    spark.stop()


if __name__ == "__main__":
    main()
