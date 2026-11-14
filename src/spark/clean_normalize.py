"""
clean_normalize.py - Spark batch job 1: clean & normalise the raw JSON and report data quality.

Steps (implemented in spark_common.py so every job shares them):
  * read each entity with an explicit schema (no inference pass, cart_contents as a MAP)
  * drop duplicate primary keys and records missing their key / user reference
  * parse ISO strings to timestamps, upper/lower-case codes, fill missing geo / referrer values
  * flatten nested structs (geo_data, device_profile) into columns
  * explode nested arrays into analysis tables: transaction_items, page_views, subcategories
  * derive fields: date / month / hour / weekday, net line revenue after discount,
    reached_checkout, is_session_linked, data-quality flag total_ok
  * enrich products with category + subcategory (broadcast joins on tiny dimension tables)

Prints a data-quality report and saves it to output/spark/data_quality.csv.
Use --parquet <dir> to persist the cleaned tables (needs Hadoop winutils on Windows; works as-is on Linux).

Usage:  python src/spark/clean_normalize.py [--parquet data/clean]
"""

import argparse
import time

from pyspark.sql import functions as F

from spark_common import get_spark, load_clean, read_raw, save, SESSIONS_SCHEMA, TRANSACTIONS_SCHEMA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=None, help="directory to write cleaned tables as Parquet")
    args = ap.parse_args()

    t0 = time.time()
    spark = get_spark("clean-normalize")
    dfs = load_clean(spark)

    raw_sessions = read_raw(spark, "sessions", SESSIONS_SCHEMA)
    raw_txn = read_raw(spark, "transactions", TRANSACTIONS_SCHEMA)
    print(f"\nInput partitions: sessions={raw_sessions.rdd.getNumPartitions()} "
          f"transactions={raw_txn.rdd.getNumPartitions()}")

    s, t, u, p = dfs["sessions"], dfs["transactions"], dfs["users"], dfs["products"]
    checks = [
        ("users", "rows", u.count()),
        ("products", "rows", p.count()),
        ("products", "missing subcategory after join", p.filter(F.col("subcategory_name").isNull()).count()),
        ("products", "inactive", p.filter(~F.col("is_active")).count()),
        ("sessions", "raw rows", raw_sessions.count()),
        ("sessions", "rows after cleaning", s.count()),
        ("sessions", "unparseable start_time", s.filter(F.col("start_time").isNull()).count()),
        ("sessions", "duration != end-start",
         s.filter(F.col("duration_seconds") != F.unix_timestamp("end_time") - F.unix_timestamp("start_time")).count()),
        ("sessions", "reached checkout with EMPTY cart",
         s.filter(F.col("reached_checkout") & (F.col("cart_items") == 0)).count()),
        ("page_views", "rows (exploded)", dfs["page_views"].count()),
        ("transactions", "raw rows", raw_txn.count()),
        ("transactions", "rows after cleaning", t.count()),
        ("transactions", "not linked to a session", t.filter(~F.col("is_session_linked")).count()),
        ("transactions", "total != subtotal - discount", t.filter(~F.col("total_ok")).count()),
        ("transaction_items", "rows (exploded)", dfs["transaction_items"].count()),
        ("transaction_items", "unknown product_id",
         dfs["transaction_items"].join(p.select("product_id"), "product_id", "left_anti").count()),
        ("transactions", "unknown user_id", t.join(u.select("user_id"), "user_id", "left_anti").count()),
    ]
    report = spark.createDataFrame(checks, "entity string, check string, value long")
    print("\n=== Data-quality report ===")
    report.show(50, truncate=False)
    save(report, "data_quality")

    print("=== Cleaned session schema ===")
    s.printSchema()
    s.select("session_id", "user_id", "start_time", "device_type", "referrer", "n_pages",
             "cart_items", "reached_checkout", "conversion_status").show(5, truncate=False)

    if args.parquet:
        for name, df in dfs.items():
            part = ["date"] if name in ("sessions", "transactions") else []
            df.write.mode("overwrite").partitionBy(*part).parquet(f"{args.parquet}/{name}")
        print(f"Cleaned tables written to {args.parquet}")

    print(f"Done in {time.time() - t0:.0f}s")
    spark.stop()


if __name__ == "__main__":
    main()
