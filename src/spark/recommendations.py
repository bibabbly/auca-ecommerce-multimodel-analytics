"""
recommendations.py - Spark batch job 2: product recommendation indicators (market-basket analysis).

A. "Customers who bought X also bought Y"  (transactions)
   For every pair of distinct products in the same order:
     support(X,Y)    = orders containing both
     confidence(X→Y) = support(X,Y) / orders containing X
     lift(X→Y)       = confidence(X→Y) / P(Y)        (>1 = bought together more than by chance)
B. "Products frequently viewed together"  (sessions.viewed_products, the clickstream in HBase)
   Same measures over sessions instead of orders. Sessions are ~4x more numerous than orders, so
   view-based pairs reach support thresholds that purchase pairs cannot (cold-start helper).
C. Category affinity matrix: lift between categories bought in the same order. With only
   25 categories every pair has enough support to be statistically meaningful.

Distributed design:
  * pairs are generated per order/session with a self-join on the basket id, keeping X < Y
    so each unordered pair is produced once (halves the shuffle), then mirrored for X→Y and Y→X
  * single-product counts are broadcast-joined (small) to compute confidence and lift
  * baskets are de-duplicated first (a product appearing twice in one order counts once)

Usage:  python src/spark/recommendations.py [--min-support 3]
"""

import argparse
import time

from pyspark.sql import Window
from pyspark.sql import functions as F

from spark_common import get_spark, load_clean, save, show


def pair_stats(baskets, basket_col, item_col, min_support):
    """baskets: one row per (basket, item), distinct. Returns X→Y rules with support/confidence/lift."""
    n_baskets = baskets.select(basket_col).distinct().count()
    item_counts = baskets.groupBy(item_col).agg(F.countDistinct(basket_col).alias("item_count"))

    a, b = baskets.alias("a"), baskets.alias("b")
    pairs = (a.join(b, (F.col(f"a.{basket_col}") == F.col(f"b.{basket_col}")) &
                    (F.col(f"a.{item_col}") < F.col(f"b.{item_col}")))
             .groupBy(F.col(f"a.{item_col}").alias("x"), F.col(f"b.{item_col}").alias("y"))
             .agg(F.count("*").alias("support"))
             .filter(F.col("support") >= min_support))
    # both directions: X→Y and Y→X have the same support but different confidence
    rules = pairs.unionByName(pairs.select(F.col("y").alias("x"), F.col("x").alias("y"), "support"))
    ic = F.broadcast(item_counts)
    return (rules
            .join(ic.select(F.col(item_col).alias("x"), F.col("item_count").alias("x_count")), "x")
            .join(ic.select(F.col(item_col).alias("y"), F.col("item_count").alias("y_count")), "y")
            .withColumn("confidence", F.round(F.col("support") / F.col("x_count"), 4))
            .withColumn("lift", F.round(F.col("confidence") / (F.col("y_count") / F.lit(n_baskets)), 2))
            .select("x", "y", "support", "x_count", "y_count", "confidence", "lift")), n_baskets


def top_n_per_item(rules, n=3):
    w = Window.partitionBy("x").orderBy(F.desc("lift"), F.desc("support"))
    return rules.withColumn("rank", F.row_number().over(w)).filter(F.col("rank") <= n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-support", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()
    spark = get_spark("recommendations")
    dfs = load_clean(spark)
    names = dfs["products"].select(F.col("product_id"), F.col("name"), F.col("category_name"))

    # ---- A. bought together
    bought = dfs["transaction_items"].select("transaction_id", "product_id").distinct()
    bought_rules, n_orders = pair_stats(bought, "transaction_id", "product_id", args.min_support)
    bought_rules = bought_rules.cache()
    multi = bought.groupBy("transaction_id").count().filter("count > 1").count()
    print(f"\nOrders: {n_orders:,} (with 2+ products: {multi:,}); "
          f"bought-together rules with support >= {args.min_support}: {bought_rules.count():,}")
    top_bought = (bought_rules.orderBy(F.desc("support"), F.desc("lift")).limit(20)
                  .join(names.withColumnRenamed("product_id", "x").withColumnRenamed("name", "x_name")
                        .drop("category_name"), "x")
                  .join(names.withColumnRenamed("product_id", "y").withColumnRenamed("name", "y_name")
                        .drop("category_name"), "y")
                  .orderBy(F.desc("support"), F.desc("lift")))
    show(top_bought.select("x", "x_name", "y", "y_name", "support", "confidence", "lift"), 10,
         "A. Customers who bought X also bought Y (top pairs by support)")
    save(top_bought, "reco_bought_together_top")
    save(top_n_per_item(bought_rules), "reco_bought_together_top3_per_product")

    # ---- B. viewed together (clickstream)
    viewed = (dfs["sessions"].select("session_id", F.explode("viewed_products").alias("product_id"))
              .distinct())
    viewed_rules, n_sessions = pair_stats(viewed, "session_id", "product_id", args.min_support)
    viewed_rules = viewed_rules.cache()
    print(f"\nSessions: {n_sessions:,}; viewed-together rules with support >= {args.min_support}: "
          f"{viewed_rules.count():,}")
    show(viewed_rules.orderBy(F.desc("support"), F.desc("lift")), 10,
         "B. Products frequently viewed together (top pairs by support)")
    save(viewed_rules.orderBy(F.desc("support")).limit(500), "reco_viewed_together_top")

    # ---- C. category affinity
    cat_baskets = (dfs["transaction_items"].join(F.broadcast(dfs["products"].select("product_id", "category_name")),
                                                 "product_id")
                   .select("transaction_id", F.col("category_name")).distinct())
    cat_rules, _ = pair_stats(cat_baskets, "transaction_id", "category_name", 1)
    show(cat_rules.orderBy(F.desc("lift")), 10, "C. Category affinity (highest lift)")
    save(cat_rules, "reco_category_affinity")

    # ---- summary: is there real affinity in the data?
    stats = bought_rules.agg(F.avg("lift").alias("avg_lift"), F.expr("percentile_approx(lift, 0.5)").alias("median_lift"),
                             F.max("lift").alias("max_lift")).first()
    cstats = cat_rules.agg(F.min("lift").alias("min"), F.max("lift").alias("max"), F.avg("lift").alias("avg")).first()
    print(f"\nProduct-pair lift: avg={stats['avg_lift']:.2f} median={stats['median_lift']:.2f} max={stats['max_lift']:.2f}")
    print(f"Category-pair lift range: {cstats['min']:.2f} .. {cstats['max']:.2f} (avg {cstats['avg']:.2f}); "
          "values near 1.0 mean categories are combined about as often as chance predicts.")
    print(f"Done in {time.time() - t0:.0f}s")
    spark.stop()


if __name__ == "__main__":
    main()
