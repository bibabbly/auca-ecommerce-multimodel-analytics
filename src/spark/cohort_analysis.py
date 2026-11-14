"""
cohort_analysis.py - Spark batch job 3: cohort analysis of user purchasing patterns.

A. Registration cohorts (as in the brief): users grouped by registration month; for each calendar
   month of activity we compute active buyers, % of the cohort buying, revenue and revenue per
   cohort member. Registration happens 90-270 days before the end of the data window while all
   orders fall in the last 90 days, so "months since registration" starts at 3 for every cohort.
   We report both the calendar-month and months-since-registration views.
B. First-purchase (acquisition) cohorts: users grouped by the month of their first order and
   tracked over months 0, 1, 2, 3 after it -> classic retention triangle (% still buying).

Window functions (min over user partition) find each user's first order; everything else is
groupBy aggregations that Spark distributes by cohort key.

Usage:  python src/spark/cohort_analysis.py
"""

import time

from pyspark.sql import Window
from pyspark.sql import functions as F

from spark_common import get_spark, load_clean, save, show


def months_between(later, earlier):
    return ((F.year(later) - F.year(earlier)) * 12 + F.month(later) - F.month(earlier)).cast("int")


def main():
    t0 = time.time()
    spark = get_spark("cohort-analysis")
    dfs = load_clean(spark)
    users = dfs["users"].select("user_id", "registration_date", "registration_month")
    orders = (dfs["transactions"].filter(F.col("status") != "cancelled")
              .select("transaction_id", "user_id", "timestamp", "month", "total"))

    # ---- A. registration-month cohorts
    cohort_size = users.groupBy("registration_month").agg(F.count("*").alias("cohort_users"))
    reg = (orders.join(users, "user_id")
           .withColumn("months_since_reg", months_between("timestamp", "registration_date"))
           .groupBy("registration_month", F.col("month").alias("order_month"), "months_since_reg")
           .agg(F.countDistinct("user_id").alias("active_buyers"), F.count("*").alias("orders"),
                F.round(F.sum("total"), 2).alias("revenue"))
           .join(cohort_size, "registration_month")
           .withColumn("pct_buying", F.round(100 * F.col("active_buyers") / F.col("cohort_users"), 1))
           .withColumn("revenue_per_member", F.round(F.col("revenue") / F.col("cohort_users"), 2))
           .orderBy("registration_month", "order_month"))
    show(reg, 40, "A. Registration cohorts x order month")
    save(reg, "cohort_registration")

    pivot = (reg.groupBy("registration_month").pivot("order_month").agg(F.first("revenue_per_member"))
             .orderBy("registration_month"))
    show(pivot, 20, "A. Revenue per cohort member (rows: registration month, cols: order month)")
    save(pivot, "cohort_registration_pivot")

    lifetime = (orders.groupBy("user_id").agg(F.sum("total").alias("spent"), F.count("*").alias("n"))
                .join(users, "user_id", "right").fillna({"spent": 0.0, "n": 0})
                .groupBy("registration_month")
                .agg(F.count("*").alias("users"), F.round(F.avg("spent"), 2).alias("avg_spent_90d"),
                     F.round(F.avg("n"), 2).alias("avg_orders_90d"))
                .orderBy("registration_month"))
    show(lifetime, 20, "A. Average 90-day spend by registration cohort (does tenure matter?)")
    save(lifetime, "cohort_registration_summary")

    # ---- B. first-purchase cohorts (retention triangle)
    w = Window.partitionBy("user_id")
    o = (orders.withColumn("first_order", F.min("timestamp").over(w))
         .withColumn("cohort", F.date_format("first_order", "yyyy-MM"))
         .withColumn("month_n", months_between("timestamp", "first_order")))
    sizes = o.select("user_id", "cohort").distinct().groupBy("cohort").agg(F.count("*").alias("cohort_users"))
    retention = (o.groupBy("cohort", "month_n")
                 .agg(F.countDistinct("user_id").alias("buyers"), F.round(F.sum("total"), 2).alias("revenue"))
                 .join(sizes, "cohort")
                 .withColumn("retention_pct", F.round(100 * F.col("buyers") / F.col("cohort_users"), 1))
                 .orderBy("cohort", "month_n"))
    show(retention, 30, "B. First-purchase cohort retention")
    save(retention, "cohort_retention")
    show(retention.groupBy("cohort", "cohort_users").pivot("month_n").agg(F.first("retention_pct"))
         .orderBy("cohort"), 10, "B. Retention % by months since first purchase")

    print(f"Done in {time.time() - t0:.0f}s")
    spark.stop()


if __name__ == "__main__":
    main()
