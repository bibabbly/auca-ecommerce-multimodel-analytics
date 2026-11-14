"""
spark_sql_analytics.py - Spark SQL over the cleaned DataFrames (Part 2.2).

The cleaned DataFrames are registered as temporary views. Conceptually:
    users, products, categories, transactions, transaction_items  -> "live" in MongoDB
    sessions, page_views                                          -> "live" in HBase
so the joins below are exactly the cross-store queries Spark would run after pulling data from
both systems (see src/integration for the version that reads the real databases).

Queries:
  Q1  Top-3 products per category by net revenue          (CTE + RANK() window)
  Q2  Conversion rate by device and referrer             (conditional aggregation)
  Q3  Engagement vs. spend: session behaviour (HBase) joined to purchases (MongoDB) per user,
      bucketed into engagement quartiles                  (multi-table join + NTILE)
  Q4  Traffic and conversion by weekday x hour            (time-series rollup)
  Q5  Price-change effect: products whose price changed vs. not
  Q6  Month-over-month revenue growth per category         (LAG window)

Usage:  python src/spark/spark_sql_analytics.py
"""

import time

from spark_common import get_spark, load_clean, register_views, save

QUERIES = {
    "q1_top_products_per_category": """
        WITH product_revenue AS (
            SELECT p.category_name, ti.product_id, p.name,
                   SUM(ti.quantity)  AS units,
                   ROUND(SUM(ti.line_net), 2) AS net_revenue,
                   ROUND(SUM(ti.line_net * p.profit_margin), 2) AS est_profit
            FROM transaction_items ti
            JOIN products p ON p.product_id = ti.product_id
            GROUP BY p.category_name, ti.product_id, p.name
        ), ranked AS (
            SELECT *, RANK() OVER (PARTITION BY category_name ORDER BY net_revenue DESC) AS rnk
            FROM product_revenue
        )
        SELECT category_name, rnk, product_id, name, units, net_revenue, est_profit
        FROM ranked WHERE rnk <= 3
        ORDER BY category_name, rnk
    """,
    "q2_conversion_by_device_referrer": """
        SELECT device_type, referrer,
               COUNT(*)                                                     AS sessions,
               SUM(CASE WHEN cart_items > 0 THEN 1 ELSE 0 END)              AS with_cart,
               SUM(CASE WHEN conversion_status = 'converted' THEN 1 ELSE 0 END) AS converted,
               ROUND(100 * AVG(CASE WHEN cart_items > 0 THEN 1 ELSE 0 END), 2)  AS cart_rate_pct,
               ROUND(100 * AVG(CASE WHEN conversion_status = 'converted' THEN 1 ELSE 0 END), 2) AS conversion_pct,
               ROUND(AVG(duration_seconds) / 60, 1)                         AS avg_minutes
        FROM sessions
        GROUP BY device_type, referrer
        ORDER BY conversion_pct DESC
    """,
    "q3_engagement_vs_spend": """
        WITH engagement AS (          -- HBase side: clickstream per user
            SELECT user_id, COUNT(*) AS sessions, SUM(n_pages) AS page_views,
                   ROUND(SUM(duration_seconds) / 3600.0, 2) AS hours_on_site,
                   SUM(n_products_viewed) AS products_viewed
            FROM sessions GROUP BY user_id
        ), purchases AS (             -- MongoDB side: orders per user
            SELECT user_id, COUNT(*) AS orders, SUM(total) AS spent
            FROM transactions GROUP BY user_id
        ), joined AS (
            SELECT u.user_id, u.country, e.sessions, e.page_views, e.hours_on_site,
                   COALESCE(p.orders, 0) AS orders, COALESCE(p.spent, 0) AS spent,
                   NTILE(4) OVER (ORDER BY e.page_views) AS engagement_quartile
            FROM users u
            JOIN engagement e ON e.user_id = u.user_id
            LEFT JOIN purchases p ON p.user_id = u.user_id
        )
        SELECT engagement_quartile, COUNT(*) AS users,
               ROUND(AVG(sessions), 1) AS avg_sessions, ROUND(AVG(page_views), 1) AS avg_page_views,
               ROUND(AVG(hours_on_site), 2) AS avg_hours, ROUND(AVG(orders), 2) AS avg_orders,
               ROUND(AVG(spent), 2) AS avg_spent,
               ROUND(SUM(spent) / SUM(page_views), 2) AS revenue_per_page_view
        FROM joined GROUP BY engagement_quartile ORDER BY engagement_quartile
    """,
    "q4_weekday_hour_traffic": """
        SELECT weekday, hour, COUNT(*) AS sessions,
               ROUND(100 * AVG(CASE WHEN conversion_status = 'converted' THEN 1 ELSE 0 END), 2) AS conversion_pct
        FROM sessions
        GROUP BY weekday, hour
        ORDER BY sessions DESC
    """,
    "q5_price_change_effect": """
        SELECT CASE WHEN p.price_changes = 0 THEN 'no change'
                    WHEN p.base_price > p.first_price THEN 'price increased'
                    ELSE 'price decreased' END AS price_trend,
               COUNT(DISTINCT p.product_id) AS products,
               ROUND(SUM(ti.quantity) / COUNT(DISTINCT p.product_id), 2) AS units_per_product,
               ROUND(SUM(ti.line_net) / COUNT(DISTINCT p.product_id), 2) AS revenue_per_product
        FROM products p
        LEFT JOIN transaction_items ti ON ti.product_id = p.product_id
        GROUP BY 1 ORDER BY 1
    """,
    "q6_category_mom_growth": """
        WITH monthly AS (
            SELECT p.category_name, date_format(ti.date, 'yyyy-MM') AS month,
                   ROUND(SUM(ti.line_net), 2) AS revenue
            FROM transaction_items ti JOIN products p ON p.product_id = ti.product_id
            GROUP BY p.category_name, date_format(ti.date, 'yyyy-MM')
        )
        SELECT category_name, month, revenue,
               LAG(revenue) OVER (PARTITION BY category_name ORDER BY month) AS prev_month,
               ROUND(100 * (revenue / LAG(revenue) OVER (PARTITION BY category_name ORDER BY month) - 1), 1)
                   AS mom_growth_pct
        FROM monthly
        ORDER BY category_name, month
    """,
}


def main():
    t0 = time.time()
    spark = get_spark("spark-sql-analytics")
    register_views(spark, load_clean(spark))
    for name, sql in QUERIES.items():
        print(f"\n=== {name} ===")
        df = spark.sql(sql)
        df.show(12, truncate=40)
        save(df, name)
    print("\nPhysical plan for Q3 (note BroadcastHashJoin / SortMergeJoin choices made by the optimizer):")
    spark.sql(QUERIES["q3_engagement_vs_spend"]).explain(mode="simple")
    print(f"Done in {time.time() - t0:.0f}s")
    spark.stop()


if __name__ == "__main__":
    main()
