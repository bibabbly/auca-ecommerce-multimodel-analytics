"""
spark_common.py - SparkSession factory, explicit schemas and the shared cleaning/normalisation layer.

Every Spark job imports `load_clean(spark)` so all analytics run on the same cleaned,
typed DataFrames:

    users, categories (dim), subcategories (dim), products, transactions, transaction_items,
    sessions, page_views

Why explicit schemas (instead of schema inference)?
  * Inference reads the whole input once more just to guess types (expensive on 600 MB+).
  * `cart_contents` uses product IDs as JSON keys. Inference would turn that into a struct with
    one field per product ever seen (thousands of columns). Declaring it as
    MAP<STRING, STRUCT<quantity, price>> keeps it a compact key/value column.
  * Types (timestamps, doubles) are guaranteed, so bad records surface as nulls we can count.
"""

import os
import sys

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (ArrayType, BooleanType, DoubleType, IntegerType, LongType, MapType,
                               StringType, StructField, StructType)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(PROJECT_ROOT, "data", "raw"))
OUT_DIR = os.path.join(PROJECT_ROOT, "output", "spark")


# --------------------------------------------------------------------------- session
def _no_spaces(path):
    """Windows: Spark's .cmd launch scripts break on paths with spaces; use the 8.3 short form."""
    if os.name != "nt" or " " not in path:
        return path
    import ctypes
    buf = ctypes.create_unicode_buffer(1024)
    ctypes.windll.kernel32.GetShortPathNameW(path, buf, 1024)
    return buf.value or path


def _use_bundled_spark():
    """Always run the Spark that ships with the installed pyspark package. A machine-wide
    SPARK_HOME (e.g. an older C:\\spark 3.x) would otherwise be picked up and its jars would not
    match the Python API. Set USE_SYSTEM_SPARK=1 to opt out (e.g. on a real cluster)."""
    if os.environ.get("USE_SYSTEM_SPARK") == "1":
        return
    import pyspark
    os.environ["SPARK_HOME"] = _no_spaces(os.path.dirname(pyspark.__file__))
    if os.name == "nt":
        for var in ("TMP", "TEMP"):  # temp dir with spaces also breaks the launch scripts
            if " " in os.environ.get(var, ""):
                os.environ[var] = _no_spaces(os.environ[var])


def get_spark(app_name="ecommerce-analytics"):
    _use_bundled_spark()
    # make the Python workers use the same interpreter (venv) as the driver
    os.environ.setdefault("PYSPARK_PYTHON", _no_spaces(sys.executable))
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", _no_spaces(sys.executable))
    spark = (SparkSession.builder
             .appName(app_name)
             .master(os.environ.get("SPARK_MASTER", "local[*]"))       # all local cores
             .config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", "6g"))
             .config("spark.sql.shuffle.partitions", "16")              # default 200 is far too many for local
             .config("spark.sql.adaptive.enabled", "true")              # AQE: coalesce partitions, handle skew
             .config("spark.sql.session.timeZone", "UTC")
             .config("spark.ui.showConsoleProgress", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# --------------------------------------------------------------------------- schemas
GEO = StructType([StructField("city", StringType()), StructField("state", StringType()),
                  StructField("country", StringType())])

USERS_SCHEMA = StructType([
    StructField("user_id", StringType()), StructField("geo_data", GEO),
    StructField("registration_date", StringType()), StructField("last_active", StringType())])

CATEGORIES_SCHEMA = StructType([
    StructField("category_id", StringType()), StructField("name", StringType()),
    StructField("subcategories", ArrayType(StructType([
        StructField("subcategory_id", StringType()), StructField("name", StringType()),
        StructField("profit_margin", DoubleType())])))])

PRODUCTS_SCHEMA = StructType([
    StructField("product_id", StringType()), StructField("name", StringType()),
    StructField("category_id", StringType()), StructField("subcategory_id", StringType()),
    StructField("base_price", DoubleType()), StructField("current_stock", LongType()),
    StructField("is_active", BooleanType()),
    StructField("price_history", ArrayType(StructType([
        StructField("price", DoubleType()), StructField("date", StringType())]))),
    StructField("creation_date", StringType())])

ITEM = StructType([StructField("product_id", StringType()), StructField("quantity", IntegerType()),
                   StructField("unit_price", DoubleType()), StructField("subtotal", DoubleType())])
TRANSACTIONS_SCHEMA = StructType([
    StructField("transaction_id", StringType()), StructField("session_id", StringType()),
    StructField("user_id", StringType()), StructField("timestamp", StringType()),
    StructField("items", ArrayType(ITEM)), StructField("subtotal", DoubleType()),
    StructField("discount", DoubleType()), StructField("total", DoubleType()),
    StructField("payment_method", StringType()), StructField("status", StringType())])

SESSIONS_SCHEMA = StructType([
    StructField("session_id", StringType()), StructField("user_id", StringType()),
    StructField("start_time", StringType()), StructField("end_time", StringType()),
    StructField("duration_seconds", IntegerType()),
    StructField("geo_data", StructType(GEO.fields + [StructField("ip_address", StringType())])),
    StructField("device_profile", StructType([StructField("type", StringType()),
                                              StructField("os", StringType()),
                                              StructField("browser", StringType())])),
    StructField("viewed_products", ArrayType(StringType())),
    StructField("page_views", ArrayType(StructType([
        StructField("timestamp", StringType()), StructField("page_type", StringType()),
        StructField("product_id", StringType()), StructField("category_id", StringType()),
        StructField("view_duration", IntegerType())]))),
    StructField("cart_contents", MapType(StringType(), StructType([
        StructField("quantity", IntegerType()), StructField("price", DoubleType())]))),
    StructField("conversion_status", StringType()), StructField("referrer", StringType())])


def _path(entity):
    """sessions -> data/raw/sessions_*.jsonl ; others -> data/raw/<entity>.jsonl (or .json)."""
    if entity == "sessions":
        return os.path.join(DATA_DIR, "sessions_*.json*")
    for ext in ("jsonl", "json"):
        p = os.path.join(DATA_DIR, f"{entity}.{ext}")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{entity} not found in {DATA_DIR} (run src/generate_data.py)")


def read_raw(spark, entity, schema):
    path = _path(entity)
    # JSON Lines is split across tasks; a JSON array file needs multiLine (one task per file)
    multiline = path.endswith(".json")
    return spark.read.schema(schema).option("multiLine", multiline).option("mode", "PERMISSIVE").json(path)


# --------------------------------------------------------------------------- cleaning
def ts(col):
    return F.to_timestamp(F.col(col))


def clean_users(df):
    return (df.dropDuplicates(["user_id"])
            .filter(F.col("user_id").isNotNull())
            .select("user_id",
                    F.coalesce(F.initcap(F.trim("geo_data.city")), F.lit("Unknown")).alias("city"),
                    F.coalesce(F.upper(F.trim("geo_data.state")), F.lit("NA")).alias("state"),
                    F.coalesce(F.upper(F.trim("geo_data.country")), F.lit("UNKNOWN")).alias("country"),
                    ts("registration_date").alias("registration_date"),
                    ts("last_active").alias("last_active"))
            # last_active can never precede registration
            .withColumn("last_active", F.greatest("last_active", "registration_date"))
            .withColumn("registration_month", F.date_format("registration_date", "yyyy-MM")))


def clean_categories(df):
    categories = df.select("category_id", F.col("name").alias("category_name")).dropDuplicates(["category_id"])
    subcategories = (df.select("category_id", F.explode_outer("subcategories").alias("s"))
                     .select("category_id", F.col("s.subcategory_id").alias("subcategory_id"),
                             F.col("s.name").alias("subcategory_name"),
                             F.col("s.profit_margin").alias("profit_margin"))
                     .dropDuplicates(["subcategory_id"]))
    return categories, subcategories


def clean_products(df, categories, subcategories):
    return (df.dropDuplicates(["product_id"])
            .filter(F.col("product_id").isNotNull())
            .withColumn("current_stock", F.greatest(F.coalesce("current_stock", F.lit(0)), F.lit(0)))
            .withColumn("is_active", F.coalesce("is_active", F.lit(False)))
            .withColumn("price_changes", F.greatest(F.size("price_history") - 1, F.lit(0)))
            .withColumn("first_price", F.col("price_history")[0]["price"])
            .withColumn("creation_date", ts("creation_date"))
            .drop("price_history")
            # broadcast joins: the dimension tables are tiny (25 / ~100 rows)
            .join(F.broadcast(categories), "category_id", "left")
            .join(F.broadcast(subcategories.drop("category_id")), "subcategory_id", "left")
            .withColumn("profit_margin", F.coalesce("profit_margin", F.lit(0.0))))


def clean_transactions(df):
    t = (df.dropDuplicates(["transaction_id"])
         .filter(F.col("transaction_id").isNotNull() & F.col("user_id").isNotNull())
         .withColumn("timestamp", ts("timestamp"))
         .withColumn("discount", F.coalesce("discount", F.lit(0.0)))
         .withColumn("payment_method", F.lower(F.trim("payment_method")))
         .withColumn("status", F.lower(F.trim("status")))
         .withColumn("is_session_linked", F.col("session_id").isNotNull())
         .withColumn("n_items", F.size("items"))
         # data-quality flag: stored total should equal subtotal - discount (to the cent)
         .withColumn("total_ok", F.abs(F.col("subtotal") - F.col("discount") - F.col("total")) < 0.011)
         .withColumn("date", F.to_date("timestamp"))
         .withColumn("month", F.date_format("timestamp", "yyyy-MM")))
    items = (t.select("transaction_id", "user_id", "timestamp", "date",
                      (F.col("total") / F.when(F.col("subtotal") > 0, F.col("subtotal")).otherwise(1)).alias("net_ratio"),
                      F.explode("items").alias("i"))
             .select("transaction_id", "user_id", "timestamp", "date",
                     F.col("i.product_id").alias("product_id"), F.col("i.quantity").alias("quantity"),
                     F.col("i.unit_price").alias("unit_price"), F.col("i.subtotal").alias("line_subtotal"),
                     # line revenue after allocating the order-level discount
                     F.round(F.col("i.subtotal") * F.col("net_ratio"), 2).alias("line_net"))
             .filter((F.col("quantity") > 0) & F.col("product_id").isNotNull()))
    return t.drop("items"), items


def clean_sessions(df):
    s = (df.dropDuplicates(["session_id"])
         .filter(F.col("session_id").isNotNull() & F.col("user_id").isNotNull())
         .withColumn("start_time", ts("start_time"))
         .withColumn("end_time", ts("end_time"))
         # recompute duration from timestamps when missing/inconsistent
         .withColumn("duration_seconds",
                     F.coalesce("duration_seconds",
                                (F.unix_timestamp("end_time") - F.unix_timestamp("start_time")).cast("int")))
         .withColumn("device_type", F.lower("device_profile.type"))
         .withColumn("os", F.col("device_profile.os"))
         .withColumn("browser", F.col("device_profile.browser"))
         .withColumn("country", F.upper("geo_data.country"))
         .withColumn("state", F.upper("geo_data.state"))
         .withColumn("referrer", F.coalesce(F.lower("referrer"), F.lit("unknown")))
         .withColumn("n_pages", F.size("page_views"))
         .withColumn("n_products_viewed", F.size("viewed_products"))
         .withColumn("cart_items", F.coalesce(F.size("cart_contents"), F.lit(0)))
         .withColumn("cart_items", F.when(F.col("cart_items") < 0, 0).otherwise(F.col("cart_items")))
         .withColumn("reached_checkout",
                     F.exists("page_views", lambda p: p["page_type"].isin("checkout", "confirmation")))
         .withColumn("date", F.to_date("start_time"))
         .withColumn("hour", F.hour("start_time"))
         .withColumn("weekday", F.date_format("start_time", "E")))
    page_views = (s.select("session_id", "user_id", F.posexplode("page_views").alias("seq", "pv"))
                  .select("session_id", "user_id", "seq", ts("pv.timestamp").alias("ts"),
                          F.col("pv.page_type").alias("page_type"), F.col("pv.product_id").alias("product_id"),
                          F.col("pv.category_id").alias("category_id"),
                          F.col("pv.view_duration").alias("view_duration")))
    return s.drop("geo_data", "device_profile", "page_views"), page_views


def load_clean(spark, cache=True):
    """Read raw JSON with explicit schemas and return a dict of cleaned DataFrames."""
    users = clean_users(read_raw(spark, "users", USERS_SCHEMA))
    categories, subcategories = clean_categories(read_raw(spark, "categories", CATEGORIES_SCHEMA))
    products = clean_products(read_raw(spark, "products", PRODUCTS_SCHEMA), categories, subcategories)
    transactions, items = clean_transactions(read_raw(spark, "transactions", TRANSACTIONS_SCHEMA))
    sessions, page_views = clean_sessions(read_raw(spark, "sessions", SESSIONS_SCHEMA))
    dfs = dict(users=users, categories=categories, subcategories=subcategories, products=products,
               transactions=transactions, transaction_items=items, sessions=sessions, page_views=page_views)
    if cache:  # reused by several queries in each job -> avoid re-parsing JSON every time
        for name in ("users", "products", "transactions", "transaction_items", "sessions"):
            dfs[name] = dfs[name].cache()
    return dfs


def register_views(spark, dfs):
    for name, df in dfs.items():
        df.createOrReplaceTempView(name)


# --------------------------------------------------------------------------- output
def save(df: DataFrame, name, max_rows=200_000):
    """Save a (small) result table to output/spark/<name>.csv via pandas.
    On Windows, Spark's own writers need Hadoop winutils.exe; results here are aggregates,
    so collecting to the driver is cheap and portable."""
    os.makedirs(OUT_DIR, exist_ok=True)
    pdf = df.limit(max_rows).toPandas()
    path = os.path.join(OUT_DIR, f"{name}.csv")
    pdf.to_csv(path, index=False)
    return pdf


def show(df, n=15, title=None):
    if title:
        print(f"\n=== {title} ===")
    df.show(n, truncate=False)
