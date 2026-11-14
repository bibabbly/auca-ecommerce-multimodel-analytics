# Distributed Multi-Model Analytics for E-commerce Data

AUCA Big Data Analytics final project. A synthetic e-commerce dataset is stored in **MongoDB**
(document model) and **HBase** (wide-column model) and analysed with **Apache Spark** (PySpark).

- **MongoDB:** catalog, customer profiles (with an embedded purchase summary), and orders (with embedded line items).
- **HBase:** the session clickstream (row key `user_id|reverse_timestamp|session_id`) and daily product metrics.
- **Spark:** cleaning, recommendation indicators, cohort analysis, Spark SQL, and a funnel + CLV analysis that joins live HBase and MongoDB data.

📄 **Technical report:** [`report/technical_report.pdf`](report/technical_report.pdf) (source: `report/technical_report.html`)
📊 **Charts:** [`output/charts/`](output/charts)

## Repository layout

```
dataset_generator.py            original course generator (Colab notebook format)
src/generate_data.py            configurable, streaming, reproducible version (JSON Lines)
src/common.py                   shared paths, connection settings, JSON readers
src/mongodb/load_mongo.py       document model + loader + indexes
src/mongodb/aggregations.py     3 aggregation pipelines (revenue/profit, RFM segments, top products)
src/hbase/create_tables.hbase   HBase schema (column families, pre-splits) - HBase shell
src/hbase/hbase_schema.py       row-key helpers (reverse timestamp, ranges)
src/hbase/load_hbase.py         loads sessions + builds product_metrics
src/hbase/query_hbase.py        access patterns: latest sessions, time range, journey, product metrics
src/spark/spark_common.py       SparkSession, explicit schemas, shared cleaning layer
src/spark/clean_normalize.py    batch job 1: cleaning + data-quality report
src/spark/recommendations.py    batch job 2: bought-together / viewed-together / category affinity
src/spark/cohort_analysis.py    batch job 3: registration and first-purchase cohorts
src/spark/spark_sql_analytics.py  6 Spark SQL queries (CTEs, window functions, cross-store joins)
src/integration/funnel_clv.py   Part 3: funnel conversion + CLV from live HBase + MongoDB
src/visualize.py                Part 4: the 6 report charts
docker-compose.yml              MongoDB 7.0 + HBase 2.1 (Thrift on 9090)
output/                         result tables (CSV/JSON) and charts
```

## Requirements

- Docker Desktop (for MongoDB and HBase)
- Python 3.11+ and Java 17 (for PySpark 4.x)
- About 8 GB RAM and 2 GB disk for the medium dataset

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce the results

Run these from the project root:

```bash
# 1. databases
docker compose up -d

# 2. dataset: 10k users, 5k products, 300k sessions, 75k orders (about 2 min)
python src/generate_data.py --preset medium        # also: small | full (original 2M sessions)

# 3. MongoDB: load + aggregation pipelines
python src/mongodb/load_mongo.py
python src/mongodb/aggregations.py

# 4. HBase: schema + load + queries
docker exec -i ecom-hbase hbase shell -n < src/hbase/create_tables.hbase
python src/hbase/load_hbase.py
python src/hbase/query_hbase.py --user user_000042 --product prod_00628

# 5. Spark batch jobs and Spark SQL
python src/spark/clean_normalize.py
python src/spark/recommendations.py
python src/spark/cohort_analysis.py
python src/spark/spark_sql_analytics.py

# 6. integrated analytics (reads live HBase + MongoDB) and charts
python src/integration/funnel_clv.py
python src/visualize.py
```

On PowerShell, step 4's input redirect needs `Get-Content`:

```bash
Get-Content src/hbase/create_tables.hbase | docker exec -i ecom-hbase hbase shell -n
```

### HBase shell examples

```
scan 'user_sessions', {ROWPREFIXFILTER => 'user_000042|', LIMIT => 3, COLUMNS => ['s']}
get  'user_sessions', '<row key from the scan above>', {COLUMN => ['pv', 'c']}
scan 'product_metrics', {STARTROW => 'prod_00628|20260901', STOPROW => 'prod_00628|20260930~'}
```

The HBase Master UI is at http://localhost:16010.

### Regenerate the report PDF

The report uses Microsoft Edge or Chrome in headless mode:

```bash
msedge --headless=new --no-pdf-header-footer --print-to-pdf="report/technical_report.pdf" report/technical_report.html
```

## Windows notes

- **Spark ignores a system-wide `SPARK_HOME`.** `spark_common.py` always uses the Spark bundled with the
  installed `pyspark` package, so an older `C:\spark` does not break the jobs. Set `USE_SYSTEM_SPARK=1` to opt out.
- **Paths with spaces** (such as `FINAL PROJECT THEO`) are converted to their 8.3 short form for Spark's launch scripts.
- **Spark cannot write files on Windows without Hadoop `winutils.exe`.** Result tables are therefore written through
  pandas. On Linux, `clean_normalize.py --parquet data/clean` persists the cleaned tables as Parquet.

## Configuration

| Variable | Default |
|---|---|
| `MONGO_URI` / `MONGO_DB` | `mongodb://localhost:27017` / `ecommerce` |
| `HBASE_HOST` / `HBASE_PORT` | `localhost` / `9090` |
| `DATA_DIR` | `data/raw` |
| `SPARK_MASTER` / `SPARK_DRIVER_MEMORY` | `local[*]` / `6g` |
