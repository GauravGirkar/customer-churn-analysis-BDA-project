"""Step 1 - Data ingestion: load the raw tables from HDFS / cloud storage / local disk and
register them as Spark SQL views (the same names a Hive database would expose)."""
import os

from churn.config import RAW_URI, TABLES


def load_raw_tables(spark, base_uri=RAW_URI, fmt=None):
    fmt = fmt or os.getenv("CHURN_RAW_FORMAT", "parquet")
    counts = {}
    for name in TABLES:
        path = f"{base_uri.rstrip('/')}/{name}.{fmt}"
        reader = spark.read
        if fmt == "csv":
            reader = reader.option("header", True).option("inferSchema", True)
        df = reader.format(fmt).load(path)
        df.createOrReplaceTempView(name)
        counts[name] = df.count()
        print(f"  ingested {name:14s} {counts[name]:>10,d} rows  <- {path}")
    return counts
