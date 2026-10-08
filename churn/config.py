"""Central configuration. Every path can be overridden with an environment variable,
so the same code runs against local disk, HDFS (hdfs://namenode:9000/...) or S3/ADLS/GCS."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Where the raw tables live. Point this at HDFS / cloud storage to ingest at scale, e.g.
#   set CHURN_RAW_URI=hdfs://namenode:9000/churn/raw
RAW_URI = os.getenv("CHURN_RAW_URI", (ROOT / "data" / "raw").as_posix())

# Real IBM / Kaggle "Telco Customer Churn" CSV that anchors the customer core + churn label.
REAL_CSV = os.getenv("CHURN_REAL_CSV", (ROOT / "data" / "real" / "Telco-Customer-Churn.csv").as_posix())

# Local folder consumed by the dashboard (small: one row per customer + metrics).
OUTPUT_DIR = Path(os.getenv("CHURN_OUTPUT_DIR", ROOT / "outputs"))

# Reference date = end of the observation window. Features only use data before it.
SNAPSHOT_DATE = "2025-12-31"

# Raw tables produced by the generator / expected by the pipeline.
TABLES = ["customers", "usage", "payments", "tickets", "service_calls"]

SEED = 42
