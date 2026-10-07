"""SparkSession factory. Defaults suit a laptop (local[*]); point CHURN_SPARK_MASTER at a cluster
(yarn / spark://host:7077) and CHURN_RAW_URI at HDFS to run the same code at scale."""
import os
import sys

import re
import shutil
from pathlib import Path

from pyspark.sql import SparkSession


def _java_major(java_home):
    """Major version from <java_home>/release, or None if unreadable."""
    try:
        text = (Path(java_home) / "release").read_text()
        v = re.search(r'JAVA_VERSION="(\d+)(?:\.(\d+))?', text)
        return (int(v.group(2)) if v.group(1) == "1" else int(v.group(1))) if v else None
    except OSError:
        return None


def _ensure_supported_java():
    """Spark 4 / Hadoop 3.4 fail on JDK 23+ ('getSubject is not supported'). If JAVA_HOME points at one,
    fall back to the first java on PATH (JDK 17/21) for this process only."""
    jh = os.getenv("CHURN_JAVA_HOME") or os.getenv("JAVA_HOME")
    major = _java_major(jh) if jh else None
    if major is not None and 17 <= major <= 21:
        os.environ["JAVA_HOME"] = jh
        return
    on_path = shutil.which("java")
    if on_path:
        candidate = Path(on_path).resolve().parent.parent
        if (_java_major(candidate) or 0) in range(17, 22):
            os.environ["JAVA_HOME"] = str(candidate)


def get_spark(app_name="churn-prediction"):
    _ensure_supported_java()
    # Make workers use the same interpreter as the driver (matters on Windows / venvs).
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    builder = (
        SparkSession.builder.appName(app_name)
        .master(os.getenv("CHURN_SPARK_MASTER", "local[*]"))
        .config("spark.driver.memory", os.getenv("CHURN_DRIVER_MEMORY", "4g"))
        .config("spark.sql.shuffle.partitions", os.getenv("CHURN_SHUFFLE_PARTITIONS", "16"))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
    )
    # Hive metastore support: `set CHURN_ENABLE_HIVE=1` lets spark.sql() read/write Hive tables.
    if os.getenv("CHURN_ENABLE_HIVE") == "1":
        builder = builder.enableHiveSupport()

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark
