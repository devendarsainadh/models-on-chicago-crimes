#!/usr/bin/env python3
import time
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


BASE_DIR = Path(__file__).resolve().parents[1]
PARQUET_PATH = BASE_DIR / "data" / "processed" / "chicago_crime_parquet"
OUT_PATH = BASE_DIR / "tableau" / "exports" / "performance_profile"


def main() -> None:
    spark = (
        SparkSession.builder.appName("ChicagoCrimePerformanceProfiler")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "64")
        .getOrCreate()
    )

    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"Parquet path not found: {PARQUET_PATH}")

    df = spark.read.parquet(str(PARQUET_PATH)).cache()
    df.count()

    rows = []
    queries = [
        ("q1_year_count", lambda x: x.groupBy("year").count()),
        ("q2_type_count", lambda x: x.groupBy("primary_type").count()),
        ("q3_year_type", lambda x: x.groupBy("year", "primary_type").agg(F.avg("arrest_label").alias("avg_arrest"))),
    ]

    for name, fn in queries:
        t0 = time.time()
        _ = fn(df).count()
        rt = time.time() - t0
        rows.append((name, float(rt)))

    out_df = spark.createDataFrame(rows, ["query_name", "runtime_sec"])
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    out_df.coalesce(1).write.mode("overwrite").option("header", True).csv(str(OUT_PATH))

    df.unpersist()
    spark.stop()
    print(f"Performance profile written to: {OUT_PATH}")


if __name__ == "__main__":
    main()
