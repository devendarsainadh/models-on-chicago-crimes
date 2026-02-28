#!/usr/bin/env python3
import os
import pickle
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pyspark import StorageLevel
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.classification import GBTClassifier, LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.param.shared import Param, Params, TypeConverters
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType, StringType, StructField, StructType


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "samples"
PROCESSED_DIR = DATA_DIR / "processed"
TABLEAU_DIR = BASE_DIR / "tableau" / "exports"
TABLEAU_STABLE_DIR = BASE_DIR / "tableau" / "exports_stable"
MODELS_DIR = PROCESSED_DIR / "models"
CHECKPOINT_DIR = PROCESSED_DIR / "checkpoints"

CSV_URL = "https://data.cityofchicago.org/resource/ijzp-q8t2.csv?$limit={limit}"


def sanitize_csv_text(value: str) -> str:
    text = str(value or "")
    text = text.replace(str(BASE_DIR), "<project>")
    text = re.sub(r"/Users/\S+", "<path>", text)
    text = re.sub(r"[A-Za-z]:\\\\\S+", "<path>", text)
    return text


class TimeFeatureTransformer(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    hourCol = Param(Params._dummy(), "hourCol", "Hour column", typeConverter=TypeConverters.toString)
    dateCol = Param(Params._dummy(), "dateCol", "Date column", typeConverter=TypeConverters.toString)

    def __init__(self, hourCol="hour", dateCol="date_ts"):
        super().__init__()
        self._setDefault(hourCol="hour", dateCol="date_ts")
        self._set(hourCol=hourCol, dateCol=dateCol)

    def _transform(self, dataset: DataFrame) -> DataFrame:
        hour_col = self.getOrDefault(self.hourCol)
        date_col = self.getOrDefault(self.dateCol)
        return (
            dataset.withColumn("is_night", F.when((F.col(hour_col) >= 20) | (F.col(hour_col) <= 5), 1).otherwise(0))
            .withColumn("is_weekend", F.when(F.dayofweek(F.col(date_col)).isin([1, 7]), 1).otherwise(0))
        )


def build_spark() -> SparkSession:
    spark = (
        SparkSession.builder.appName("ChicagoCrimeSimple")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.autoBroadcastJoinThreshold", str(10 * 1024 * 1024))
        .getOrCreate()
    )
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    spark.sparkContext.setCheckpointDir(str(CHECKPOINT_DIR))
    return spark


def fallback_df(spark: SparkSession) -> DataFrame:
    rows = [
        ("1001", "2024-01-03T11:15:00", "THEFT", "true", "false", "12", "121"),
        ("1002", "2024-01-04T23:48:00", "BATTERY", "false", "true", "7", "713"),
        ("1003", "2024-02-01T03:20:00", "ROBBERY", "true", "false", "11", "1112"),
        ("1004", "2024-02-19T15:30:00", "THEFT", "false", "false", "12", "1212"),
        ("1005", "2024-03-21T21:10:00", "ASSAULT", "true", "true", "4", "412"),
        ("1006", "2024-04-10T08:45:00", "NARCOTICS", "true", "false", "1", "112"),
    ]
    schema = StructType(
        [
            StructField("id", StringType(), True),
            StructField("date", StringType(), True),
            StructField("primary_type", StringType(), True),
            StructField("arrest", StringType(), True),
            StructField("domestic", StringType(), True),
            StructField("district", StringType(), True),
            StructField("beat", StringType(), True),
        ]
    )
    return spark.createDataFrame(rows, schema)


def ingest_validate(spark: SparkSession, row_limit: int) -> Tuple[DataFrame, List[Dict[str, str]]]:
    lineage: List[Dict[str, str]] = []
    url = CSV_URL.format(limit=row_limit)
    local_full_csv = RAW_DIR / "chicago_crime_full.csv"

    if local_full_csv.exists():
        df = spark.read.csv(str(local_full_csv), header=True, inferSchema=True)
        ingest_max_rows = int(os.getenv("INGEST_MAX_ROWS", "500000"))
        if ingest_max_rows > 0:
            df = df.limit(ingest_max_rows)
        lineage.append({"step": "ingestion", "status": "success", "source": "local_sample_csv", "ts": str(datetime.utcnow())})
    else:
        try:
            df = spark.read.csv(url, header=True, inferSchema=True)
            lineage.append({"step": "ingestion", "status": "success", "source": "city_of_chicago_api", "ts": str(datetime.utcnow())})
        except Exception as exc:
            df = fallback_df(spark)
            lineage.append({"step": "ingestion", "status": "fallback", "source": "in-memory", "details": str(exc)[:300], "ts": str(datetime.utcnow())})

    df = df.toDF(*[c.lower().strip() for c in df.columns])

    required = ["id", "date", "primary_type", "arrest", "domestic", "district", "beat"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    cleaned = (
        df.withColumn("id", F.col("id").cast("string"))
        .withColumn("date_ts", F.to_timestamp("date"))
        .withColumn("year", F.year("date_ts"))
        .withColumn("month", F.month("date_ts"))
        .withColumn("hour", F.hour("date_ts"))
        .withColumn("arrest_label", F.when(F.lower(F.col("arrest").cast("string")) == "true", 1.0).otherwise(0.0))
        .withColumn("domestic_int", F.when(F.lower(F.col("domestic").cast("string")) == "true", 1).otherwise(0))
        .dropna(subset=["id", "date_ts", "primary_type", "district", "beat"])
        .dropDuplicates(["id"])
    )

    return cleaned, lineage


def store_data(df: DataFrame, lineage: List[Dict[str, str]]) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / "chicago_crime_ingested_csv"
    raw_snapshot_rows = int(os.getenv("RAW_SNAPSHOT_ROWS", "500000"))
    df.limit(raw_snapshot_rows).coalesce(1).write.mode("overwrite").option("header", True).csv(str(raw_path))

    parquet_path = PROCESSED_DIR / "chicago_crime_parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.repartition("year", "primary_type").write.mode("overwrite").partitionBy("year", "primary_type").parquet(str(parquet_path))
    lineage.append({"step": "storage", "status": "success", "source": "processed_parquet_dataset", "ts": str(datetime.utcnow())})
    return parquet_path


def data_quality_and_business(df: DataFrame) -> None:
    TABLEAU_DIR.mkdir(parents=True, exist_ok=True)

    dim = df.sparkSession.createDataFrame(
        [("THEFT", "Property"), ("BATTERY", "Violent"), ("ROBBERY", "Violent"), ("ASSAULT", "Violent"), ("NARCOTICS", "Drug")],
        ["primary_type", "crime_family"],
    )

    joined = df.join(F.broadcast(dim), "primary_type", "left").persist(StorageLevel.MEMORY_AND_DISK)
    joined = joined.withColumn("crime_family", F.coalesce(F.col("crime_family"), F.lit("Other")))

    quality_all = joined.agg(
        F.lit("ALL").alias("year"),
        F.count("*").alias("row_count"),
        F.countDistinct("id").alias("distinct_ids"),
        (F.count("*") - F.countDistinct("id")).alias("duplicate_id_count"),
        F.sum(F.when(F.col("date_ts").isNull(), 1).otherwise(0)).alias("null_date_count"),
        F.avg("arrest_label").alias("arrest_rate"),
        F.avg("domestic_int").alias("domestic_rate"),
        F.min("date_ts").alias("min_date"),
        F.max("date_ts").alias("max_date"),
    )
    quality_yearly = (
        joined.groupBy("year")
        .agg(
            F.count("*").alias("row_count"),
            F.countDistinct("id").alias("distinct_ids"),
            (F.count("*") - F.countDistinct("id")).alias("duplicate_id_count"),
            F.sum(F.when(F.col("date_ts").isNull(), 1).otherwise(0)).alias("null_date_count"),
            F.avg("arrest_label").alias("arrest_rate"),
            F.avg("domestic_int").alias("domestic_rate"),
            F.min("date_ts").alias("min_date"),
            F.max("date_ts").alias("max_date"),
        )
        .withColumn("year", F.col("year").cast("string"))
    )
    quality = quality_all.unionByName(quality_yearly).orderBy("year")

    business = (
        joined.groupBy("year", "primary_type", "crime_family")
        .agg(F.count("*").alias("crime_count"), F.avg("arrest_label").alias("arrest_rate"), F.avg("domestic_int").alias("domestic_rate"))
        .withColumn("year_total", F.sum("crime_count").over(Window.partitionBy("year")))
        .withColumn("crime_share_pct", (F.col("crime_count") / F.col("year_total")) * F.lit(100.0))
        .withColumn("prev_year_count", F.lag("crime_count").over(Window.partitionBy("primary_type").orderBy("year")))
        .withColumn(
            "yoy_change_pct",
            F.when(F.col("prev_year_count").isNull() | (F.col("prev_year_count") == 0), None).otherwise(
                ((F.col("crime_count") - F.col("prev_year_count")) / F.col("prev_year_count")) * F.lit(100.0)
            ),
        )
        .withColumn("rank_in_year", F.row_number().over(Window.partitionBy("year").orderBy(F.desc("crime_count"))))
        .drop("year_total", "prev_year_count")
        .orderBy(F.desc("year"), F.asc("rank_in_year"))
    )

    quality.coalesce(1).write.mode("overwrite").option("header", True).csv(str(TABLEAU_DIR / "dashboard1_data_quality"))
    business.coalesce(1).write.mode("overwrite").option("header", True).csv(str(TABLEAU_DIR / "dashboard3_business_insights"))

    joined.unpersist()


def temporal_split(df: DataFrame) -> Tuple[DataFrame, DataFrame, DataFrame]:
    bounds = df.select(F.min("date_ts").alias("min_d"), F.max("date_ts").alias("max_d")).collect()[0]
    min_d, max_d = bounds["min_d"], bounds["max_d"]
    if min_d is None or max_d is None or min_d == max_d:
        train, rem = df.randomSplit([0.7, 0.3], seed=42)
        val, test = rem.randomSplit([0.5, 0.5], seed=42)
        return train, val, test

    train_cut = min_d + (max_d - min_d) * 0.7
    val_cut = min_d + (max_d - min_d) * 0.85

    train = df.filter(F.col("date_ts") <= F.lit(train_cut))
    val = df.filter((F.col("date_ts") > F.lit(train_cut)) & (F.col("date_ts") <= F.lit(val_cut)))
    test = df.filter(F.col("date_ts") > F.lit(val_cut))

    if train.count() < 5 or val.count() < 2 or test.count() < 2:
        train, rem = df.randomSplit([0.7, 0.3], seed=42)
        val, test = rem.randomSplit([0.5, 0.5], seed=42)
    return train, val, test


def make_pipeline(clf) -> Pipeline:
    return Pipeline(
        stages=[
            TimeFeatureTransformer(),
            StringIndexer(inputCol="primary_type", outputCol="primary_type_idx", handleInvalid="keep"),
            StringIndexer(inputCol="district", outputCol="district_idx", handleInvalid="keep"),
            StringIndexer(inputCol="beat", outputCol="beat_idx", handleInvalid="keep"),
            OneHotEncoder(
                inputCols=["primary_type_idx", "district_idx", "beat_idx"],
                outputCols=["primary_type_ohe", "district_ohe", "beat_ohe"],
            ),
            VectorAssembler(
                inputCols=["domestic_int", "hour", "month", "is_night", "is_weekend", "primary_type_ohe", "district_ohe", "beat_ohe"],
                outputCol="features",
            ),
            clf,
        ]
    )


def train_models(train: DataFrame, val: DataFrame, test: DataFrame) -> Tuple[DataFrame, str]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    evaluator_auc = BinaryClassificationEvaluator(labelCol="arrest_label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol="arrest_label", predictionCol="prediction", metricName="f1")
    evaluator_precision = MulticlassClassificationEvaluator(
        labelCol="arrest_label", predictionCol="prediction", metricName="weightedPrecision"
    )
    evaluator_recall = MulticlassClassificationEvaluator(
        labelCol="arrest_label", predictionCol="prediction", metricName="weightedRecall"
    )

    train_count = train.count()
    label_distinct = train.select("arrest_label").distinct().count()
    type_distinct = train.select("primary_type").distinct().count()
    if train_count < 200 or label_distinct < 2 or type_distinct < 2:
        records = [
            ("logistic_regression", None, None, None, None, None, "not_trained_tiny_data", "mllib", "not_trained_tiny_data"),
            ("random_forest", None, None, None, None, None, "not_trained_tiny_data", "mllib", "not_trained_tiny_data"),
            ("gbt", None, None, None, None, None, "not_trained_tiny_data", "mllib", "not_trained_tiny_data"),
        ]
        metrics = train.sparkSession.createDataFrame(
            records,
            ["model_name", "val_auc", "test_auc", "test_f1", "test_precision", "test_recall", "artifact_path", "framework", "model_status"],
        )
        return metrics, "logistic_regression"

    specs = [
        ("logistic_regression", LogisticRegression(labelCol="arrest_label", featuresCol="features", maxIter=20), "lr"),
        ("random_forest", RandomForestClassifier(labelCol="arrest_label", featuresCol="features", numTrees=40), "rf"),
        ("gbt", GBTClassifier(labelCol="arrest_label", featuresCol="features", maxIter=20), "gbt"),
    ]

    records = []
    best_model_name = "logistic_regression"
    best_auc = -1.0

    for name, clf, kind in specs:
        pipe = make_pipeline(clf)
        est = pipe.getStages()[-1]
        grid = ParamGridBuilder()
        if kind == "lr":
            grid = grid.addGrid(est.regParam, [0.0, 0.1]).addGrid(est.elasticNetParam, [0.0, 0.5])
        elif kind == "rf":
            grid = grid.addGrid(est.maxDepth, [3, 6]).addGrid(est.minInstancesPerNode, [1, 3])
        else:
            grid = grid.addGrid(est.maxDepth, [3, 5]).addGrid(est.stepSize, [0.05, 0.1])

        cv = CrossValidator(estimator=pipe, estimatorParamMaps=grid.build(), evaluator=evaluator_auc, numFolds=2, parallelism=2, seed=42)
        cv_model = cv.fit(train)

        val_pred = cv_model.transform(val)
        test_pred = cv_model.transform(test)

        val_label_distinct = val_pred.select("arrest_label").distinct().count()
        test_label_distinct = test_pred.select("arrest_label").distinct().count()
        val_auc = float(evaluator_auc.evaluate(val_pred)) if val_label_distinct >= 2 else None
        test_auc = float(evaluator_auc.evaluate(test_pred)) if test_label_distinct >= 2 else None
        test_f1 = float(evaluator_f1.evaluate(test_pred))
        test_precision = float(evaluator_precision.evaluate(test_pred))
        test_recall = float(evaluator_recall.evaluate(test_pred))

        path = MODELS_DIR / f"{name}_cv_model"
        cv_model.bestModel.write().overwrite().save(str(path))
        records.append((name, val_auc, test_auc, test_f1, test_precision, test_recall, str(path), "mllib", "trained"))

        auc_for_rank = test_auc if test_auc is not None else -1.0
        if auc_for_rank > best_auc:
            best_auc = auc_for_rank
            best_model_name = name

    metrics = train.sparkSession.createDataFrame(
        records,
        ["model_name", "val_auc", "test_auc", "test_f1", "test_precision", "test_recall", "artifact_path", "framework", "model_status"],
    )
    return metrics, best_model_name


def sklearn_baseline(
    train: DataFrame, test: DataFrame
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], str, str]:
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.pipeline import Pipeline as SKPipeline
    from sklearn.preprocessing import OneHotEncoder as SKOneHotEncoder

    cols = ["primary_type", "district", "beat", "domestic_int", "hour", "month", "arrest_label"]
    train_pd = train.select(*cols).limit(50000).toPandas()
    test_pd = test.select(*cols).limit(20000).toPandas()

    if len(train_pd) < 10 or len(test_pd) < 5:
        return None, None, None, None, "insufficient_rows_for_sklearn", "insufficient_rows_for_sklearn"

    x_train = train_pd.drop(columns=["arrest_label"])
    y_train = train_pd["arrest_label"].astype(int)
    x_test = test_pd.drop(columns=["arrest_label"])
    y_test = test_pd["arrest_label"].astype(int)
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        return None, None, None, None, "single_class_split", "single_class_split"

    pre = ColumnTransformer(
        transformers=[
            ("cat", SKOneHotEncoder(handle_unknown="ignore"), ["primary_type", "district", "beat"]),
            ("num", "passthrough", ["domestic_int", "hour", "month"]),
        ]
    )
    pipe = SKPipeline(steps=[("pre", pre), ("clf", LogisticRegression(max_iter=200))])
    pipe.fit(x_train, y_train)
    prob = pipe.predict_proba(x_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    auc = float(roc_auc_score(y_test, prob))
    f1 = float(f1_score(y_test, pred, zero_division=0))
    precision = float(precision_score(y_test, pred, zero_division=0))
    recall = float(recall_score(y_test, pred, zero_division=0))

    out_path = MODELS_DIR / "sklearn_logistic.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(pipe, f)
    return auc, f1, precision, recall, "trained", str(out_path)


def bootstrap_accuracy_ci(pred_df: DataFrame, n_boot: int = 100) -> Tuple[float, float]:
    import random

    rows = pred_df.select("prediction", "arrest_label").limit(20000).collect()
    if len(rows) < 20:
        return 0.0, 0.0

    vals = [(int(r[0]), int(r[1])) for r in rows]
    scores = []
    for _ in range(n_boot):
        sample = [vals[random.randint(0, len(vals) - 1)] for _ in range(len(vals))]
        acc = sum(1 for p, y in sample if p == y) / len(sample)
        scores.append(acc)
    scores.sort()
    return float(scores[int(0.025 * len(scores))]), float(scores[int(0.975 * len(scores))])


def model_and_scaling_outputs(df: DataFrame, lineage: List[Dict[str, str]]) -> None:
    ml_sample_fraction = float(os.getenv("ML_SAMPLE_FRACTION", "0.05"))
    ml_df = df.sample(withReplacement=False, fraction=ml_sample_fraction, seed=42)
    if ml_df.count() < 200:
        ml_df = df

    train, val, test = temporal_split(ml_df)
    train.persist(StorageLevel.MEMORY_AND_DISK)
    val.persist(StorageLevel.MEMORY_AND_DISK)
    test.persist(StorageLevel.MEMORY_AND_DISK)

    mllib, best_name = train_models(train, val, test)
    sk_auc, sk_f1, sk_precision, sk_recall, sk_status, sk_path = sklearn_baseline(train, test)

    sk_schema = StructType(
        [
            StructField("model_name", StringType(), False),
            StructField("val_auc", DoubleType(), True),
            StructField("test_auc", DoubleType(), True),
            StructField("test_f1", DoubleType(), True),
            StructField("test_precision", DoubleType(), True),
            StructField("test_recall", DoubleType(), True),
            StructField("artifact_path", StringType(), False),
            StructField("framework", StringType(), False),
            StructField("model_status", StringType(), False),
        ]
    )
    sk = df.sparkSession.createDataFrame(
        [("sklearn_logistic", None, sk_auc, sk_f1, sk_precision, sk_recall, sk_path, "sklearn", sk_status)],
        schema=sk_schema,
    )
    perf = mllib.unionByName(sk)

    test_rows_count = test.count()
    test_positive_rows = test.filter(F.col("arrest_label") == 1.0).count()
    train_pos_rate = float(train.agg(F.avg("arrest_label").alias("r")).first()["r"] or 0.0)
    val_pos_rate = float(val.agg(F.avg("arrest_label").alias("r")).first()["r"] or 0.0)
    test_pos_rate = float(test.agg(F.avg("arrest_label").alias("r")).first()["r"] or 0.0)

    best_path = mllib.filter(F.col("model_name") == best_name).first()["artifact_path"]
    if best_path != "not_trained_tiny_data" and Path(best_path).exists() and test_rows_count >= 50:
        best_model = PipelineModel.load(best_path)
        pred = best_model.transform(test)
        ci_lo, ci_hi = bootstrap_accuracy_ci(pred, n_boot=100)
    else:
        ci_lo, ci_hi = None, None

    baseline_auc = perf.filter(F.col("model_name") == "logistic_regression").select("test_auc").first()
    baseline_auc_val = float(baseline_auc["test_auc"]) if baseline_auc and baseline_auc["test_auc"] is not None else 0.0
    best_auc_val = float(perf.agg(F.max("test_auc").alias("mx")).first()["mx"] or 0.0)

    perf = (
        perf.withColumn("best_model", F.when(F.col("model_name") == best_name, 1).otherwise(0))
        .withColumn("auc_rank", F.dense_rank().over(Window.orderBy(F.desc("test_auc"))))
        .withColumn("auc_gap_vs_best", F.lit(best_auc_val) - F.col("test_auc"))
        .withColumn("auc_lift_vs_logreg", F.col("test_auc") - F.lit(baseline_auc_val))
        .withColumn("bootstrap_acc_ci_low", F.lit(ci_lo))
        .withColumn("bootstrap_acc_ci_high", F.lit(ci_hi))
        .withColumn("business_metric_expected_arrest_gain", F.col("test_recall") * F.lit(float(test_positive_rows)))
        .withColumn("train_rows", F.lit(train.count()))
        .withColumn("val_rows", F.lit(val.count()))
        .withColumn("test_rows", F.lit(test_rows_count))
        .withColumn("test_positive_rows", F.lit(test_positive_rows))
        .withColumn("train_positive_rate", F.lit(train_pos_rate))
        .withColumn("val_positive_rate", F.lit(val_pos_rate))
        .withColumn("test_positive_rate", F.lit(test_pos_rate))
        .withColumn("run_ts", F.current_timestamp())
    )
    perf_export_cols = [
        "model_name",
        "val_auc",
        "test_auc",
        "test_f1",
        "test_precision",
        "test_recall",
        "framework",
        "model_status",
        "best_model",
        "auc_rank",
        "auc_gap_vs_best",
        "auc_lift_vs_logreg",
        "bootstrap_acc_ci_low",
        "bootstrap_acc_ci_high",
        "business_metric_expected_arrest_gain",
        "train_rows",
        "val_rows",
        "test_rows",
        "test_positive_rows",
        "train_positive_rate",
        "val_positive_rate",
        "test_positive_rate",
        "run_ts",
    ]
    perf.select(*perf_export_cols).coalesce(1).write.mode("overwrite").option("header", True).csv(
        str(TABLEAU_DIR / "dashboard2_model_performance")
    )

    base = df.select("year", "primary_type", "arrest_label").cache()
    base.count()
    rows = []
    for workers in [1, 2, 4]:
        t0 = time.time()
        _ = base.repartition(workers * 8).groupBy("year", "primary_type").count().count()
        rt = time.time() - t0
        rows.append(("strong_scaling", workers, 1.0, rt, rt * workers, "shuffle"))
    for workers in [1, 2, 4]:
        frac = min(1.0, 0.25 * workers)
        t0 = time.time()
        _ = base.sample(False, frac, 42).repartition(workers * 8).groupBy("year").avg("arrest_label").count()
        rt = time.time() - t0
        rows.append(("weak_scaling", workers, frac, rt, rt * workers, "io+shuffle"))
    base.unpersist()

    scaling = df.sparkSession.createDataFrame(rows, ["test_type", "workers", "data_fraction", "runtime_sec", "cost_index", "bottleneck_hint"])
    strong_base_rt = float(
        scaling.filter((F.col("test_type") == "strong_scaling") & (F.col("workers") == 1)).select("runtime_sec").first()["runtime_sec"]
    )
    scaling = (
        scaling.withColumn(
            "speedup",
            F.when(F.col("test_type") == "strong_scaling", F.lit(strong_base_rt) / F.col("runtime_sec")).otherwise(F.lit(None)),
        )
        .withColumn(
            "efficiency",
            F.when(F.col("test_type") == "strong_scaling", F.col("speedup") / F.col("workers")).otherwise(F.lit(None)),
        )
        .withColumn("throughput_proxy", F.col("data_fraction") / F.col("runtime_sec"))
    )
    scaling.coalesce(1).write.mode("overwrite").option("header", True).csv(str(TABLEAU_DIR / "dashboard4_scalability"))

    lineage.append({"step": "ml_scaling", "status": "success", "source": "mllib+sklearn+scaling", "details": f"best={best_name}", "ts": str(datetime.utcnow())})

    train.unpersist()
    val.unpersist()
    test.unpersist()


def write_lineage(spark: SparkSession, lineage: List[Dict[str, str]]) -> None:
    rows = [
        (
            x.get("step", ""),
            x.get("status", ""),
            sanitize_csv_text(x.get("details", "")),
            x.get("ts", ""),
        )
        for x in lineage
    ]
    df = spark.createDataFrame(rows, ["step", "status", "details", "ts"])
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(str(TABLEAU_DIR / "lineage_log"))


def materialize_stable_tableau_csvs() -> None:
    TABLEAU_STABLE_DIR.mkdir(parents=True, exist_ok=True)
    for dataset_dir in sorted(TABLEAU_DIR.iterdir()):
        if not dataset_dir.is_dir():
            continue
        part_files = sorted(dataset_dir.glob("part-*.csv"))
        if not part_files:
            continue
        stable_file = TABLEAU_STABLE_DIR / f"{dataset_dir.name}.csv"
        shutil.copyfile(part_files[0], stable_file)


def main() -> None:
    row_limit = int(os.getenv("ROW_LIMIT", "100000"))
    spark = build_spark()
    lineage: List[Dict[str, str]] = []
    try:
        df, lineage = ingest_validate(spark, row_limit)
        _ = store_data(df, lineage)
        data_quality_and_business(df)
        model_and_scaling_outputs(df, lineage)
        write_lineage(spark, lineage)
        materialize_stable_tableau_csvs()
        print(f"Done. Tableau exports: {TABLEAU_DIR}")
    except Exception as exc:
        lineage.append({"step": "pipeline", "status": "failed", "source": "main", "details": str(exc)[:800], "ts": str(datetime.utcnow())})
        write_lineage(spark, lineage)
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
