# Chicago Crime Simple Project

This follows the exact schema you requested and keeps everything straightforward.

## Run
```bash
cd project
bash scripts/setup_environment.sh
python scripts/run_pipeline.py
python scripts/performance_profiler.py
```

## Data source
- Socrata CSV endpoint:
  - `https://data.cityofchicago.org/resource/ijzp-q8t2.csv?$limit=1000000`

## Main outputs
- Parquet: `data/processed/chicago_crime_parquet`
- Models: `data/processed/models`
- Tableau export CSV folders:
  - `tableau/exports/dashboard1_data_quality`
  - `tableau/exports/dashboard2_model_performance`
  - `tableau/exports/dashboard3_business_insights`
  - `tableau/exports/dashboard4_scalability`
  - `tableau/exports/lineage_log`
  - `tableau/exports/performance_profile`

## Technical requirement coverage (simple)
1. Data engineering
- SparkSession tuning, ingestion validation, Parquet partitioning, lineage log.
- Broadcast join + persist/unpersist.

2. Distributed ML
- 3 MLlib models: LR, RF, GBT.
- sklearn baseline logistic regression.
- CrossValidator with parallelism.
- Custom transformer (`TimeFeatureTransformer`).
- Model serialization and bootstrap CI.

3. Tableau
- Four dashboard-ready CSV outputs.
- Tableau README + config JSON for mapping.

4. Evaluation
- Temporal split fallback to random split.
- Cross-validation and basic business metric proxy.

## Notes
- If endpoint access fails, script uses a built-in tiny fallback dataset so pipeline still runs.
- Use `ROW_LIMIT` env var to control ingestion volume.
