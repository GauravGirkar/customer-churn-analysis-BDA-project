# End-to-End Customer Churn Prediction (Big Data Analytics mini-project)

Predicts which telecom customers are likely to churn using the full BDA pipeline:
**ingestion → Spark SQL processing → distributed modelling → evaluation → dashboard.**

```
 raw tables (Parquet/CSV)          Spark SQL                    Spark MLlib / boosters            Streamlit
 customers  usage  payments  ──►  per-customer feature  ──►  LogReg · RandomForest · GBT  ──►  churn probability
 tickets  service_calls  (HDFS     table (36 features,         LightGBM · XGBoost                per customer, ROC/PR,
 or local / S3 / ADLS)             6-month window)             ROC-AUC · PR-AUC · P/R/F1         drivers, lookup, export
```

## Quick start

Requirements: Python 3.10+, **JDK 17 or 21** (Spark 4 does not run on JDK 23+; the code auto-selects a JDK 17/21
from `PATH` if `JAVA_HOME` points at a newer one, or set `CHURN_JAVA_HOME`).

```bash
pip install -r requirements-pipeline.txt             # full Spark + boosters stack (dashboard-only deps live in dashboard/requirements.txt)
python -m churn.generate_data --customers 100000     # synthetic data -> data/raw/*.parquet
python -m churn.pipeline                             # Spark pipeline  -> outputs/scores.parquet + metrics.json
streamlit run dashboard/app.py                       # dashboard at http://localhost:8501
```

Windows one-liner: `.\run_all.ps1` (add `-Tune` for grid-search + 3-fold CV, `-Customers 500000` for more data).

## What maps to the brief

| Requirement | Where |
|---|---|
| Data ingestion from HDFS / cloud storage | [churn/ingest.py](churn/ingest.py) — any Spark URI via `CHURN_RAW_URI` (`hdfs://…`, `s3a://…`, local path) |
| Feature engineering: usage frequency, complaints/tickets, payment delays, tenure, customer-service interactions | [churn/features.py](churn/features.py) — one Spark SQL query, 36 features (usage level **and** trend, late-payment counts/ratios, complaints, escalations, cancellation inquiries, unresolved calls, sentiment, tenure…) |
| Hive / Spark SQL preprocessing | Plain `spark.sql(...)` over temp views; set `CHURN_ENABLE_HIVE=1` to use a Hive metastore |
| Logistic Regression, Random Forest, Gradient Boosting | [churn/models.py](churn/models.py) — MLlib `LogisticRegression`, `RandomForestClassifier`, `GBTClassifier` in `Pipeline`s (index → one-hot → assemble → scale) |
| XGBoost / LightGBM | Same file; trained on the Spark-engineered features, early-stopped on the validation split |
| ROC-AUC, Precision/Recall, F1 | [churn/evaluate.py](churn/evaluate.py) — plus PR-AUC, confusion matrix, top-decile lift |
| Dashboard of churn probability per customer | [dashboard/app.py](dashboard/app.py) |

## Results (100k customers, 22% churn, held-out test split)

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Lift @ top 10% |
|---|---|---|---|---|---|---|
| Logistic Regression (MLlib) | 0.815 | 0.588 | 0.498 | 0.627 | 0.555 | 3.22× |
| Random Forest (MLlib) | 0.807 | 0.582 | 0.466 | 0.653 | 0.544 | 3.20× |
| Gradient Boosted Trees (MLlib) | 0.818 | 0.592 | 0.500 | 0.640 | 0.562 | 3.26× |
| LightGBM | 0.823 | 0.605 | 0.490 | 0.663 | 0.564 | 3.31× |
| **XGBoost** | **0.823** | **0.605** | 0.507 | 0.639 | 0.566 | 3.31× |

Precision/recall/F1 use a threshold tuned for max F1 on the validation split (not the test split). The production
model is picked by *validation* AUC so the test numbers stay unbiased. Numbers will vary slightly with data seed and library versions.

## Dashboard tabs
- **Overview** – risk-band counts, probability histogram, predicted vs actual churn by contract / plan / region…
- **Customers** – filterable table of churn probability + plain-language risk factors, CSV export, single-customer lookup with every model's score
- **Model performance** – comparison table, ROC & PR curves, confusion matrix, cumulative-gains curve
- **Churn drivers** – feature importance per model, churn rate by complaints / late payments / tenure
- **Pipeline** – row counts, split sizes, run metadata

## About the data
Real telco data is private, so [churn/generate_data.py](churn/generate_data.py) simulates five relational tables
(100k customers → 600k usage rows, 600k payment rows, ~60k tickets, ~180k service calls). A hidden "dissatisfaction"
factor drives both behaviour (fading usage, late payments, tickets, unresolved calls) and the churn label, together with
contract, tenure, price, and a few non-linear effects (new-customer cliff, month-to-month × e-check, price-shocked premium users).
The model never sees the hidden factor. **To use a real dataset**, produce the same five tables (column names in the generator) and
point `CHURN_RAW_URI` at them — nothing else changes.

Leakage guard: features use only the 6-month window ending at `SNAPSHOT_DATE`; `churned` is the outcome after it.
The dashboard scores *all* customers; scores of training-split customers are optimistic, so use the
"held-out only" filter for an honest preview.

## Deploying the dashboard
The dashboard only needs `outputs/scores.parquet` + `outputs/metrics.json` (~7 MB, committed to git), so it deploys without Spark/Java.
Heavy training stays offline; re-run `run.bat rebuild`, commit the two files, push, and the app redeploys.
1. Push this folder to a GitHub repo (`outputs/` included).
2. [share.streamlit.io](https://share.streamlit.io) → *New app* → pick the repo, branch `main`, main file `dashboard/app.py`.
   It installs `dashboard/requirements.txt` (lightweight, sitting next to the app) — the only `requirements.txt` in the repo,
   so the Spark/booster training stack (`requirements-pipeline.txt`) is never pulled into the cloud build.
3. Deploy. Share the `*.streamlit.app` URL (set app visibility / viewer emails under *Settings → Sharing* if it must stay private).

## Running at scale (HDFS / cluster)
```bash
python -m churn.generate_data --customers 5000000 --out staging/
hdfs dfs -mkdir -p /churn/raw && hdfs dfs -put staging/*.parquet /churn/raw/

export CHURN_RAW_URI=hdfs://namenode:9000/churn/raw
export CHURN_SPARK_MASTER=yarn            # or spark://host:7077
export CHURN_DRIVER_MEMORY=8g CHURN_SHUFFLE_PARTITIONS=200
python -m churn.pipeline
```
Ingestion, SQL feature engineering and the three MLlib models run distributed. Two steps are intentionally driver-side
and are the scaling limits to revisit beyond ~10M customers: `toPandas()` of the feature table (used by the boosters and the
scores export) and metric computation in pandas. Replace them with `df.write.parquet(...)` of the scores and MLlib's
`BinaryClassificationEvaluator`, or train LightGBM/XGBoost with their Spark integrations.

## Layout
```
churn/            config · generate_data · spark_session · ingest · features · models · evaluate · pipeline
dashboard/app.py  Streamlit app (reads outputs/)
data/raw/         generated tables      outputs/  scores.parquet + metrics.json
```
