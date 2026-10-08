# End-to-End Customer Churn Prediction (Big Data Analytics mini-project)

Predicts which telecom customers are likely to churn using the full BDA pipeline:
**ingestion → Spark SQL processing → distributed modelling → evaluation → dashboard.**

Customer attributes and the churn label come from the **real** [IBM / Kaggle *Telco Customer Churn* dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
(7,043 real customers). See [About the data](#about-the-data) for how the behavioural tables are built on top of it.

```
 real Telco customers              Spark SQL                    Spark MLlib / boosters            Streamlit
 customers  usage  payments  ──►  per-customer feature  ──►  LogReg · RandomForest · GBT  ──►  churn probability
 tickets  service_calls  (HDFS     table (36 features,         LightGBM · XGBoost                per customer, ROC/PR,
 or local / S3 / ADLS)             6-month window)             ROC-AUC · PR-AUC · P/R/F1         drivers, lookup, export
```

## Quick start

Requirements: Python 3.10+, **JDK 17 or 21** (Spark 4 does not run on JDK 23+; the code auto-selects a JDK 17/21
from `PATH` if `JAVA_HOME` points at a newer one, or set `CHURN_JAVA_HOME`).

```bash
pip install -r requirements-pipeline.txt   # full Spark + boosters stack (dashboard-only deps live in dashboard/requirements.txt)
python -m churn.generate_data              # real Telco core -> 5 relational tables in data/raw/*.parquet
python -m churn.pipeline                   # Spark pipeline  -> outputs/scores.parquet + metrics.json
streamlit run dashboard/app.py             # dashboard at http://localhost:8501
```

The real dataset ships in the repo at `data/real/Telco-Customer-Churn.csv`, so `generate_data` runs offline.

Windows one-liner: `.\run_all.ps1` (add `-Tune` for grid-search + 3-fold CV, `-Scale 500000` to bootstrap to cluster-scale volume).

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

## Results (7,043 real customers, 26.5% churn, held-out test split)

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Lift @ top 10% |
|---|---|---|---|---|---|---|
| **Logistic Regression (MLlib)** | 0.865 | 0.692 | 0.616 | 0.710 | 0.660 | 2.75× |
| Random Forest (MLlib) | 0.858 | 0.670 | 0.618 | 0.710 | 0.661 | 2.65× |
| Gradient Boosted Trees (MLlib) | 0.852 | 0.664 | 0.625 | 0.661 | 0.642 | 2.58× |
| LightGBM | 0.865 | 0.690 | 0.599 | 0.765 | 0.672 | 2.88× |
| XGBoost | 0.867 | 0.692 | 0.617 | 0.736 | 0.672 | 2.85× |

The five models land within ~0.015 AUC of each other — the Telco signal is largely linear, so **Logistic Regression**
wins on *validation* AUC and becomes the production model (XGBoost edges it on the test split). Precision/recall/F1 use a
threshold tuned for max F1 on the validation split, not the test split, so the test numbers stay unbiased. Numbers vary
slightly with library versions.

## Dashboard tabs
A **guided tour** opens on first visit (reopen any time via the *🧭 Take the tour* button) and walks through each tab below.

- **Overview** – risk-band counts, probability histogram, predicted vs actual churn by contract / plan / region…
- **Customers** – filterable table of churn probability + plain-language risk factors, CSV export, single-customer lookup with every model's score
- **Model performance** – comparison table, ROC & PR curves, confusion matrix, cumulative-gains curve
- **Churn drivers** – feature importance per model, churn rate by complaints / late payments / tenure
- **Pipeline** – row counts, split sizes, run metadata

## About the data
The customer core is **real**: [`data/real/Telco-Customer-Churn.csv`](data/real/Telco-Customer-Churn.csv) is the public
IBM / Kaggle *Telco Customer Churn* dataset — 7,043 real customers with their real tenure, contract, internet service,
payment method, monthly charges, senior-citizen flag, add-on subscriptions and **actual churn outcome** (26.5% churned).

No public dataset ships the behavioural *history* a churn model needs, so [churn/generate_data.py](churn/generate_data.py)
reconstructs the five relational tables (customers, 6-month usage, payments, tickets, service_calls) for each real customer.
A latent "dissatisfaction" — **correlated with that customer's real churn outcome** and their real contract/tenure/charges —
drives fading usage, late payments, complaints and unresolved calls, so the reconstructed behaviour lines up with people who
really left. The model never sees the label or the latent; it recovers the signal from the engineered behaviour, which keeps
results realistic (AUC ~0.86) rather than trivially perfect.

**To run on a fully real behavioural dataset**, produce the same five tables (column names are in the generator) and point
`CHURN_RAW_URI` at them — nothing downstream changes. `--scale N` bootstrap-samples the real customers up to `N` rows to
demonstrate cluster-scale volume.

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
python -m churn.generate_data --scale 5000000 --out staging/
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
dashboard/app.py  Streamlit app (reads outputs/) — professional UI + built-in guided tour
data/real/        real Telco Customer Churn CSV (committed)   data/raw/  reconstructed tables (git-ignored)
outputs/          scores.parquet + metrics.json (committed; the dashboard reads these)
```
