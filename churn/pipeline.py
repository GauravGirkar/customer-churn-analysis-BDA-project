"""End-to-end pipeline: ingest -> Spark SQL features -> train 5 models -> evaluate -> export for dashboard.

    python -m churn.pipeline                 # all models
    python -m churn.pipeline --tune          # 3-fold CV grid search on the MLlib models
    python -m churn.pipeline --models logistic_regression random_forest
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone

import numpy as np

from churn.config import OUTPUT_DIR, RAW_URI, SNAPSHOT_DATE
from churn.evaluate import best_f1_threshold, evaluate_model
from churn.features import CATEGORICAL, NUMERIC, build_features
from churn.ingest import load_raw_tables
from churn.models import BOOSTER_MODELS, DISPLAY_NAMES, SPARK_MODELS, train_booster, train_spark_model
from churn.spark_session import get_spark

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))  # silences a joblib warning on Windows

HIGH, MEDIUM = 0.50, 0.25  # risk-band cut-offs on churn probability


def risk_factors(r):
    """Plain-language drivers for a customer row (rule-based, complements the model score)."""
    out = []
    if r.contract == "Month-to-month":
        out.append("Month-to-month contract")
    if r.late_payments_recent >= 2:
        out.append(f"{int(r.late_payments_recent)} late payments in last 3 months")
    if r.minutes_trend < 0.75 or r.logins_trend < 0.75:
        out.append("Usage declining")
    if r.cancellation_inquiries >= 1:
        out.append("Asked about cancelling")
    if r.complaint_count >= 2:
        out.append(f"{int(r.complaint_count)} complaints")
    if r.unresolved_calls >= 2:
        out.append(f"{int(r.unresolved_calls)} unresolved calls")
    if r.avg_satisfaction <= 2.5:
        out.append("Low satisfaction score")
    if r.payment_method == "Electronic check":
        out.append("Pays by electronic check")
    return "; ".join(out[:4]) if out else "No standout risk signals"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=RAW_URI, help="folder / hdfs:// / s3a:// URI holding the raw tables")
    ap.add_argument("--models", nargs="+", default=SPARK_MODELS + BOOSTER_MODELS,
                    choices=SPARK_MODELS + BOOSTER_MODELS)
    ap.add_argument("--tune", action="store_true", help="grid-search + 3-fold CV for the MLlib models (slower)")
    args = ap.parse_args()

    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    spark = get_spark()

    print("[1/4] Ingesting raw tables")
    counts = load_raw_tables(spark, args.raw)

    print("[2/4] Engineering features with Spark SQL")
    sdf = build_features(spark)
    pdf = sdf.toPandas().sort_values("customer_id").reset_index(drop=True)
    pdf["label"] = pdf["label"].astype(int)
    pdf[NUMERIC] = pdf[NUMERIC].astype(float)  # Spark DECIMAL columns arrive as Python objects
    print(f"  {len(pdf):,} customers x {len(NUMERIC) + len(CATEGORICAL)} features | churn rate {pdf.label.mean():.1%} | "
          f"split {pdf.split.value_counts().to_dict()}")

    print("[3/4] Training models")
    results = {}
    for name in args.models:
        print(f"  - {DISPLAY_NAMES[name]} ...", end=" ", flush=True)
        results[name] = (train_spark_model(name, sdf, pdf, args.tune) if name in SPARK_MODELS
                         else train_booster(name, pdf))
        print(f"{results[name]['train_seconds']}s")

    print("[4/4] Evaluating")
    y = pdf["label"].to_numpy()
    val, test = (pdf["split"] == "val").to_numpy(), (pdf["split"] == "test").to_numpy()
    models_out = {}
    for name, res in results.items():
        ev = evaluate_model(y[val], res["proba"][val], y[test], res["proba"][test])
        ev.update(display_name=DISPLAY_NAMES[name], engine="Spark MLlib" if name in SPARK_MODELS else "single-node",
                  train_seconds=res["train_seconds"], params=res["params"], importance=res["importance"])
        models_out[name] = ev
        t = ev["at_tuned"]
        print(f"  {DISPLAY_NAMES[name]:32s} AUC {ev['roc_auc']:.4f}  PR-AUC {ev['pr_auc']:.4f}  "
              f"P {t['precision']:.3f}  R {t['recall']:.3f}  F1 {t['f1']:.3f}  lift@10% {ev['top_decile_lift']:.2f}x")

    # Best model is picked on the VALIDATION split so the test split stays an honest estimate.
    from sklearn.metrics import roc_auc_score
    best = max(results, key=lambda n: roc_auc_score(y[val], results[n]["proba"][val]))
    best_thr = best_f1_threshold(y[val], results[best]["proba"][val])
    print(f"\n  Best model (validation AUC): {DISPLAY_NAMES[best]}  | tuned threshold {best_thr:.3f}")

    # ---- export: one row per customer for the dashboard
    scores = pdf.copy()
    scores["churn_probability"] = np.round(results[best]["proba"], 4)
    scores["risk_band"] = np.select([scores.churn_probability >= HIGH, scores.churn_probability >= MEDIUM],
                                    ["High", "Medium"], "Low")
    scores["predicted_churn"] = (scores.churn_probability >= best_thr).astype(int)
    scores["risk_factors"] = [risk_factors(r) for r in scores.itertuples()]
    scores = scores.rename(columns={"label": "actual_churn"})
    for name, res in results.items():
        scores[f"prob_{name}"] = np.round(res["proba"], 4)
    scores.to_parquet(OUTPUT_DIR / "scores.parquet", index=False)

    # Record a short, non-identifying label for the raw source: keep remote URIs
    # (hdfs://, s3a://, ...) as-is, but reduce a local filesystem path to its last
    # two components so the committed metrics never leak an absolute local path.
    if "://" in args.raw:
        raw_source = args.raw
    else:
        parts = os.path.normpath(args.raw).replace("\\", "/").split("/")
        raw_source = "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "snapshot_date": SNAPSHOT_DATE, "raw_source": raw_source, "tuned": args.tune,
        "table_rows": counts, "n_customers": int(len(pdf)), "churn_rate": float(y.mean()),
        "split_sizes": pdf["split"].value_counts().to_dict(), "n_features": len(NUMERIC) + len(CATEGORICAL),
        "best_model": best, "best_threshold": best_thr, "band_cutoffs": {"high": HIGH, "medium": MEDIUM},
        "models": models_out, "pipeline_seconds": round(time.time() - t_start, 1),
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nDone in {metrics['pipeline_seconds']}s -> {OUTPUT_DIR}\n"
          f"Launch the dashboard:  streamlit run dashboard/app.py")
    spark.stop()


if __name__ == "__main__":
    main()
