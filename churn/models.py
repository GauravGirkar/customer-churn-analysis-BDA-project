"""Step 3 - Modelling.

Distributed (Spark MLlib):  Logistic Regression, Random Forest, Gradient-Boosted Trees
Single-node comparison:      LightGBM, XGBoost  (trained on the same Spark-engineered features)

Every trainer returns {"proba": ndarray aligned with `pdf`, "train_seconds", "importance", "params"}.
"""
import time
import warnings

import numpy as np
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import functions as F

from churn.config import SEED
from churn.features import CATEGORICAL, NUMERIC

warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")  # newer LightGBM prefers eval_X/eval_y

SPARK_MODELS = ["logistic_regression", "random_forest", "gbt"]
BOOSTER_MODELS = ["lightgbm", "xgboost"]
DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression (MLlib)",
    "random_forest": "Random Forest (MLlib)",
    "gbt": "Gradient Boosted Trees (MLlib)",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
}


# ------------------------------------------------------------------ Spark MLlib
def _preprocessing_stages():
    idx = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in CATEGORICAL]
    ohe = OneHotEncoder(inputCols=[f"{c}_idx" for c in CATEGORICAL],
                        outputCols=[f"{c}_ohe" for c in CATEGORICAL], dropLast=False)
    asm = VectorAssembler(inputCols=NUMERIC + [f"{c}_ohe" for c in CATEGORICAL], outputCol="raw_features")
    return idx + [ohe, asm]


def _spark_estimator(name):
    """Return (preprocessing+classifier Pipeline, param grid for --tune)."""
    stages = _preprocessing_stages()
    if name == "logistic_regression":
        stages.append(StandardScaler(inputCol="raw_features", outputCol="features", withMean=False, withStd=True))
        clf = LogisticRegression(featuresCol="features", labelCol="label", maxIter=100, regParam=0.01)
        grid = ParamGridBuilder().addGrid(clf.regParam, [0.001, 0.01, 0.1]).addGrid(
            clf.elasticNetParam, [0.0, 0.5]).build()
    elif name == "random_forest":
        clf = RandomForestClassifier(featuresCol="raw_features", labelCol="label", numTrees=100,
                                     maxDepth=8, subsamplingRate=0.8, seed=SEED)
        grid = ParamGridBuilder().addGrid(clf.maxDepth, [6, 8]).addGrid(clf.numTrees, [60, 100]).build()
    elif name == "gbt":
        clf = GBTClassifier(featuresCol="raw_features", labelCol="label", maxIter=100, maxDepth=5,
                            stepSize=0.1, subsamplingRate=0.8, seed=SEED)
        grid = ParamGridBuilder().addGrid(clf.maxDepth, [3, 5]).addGrid(clf.maxIter, [60, 120]).build()
    else:
        raise ValueError(name)
    return Pipeline(stages=stages + [clf]), clf, grid


def _spark_feature_names(transformed_schema):
    attrs = transformed_schema["raw_features"].metadata["ml_attr"]["attrs"]
    flat = [a for group in attrs.values() for a in group]
    return [a["name"] for a in sorted(flat, key=lambda a: a["idx"])]


def _top_importance(names, values, k=15):
    values = np.abs(np.asarray(values, dtype=float))
    values = values / values.sum() if values.sum() else values
    order = np.argsort(values)[::-1][:k]
    return [{"feature": names[i], "importance": round(float(values[i]), 5)} for i in order]


def train_spark_model(name, sdf, pdf, tune=False):
    """Fit on split=='train', then score EVERY customer. Returns result dict."""
    sdf = sdf.withColumn("label", F.col("label").cast("double"))
    train = sdf.filter(F.col("split") == "train")
    pipeline, clf, grid = _spark_estimator(name)

    t0 = time.time()
    if tune:
        cv = CrossValidator(estimator=pipeline, estimatorParamMaps=grid, numFolds=3, parallelism=2, seed=SEED,
                            evaluator=BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC"))
        model = cv.fit(train).bestModel
    else:
        model = pipeline.fit(train)
    seconds = time.time() - t0

    scored = model.transform(sdf)
    names = _spark_feature_names(scored.schema)
    fitted = model.stages[-1]
    if name == "logistic_regression":
        imp = fitted.coefficients.toArray()
    else:
        imp = fitted.featureImportances.toArray()

    probs = (scored.select("customer_id", vector_to_array("probability")[1].alias("p")).toPandas()
             .set_index("customer_id")["p"])
    params = {k.name: v for k, v in fitted.extractParamMap().items()
              if k.name in ("regParam", "elasticNetParam", "maxIter", "maxDepth", "numTrees", "stepSize")}
    return {"proba": probs.reindex(pdf["customer_id"]).to_numpy(), "train_seconds": round(seconds, 1),
            "importance": _top_importance(names, imp), "params": params}


# ----------------------------------------------------------- single-node boosters
def _to_model_frame(pdf):
    X = pdf[NUMERIC + CATEGORICAL].copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype("category")
    return X


def train_booster(name, pdf):
    X, y = _to_model_frame(pdf), pdf["label"].to_numpy()
    tr, va = (pdf["split"] == "train").to_numpy(), (pdf["split"] == "val").to_numpy()

    t0 = time.time()
    if name == "lightgbm":
        import lightgbm as lgb
        model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.03, num_leaves=31, min_child_samples=40,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                   random_state=SEED, verbose=-1)
        model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric="auc",
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        best_iter = model.best_iteration_
    elif name == "xgboost":
        import xgboost as xgb
        model = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.03, max_depth=4, subsample=0.8,
                                  colsample_bytree=0.8, tree_method="hist", enable_categorical=True,
                                  eval_metric="auc", early_stopping_rounds=50, random_state=SEED)
        model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        best_iter = model.best_iteration
    else:
        raise ValueError(name)
    seconds = time.time() - t0

    return {"proba": model.predict_proba(X)[:, 1], "train_seconds": round(seconds, 1),
            "importance": _top_importance(list(X.columns), model.feature_importances_),
            "params": {"best_iteration": int(best_iter)}}
