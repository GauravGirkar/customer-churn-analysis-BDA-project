"""Step 4 - Evaluation: ROC-AUC, PR-AUC, precision / recall / F1, confusion matrix, top-decile lift.
The decision threshold is tuned on the validation split (max F1) and reported on the untouched test split."""
import numpy as np
from sklearn.metrics import (average_precision_score, precision_recall_curve, roc_auc_score, roc_curve)


def best_f1_threshold(y_true, p):
    prec, rec, thr = precision_recall_curve(y_true, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.argmax(f1))])


def threshold_metrics(y_true, p, threshold):
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": round(float(threshold), 4), "precision": precision, "recall": recall, "f1": f1,
            "accuracy": (tp + tn) / len(y_true), "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}}


def top_decile_lift(y_true, p):
    k = max(1, len(p) // 10)
    top = np.argsort(p)[::-1][:k]
    return float(y_true[top].mean() / y_true.mean())


def _downsample(x, y, n=120):
    idx = np.unique(np.linspace(0, len(x) - 1, n).astype(int))
    return [[round(float(x[i]), 4), round(float(y[i]), 4)] for i in idx]


def evaluate_model(y_val, p_val, y_test, p_test):
    thr = best_f1_threshold(y_val, p_val)
    fpr, tpr, _ = roc_curve(y_test, p_test)
    prec, rec, _ = precision_recall_curve(y_test, p_test)
    return {
        "roc_auc": float(roc_auc_score(y_test, p_test)),
        "pr_auc": float(average_precision_score(y_test, p_test)),
        "top_decile_lift": top_decile_lift(y_test, p_test),
        "at_default_0.5": threshold_metrics(y_test, p_test, 0.5),
        "at_tuned": threshold_metrics(y_test, p_test, thr),
        "roc_curve": _downsample(fpr, tpr),
        "pr_curve": _downsample(rec, prec),
    }
