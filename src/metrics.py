"""Metric helpers shared by training (CV) and evaluation (test set)."""
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def binary_metrics(y_true, prob, threshold: float = 0.5) -> dict:
    """Accuracy, precision, recall, F1, ROC-AUC and the confusion matrix."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "threshold": float(threshold),
    }


def cv_summary(y, fold_probs: list, folds: list, threshold: float = 0.5) -> dict:
    """Mean and standard deviation of every metric across the CV folds."""
    y = np.asarray(y)
    per_fold = [binary_metrics(y[va], p, threshold) for (_, va), p in zip(folds, fold_probs)]
    out = {}
    for m in METRIC_NAMES:
        vals = [f[m] for f in per_fold]
        out[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "folds": vals}
    return out
