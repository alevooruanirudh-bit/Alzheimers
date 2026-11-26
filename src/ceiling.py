"""
Label-noise / accuracy-ceiling analysis (training data only).

EDA on the training set showed that the diagnosis label closely follows a
simple clinical rule:

    Alzheimer's  <=>  MMSE < 24  AND  at least 2 of
                      {FunctionalAssessment < 5, ADL < 5, MemoryComplaints, BehavioralProblems}

This script measures (on TRAINING data only):
  1. how much of the label the rule explains, and
  2. whether the patients that break the rule can be predicted from ANY of the
     features (cross-validated AUC). An AUC near 0.5 means those labels behave
     like random noise, which puts a hard ceiling on achievable accuracy.
"""
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src import config
from src.data import get_train


def clinical_rule(X: pd.DataFrame) -> np.ndarray:
    domains = ((X["FunctionalAssessment"] < 5).astype(int) + (X["ADL"] < 5).astype(int)
               + X["MemoryComplaints"] + X["BehavioralProblems"])
    return ((X["MMSE"] < 24) & (domains >= 2)).astype(int).values


def analyse() -> dict:
    X, y = get_train()
    y = y.values
    rule = clinical_rule(X)
    exceptions = (rule != y).astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    model = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                              random_state=config.SEED, n_jobs=4, verbosity=0)
    p = cross_val_predict(model, X.assign(rule=rule), exceptions, cv=cv, method="predict_proba")[:, 1]
    rate = float(exceptions.mean())
    n_test = round(len(y) / (1 - config.TEST_SIZE) * config.TEST_SIZE)
    # If ~rate of labels are unpredictable, test accuracy ~ Binomial around (1 - rate)
    sd = float(np.sqrt(rate * (1 - rate) / n_test))
    return {
        "rule": "MMSE < 24 AND >= 2 of {FA < 5, ADL < 5, MemoryComplaints, BehavioralProblems}",
        "rule_accuracy_train": float((rule == y).mean()),
        "n_exceptions_train": int(exceptions.sum()),
        "exception_rate_train": rate,
        "exception_predictability_cv_auc": float(roc_auc_score(exceptions, p)),
        "expected_accuracy_ceiling": 1 - rate,
        "test_accuracy_sd_at_ceiling": sd,
        "z_needed_for_98_1": float((0.981 - (1 - rate)) / sd),
    }


if __name__ == "__main__":
    res = analyse()
    (config.REPORTS_DIR / "ceiling_analysis.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
