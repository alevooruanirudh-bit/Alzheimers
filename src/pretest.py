"""
Pre-test (home screening) evaluation:  python -m src.pretest

Question: how well does the screening work BEFORE any clinical test is done,
i.e. without blood tests, possibly without blood pressure, and with memory /
daily-life scores estimated at home instead of measured at a clinic?

Scenarios (all scored on the held-out TEST set; nothing is fitted on it):
  A  All inputs (reference, clinic-measured scores)
  B  No blood tests             -> cholesterol/triglyceride values replaced by training medians
  C  No health check numbers    -> blood tests AND blood pressure replaced by training medians
  D  C + error in home-estimated scores: Gaussian noise added to MMSE / functional / ADL
     (two error sizes, 20 repetitions each, fixed seed) to show how sensitive the
     result is to imperfect home estimates
  E  No memory or daily-life information at all: a separate XGBoost trained (on
     the training set, Optuna + 5-fold CV) only on demographics, lifestyle,
     health history and symptoms

Train the scenario-E model:  part of `python -m src.train` (or this module)
Evaluate all scenarios:      part of `python -m src.evaluate` (or this module)
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.data import cv_folds, get_test, get_train
from src.metrics import binary_metrics, cv_summary
from src.models.bundle import ModelBundle
from src.models.calibration import choose_calibrator, choose_threshold
from src.models.xgb_tuning import cv_predict, fit_xgb, tune_xgb

BLOOD_TESTS = ["CholesterolTotal", "CholesterolLDL", "CholesterolHDL", "CholesterolTriglycerides"]
BLOOD_PRESSURE = ["SystolicBP", "DiastolicBP"]
SCORES = {"MMSE": (0, 30), "FunctionalAssessment": (0, 10), "ADL": (0, 10)}
# Scenario E inputs: everything a person can answer without any test or score
NO_SCORE_FEATURES = [f for f in config.RAW_FEATURES if f not in BLOOD_TESTS + BLOOD_PRESSURE + list(SCORES)]
NOISE_LEVELS = {"small error (MMSE +/-2, other scores +/-1)": (2.0, 1.0),
                "large error (MMSE +/-4, other scores +/-2)": (4.0, 2.0)}
N_REPEATS = 20


def _no_score_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Scenario-E features: raw lay inputs plus the symptom count (a formula, nothing fitted)."""
    out = X[NO_SCORE_FEATURES].astype(float).copy()
    out["SymptomCount"] = X[config.SYMPTOM_FEATURES].sum(axis=1)
    return out


def train_no_score_model(n_trials: int = config.XGB_OPTUNA_TRIALS) -> dict:
    """Tune and fit the scenario-E model on the TRAINING set only."""
    X, y_s = get_train()
    y = y_s.values
    folds = cv_folds(y_s)
    Xn = _no_score_matrix(X)
    fdat = [(Xn.iloc[tr], y[tr], Xn.iloc[va], y[va]) for tr, va in folds]
    best, _ = tune_xgb(fdat, n_trials, "no_scores")
    probs = cv_predict(best, fdat)
    oof = np.zeros(len(y))
    for (_, va), p in zip(folds, probs):
        oof[va] = p
    cal, _ = choose_calibrator(oof, y)
    thr = choose_threshold(cal.transform(oof), y)
    model = fit_xgb(best, Xn, y)
    joblib_path = "pretest_no_scores.joblib"
    ModelBundle("XGBoost without memory/daily-life scores", "sklearn", model=model, calibrator=cal,
                threshold=thr, info={"best_params": best, "features": list(Xn.columns)}).save(joblib_path)
    summary = {"file": joblib_path, "features": list(Xn.columns), "best_params": best,
               "threshold": thr, "calibration": cal.method, "cv": cv_summary(y, probs, folds)}
    (config.REPORTS_DIR / "pretest_training.json").write_text(json.dumps(summary, indent=2))
    print(f"  [no_scores] CV acc {summary['cv']['accuracy']['mean']:.4f} +/- {summary['cv']['accuracy']['std']:.4f}")
    return summary


def _impute(X: pd.DataFrame, cols: list[str], medians: pd.Series) -> pd.DataFrame:
    X = X.copy()
    for c in cols:
        X[c] = medians[c]
    return X


def evaluate() -> dict:
    from src.evaluate import bootstrap_ci  # local import avoids a circular import

    X_train, _ = get_train()
    medians = X_train[config.RAW_FEATURES].median()  # training data only
    X_test, y_test = get_test()
    y = y_test.values
    hybrid = ModelBundle.load("hybrid.joblib")

    def score(name, prob, threshold, note):
        m = binary_metrics(y, prob, threshold)
        m["accuracy_95ci"] = bootstrap_ci(y, (np.asarray(prob) >= threshold).astype(int))
        m["note"] = note
        print(f"  {name:58s} acc {m['accuracy']:.4f}  recall {m['recall']:.4f}  auc {m['roc_auc']:.4f}")
        return m

    print("Pre-test scenarios (test set, proposed hybrid unless stated):")
    res = {}
    res["A. All inputs (clinic scores)"] = score(
        "A. All inputs (clinic scores)", hybrid.predict_proba(X_test), hybrid.threshold, "Reference.")
    X_b = _impute(X_test, BLOOD_TESTS, medians)
    res["B. No blood tests"] = score(
        "B. No blood tests", hybrid.predict_proba(X_b), hybrid.threshold,
        "Cholesterol and triglycerides replaced by training medians.")
    X_c = _impute(X_test, BLOOD_TESTS + BLOOD_PRESSURE, medians)
    p_c = hybrid.predict_proba(X_c)
    res["C. No blood tests, no blood pressure"] = score(
        "C. No blood tests, no blood pressure", p_c, hybrid.threshold,
        "Blood tests and blood pressure replaced by training medians.")

    rng = np.random.default_rng(config.SEED)
    for label, (sd_mmse, sd_other) in NOISE_LEVELS.items():
        accs, recalls, aucs = [], [], []
        for _ in range(N_REPEATS):
            Xn = X_c.copy()
            for col, (lo, hi) in SCORES.items():
                sd = sd_mmse if col == "MMSE" else sd_other
                Xn[col] = np.clip(Xn[col] + rng.normal(0, sd, len(Xn)), lo, hi)
            m = binary_metrics(y, hybrid.predict_proba(Xn), hybrid.threshold)
            accs.append(m["accuracy"]), recalls.append(m["recall"]), aucs.append(m["roc_auc"])
        key = f"D. C + {label}"
        res[key] = {"accuracy": float(np.mean(accs)), "accuracy_sd": float(np.std(accs)),
                    "recall": float(np.mean(recalls)), "roc_auc": float(np.mean(aucs)),
                    "note": f"Mean of {N_REPEATS} repetitions with random error added to the three scores."}
        print(f"  {key:58s} acc {np.mean(accs):.4f} +/- {np.std(accs):.4f}")

    no_score = ModelBundle.load("pretest_no_scores.joblib")
    Xe = _no_score_matrix(X_test)
    res["E. No memory/daily-life information"] = score(
        "E. No memory/daily-life information (separate model)", no_score.model.predict_proba(Xe)[:, 1]
        if no_score.calibrator is None else no_score.calibrator.transform(no_score.model.predict_proba(Xe)[:, 1]),
        no_score.threshold, "Separate XGBoost using only demographics, lifestyle, health history and symptoms.")
    res["E. No memory/daily-life information"]["cv"] = json.loads(
        (config.REPORTS_DIR / "pretest_training.json").read_text())["cv"]

    out = {"n_test": int(len(y)), "scenarios": res,
           "imputation": "training-set medians", "blood_tests": BLOOD_TESTS, "blood_pressure": BLOOD_PRESSURE}
    (config.REPORTS_DIR / "pretest_results.json").write_text(json.dumps(out, indent=2))

    # figure
    names = list(res)
    acc = [res[n]["accuracy"] for n in names]
    err = [res[n].get("accuracy_sd", 0) for n in names]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["#0b8396", "#0e9fb3", "#14b8a6", "#f59e0b", "#d97706", "#94a3b8"]
    ax.barh(names[::-1], acc[::-1], xerr=err[::-1], color=colors[:len(names)][::-1])
    for i, v in enumerate(acc[::-1]):
        ax.text(v + 0.003, i, f"{v:.1%}", va="center", fontsize=9)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("Test accuracy")
    ax.set_title("Screening accuracy before clinical tests (test set, 430 patients)")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "pretest_scenarios.png", dpi=140)
    plt.close(fig)
    return out


if __name__ == "__main__":
    import torch

    torch.set_num_threads(1)
    train_no_score_model()
    evaluate()
