"""
Train every model:  python -m src.train   (add --fast for a quick smoke test)

IMPORTANT: this script only ever loads the TRAINING split. All tuning,
feature selection, calibration and threshold choice use 5-fold stratified
cross-validation on the training data. The test set is used exactly once,
later, by `python -m src.evaluate`.

Models produced (saved to models/):
  baseline_*.joblib          Logistic Regression, SVM, Random Forest, plain XGBoost
  xgb_tuned.joblib           Optuna-tuned XGBoost on engineered features
  transformer.joblib/.pt     FT-Transformer on engineered features
  hybrid.joblib              Proposed hybrid (Transformer embeddings + XGBoost)
  transformer_raw.pt,
  hybrid_nofe.joblib         Ablation: hybrid without feature engineering
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.metrics import log_loss

from src import config
from src.data import cv_folds, get_train
from src.features import FeatureEngineer
from src.metrics import cv_summary
from src.models import hybrid as H
from src.models.baselines import BASELINES
from src.models.bundle import ModelBundle
from src.models.calibration import choose_calibrator, choose_threshold
from src.models.xgb_tuning import cv_predict, fit_xgb, tune_xgb

torch.set_num_threads(4)


def oof_from_folds(n: int, folds, fold_probs) -> np.ndarray:
    """Stitch per-fold validation predictions into one out-of-fold vector."""
    oof = np.zeros(n)
    for (_, va), p in zip(folds, fold_probs):
        oof[va] = p
    return oof


def calibrate(name: str, oof_prob, y) -> tuple:
    """Fit calibrator + threshold on out-of-fold predictions (training data only)."""
    cal, scores = choose_calibrator(oof_prob, y)
    thr = choose_threshold(cal.transform(oof_prob), y)
    print(f"  [{name}] calibration={cal.method} (CV Brier {scores}), threshold={thr:.2f}")
    return cal, thr, scores


def fe_folds(X, y, folds, fe_kwargs) -> tuple[list, list]:
    """Per-fold feature engineering: fitted on the fold's training part only."""
    out, fes = [], []
    for tr, va in folds:
        fe = FeatureEngineer(**fe_kwargs).fit(X.iloc[tr], y[tr])
        out.append((fe.transform(X.iloc[tr]), y[tr], fe.transform(X.iloc[va]), y[va]))
        fes.append(fe)
    return out, fes


def main(fast: bool = False) -> None:
    t0 = time.time()
    xgb_trials = 5 if fast else config.XGB_OPTUNA_TRIALS
    tx_trials = 2 if fast else config.TRANSFORMER_OPTUNA_TRIALS
    if fast:
        config.TRANSFORMER_MAX_EPOCHS = 30

    X, y_series = get_train()
    y = y_series.values
    folds = cv_folds(y_series)
    n = len(y)
    summary = {"n_train": int(n), "n_folds": config.N_FOLDS, "seed": config.SEED,
               "fast_mode": fast, "models": {}}
    print(f"Training patients: {n}  (positive rate {y.mean():.3f})")

    # ------------------------------------------------------------------ 1
    print("\n[1/6] Baselines on raw features")
    for name, factory in BASELINES.items():
        probs = []
        for tr, va in folds:
            m = clone(factory()).fit(X.iloc[tr], y[tr])
            probs.append(m.predict_proba(X.iloc[va])[:, 1])
        final = factory().fit(X, y)
        fname = "baseline_" + name.split()[0].lower() + ".joblib"
        info = {"best_params": getattr(final, "best_params_", None)}
        ModelBundle(name, "sklearn", model=final, threshold=0.5, info=info).save(fname)
        summary["models"][name] = {"file": fname, "group": "baseline",
                                   "cv": cv_summary(y, probs, folds), **info}
        print(f"  {name:22s} CV acc {summary['models'][name]['cv']['accuracy']['mean']:.4f}")

    # ------------------------------------------------------------------ 2
    print("\n[2/6] Feature set comparison (full engineered vs selected)")
    ref = {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.9,
           "colsample_bytree": 0.9, "imbalance": "none"}
    fs_compare = {}
    for label, kw in [("raw", {"engineer": False}), ("engineered_full", {"engineer": True}),
                      ("engineered_selected", {"engineer": True, "select": True})]:
        fdat, fes = fe_folds(X, y, folds, kw)
        probs = cv_predict(ref, fdat)
        fs_compare[label] = {"kwargs": kw, "cv": cv_summary(y, probs, folds),
                             "cv_logloss": float(np.mean([log_loss(f[3], p) for f, p in zip(fdat, probs)])),
                             "n_features_per_fold": [len(f.selected_features_) for f in fes]}
        print(f"  {label:20s} CV acc {fs_compare[label]['cv']['accuracy']['mean']:.4f}  "
              f"logloss {fs_compare[label]['cv_logloss']:.4f}  n_feat {fs_compare[label]['n_features_per_fold']}")
    # pick full vs selected by CV accuracy, then log-loss; ties favour the smaller set
    cands = ["engineered_selected", "engineered_full"]
    chosen = max(cands, key=lambda k: (round(fs_compare[k]["cv"]["accuracy"]["mean"], 4),
                                       -round(fs_compare[k]["cv_logloss"], 4)))
    fe_kwargs = fs_compare[chosen]["kwargs"]
    summary["feature_selection"] = {"comparison": fs_compare, "chosen": chosen}
    print(f"  -> using '{chosen}'")

    # ------------------------------------------------------------------ 3
    print("\n[3/6] Tuned XGBoost on engineered features (Optuna)")
    fdat, _ = fe_folds(X, y, folds, fe_kwargs)
    best_xgb, study = tune_xgb(fdat, xgb_trials, "xgb_tuned")
    probs = cv_predict(best_xgb, fdat)
    oof = oof_from_folds(n, folds, probs)
    cal, thr, cal_scores = calibrate("xgb_tuned", oof, y)
    fe_final = FeatureEngineer(**fe_kwargs).fit(X, y)
    xgb_final = fit_xgb(best_xgb, fe_final.transform(X), y)
    ModelBundle("XGBoost (tuned)", "sklearn", model=xgb_final, fe=fe_final, calibrator=cal,
                threshold=thr, info={"best_params": best_xgb}).save("xgb_tuned.joblib")
    summary["models"]["XGBoost (tuned)"] = {
        "file": "xgb_tuned.joblib", "group": "proposed-components", "cv": cv_summary(y, probs, folds),
        "best_params": best_xgb, "calibration": cal.method, "calibration_brier": cal_scores,
        "threshold": thr, "optuna_trials": [t.value for t in study.trials],
        "features": fe_final.selected_features_, "interactions": fe_final.interactions_}

    # ------------------------------------------------------------------ 4
    print("\n[4/6] FT-Transformer: Optuna tuning + out-of-fold training")
    tx_params, tx_study = H.tune_transformer(X, y_series, fe_kwargs, tx_trials)
    print(f"  best params: {tx_params}")
    oof_tx = H.run_transformer_oof(X, y_series, folds, fe_kwargs, tx_params)
    oof_tx["transformer_full"].save(config.MODELS_DIR / "transformer.pt")
    tx_fold_probs = [oof_tx["oof_prob"][va] for _, va in folds]
    cal, thr, cal_scores = calibrate("transformer", oof_tx["oof_prob"], y)
    ModelBundle("FT-Transformer", "transformer", fe=oof_tx["fe_full"], calibrator=cal, threshold=thr,
                transformer_file="transformer.pt", info={"params": tx_params}).save("transformer.joblib")
    summary["models"]["FT-Transformer"] = {
        "file": "transformer.joblib", "group": "proposed-components",
        "cv": cv_summary(y, tx_fold_probs, folds), "params": tx_params, "calibration": cal.method,
        "calibration_brier": cal_scores, "threshold": thr,
        "optuna_trials": [t.value for t in tx_study.trials],
        "history_full": oof_tx["transformer_full"].history,
        "best_epoch_full": oof_tx["transformer_full"].best_epoch_,
        "fold_histories": oof_tx["fold_histories"], "alignment_r2": oof_tx["alignment_r2"]}

    # ------------------------------------------------------------------ 5
    print("\n[5/6] Proposed hybrid: Transformer embeddings + tuned XGBoost")
    hfolds = H.hybrid_folds(X, y_series, folds, oof_tx)
    best_h, h_study = tune_xgb(hfolds, xgb_trials, "hybrid")
    probs = cv_predict(best_h, hfolds)
    oof = oof_from_folds(n, folds, probs)
    cal, thr, cal_scores = calibrate("hybrid", oof, y)
    Xh = H.hybrid_train_matrix(X, oof_tx)
    xgb_h = fit_xgb(best_h, Xh, y)
    ModelBundle("Hybrid Transformer + XGBoost", "hybrid", model=xgb_h, fe=oof_tx["fe_full"],
                calibrator=cal, threshold=thr, transformer_file="transformer.pt",
                info={"best_params": best_h, "feature_names": list(Xh.columns),
                      "model_version": config.MODEL_VERSION}).save("hybrid.joblib")
    summary["models"]["Hybrid Transformer + XGBoost"] = {
        "file": "hybrid.joblib", "group": "proposed", "cv": cv_summary(y, probs, folds),
        "best_params": best_h, "calibration": cal.method, "calibration_brier": cal_scores,
        "threshold": thr, "optuna_trials": [t.value for t in h_study.trials],
        "n_input_features": int(Xh.shape[1])}

    # ------------------------------------------------------------------ 6
    print("\n[6/6] Ablation: hybrid WITHOUT feature engineering")
    raw_kwargs = {"engineer": False}
    oof_raw = H.run_transformer_oof(X, y_series, folds, raw_kwargs, tx_params)
    oof_raw["transformer_full"].save(config.MODELS_DIR / "transformer_raw.pt")
    cal_r, thr_r, _ = calibrate("transformer_raw", oof_raw["oof_prob"], y)
    ModelBundle("FT-Transformer (no FE)", "transformer", fe=oof_raw["fe_full"], calibrator=cal_r,
                threshold=thr_r, transformer_file="transformer_raw.pt").save("transformer_raw.joblib")
    summary["models"]["FT-Transformer (no FE)"] = {
        "file": "transformer_raw.joblib", "group": "ablation", "threshold": thr_r,
        "cv": cv_summary(y, [oof_raw["oof_prob"][va] for _, va in folds], folds)}
    hfolds_raw = H.hybrid_folds(X, y_series, folds, oof_raw)
    best_hr, _ = tune_xgb(hfolds_raw, xgb_trials, "hybrid_nofe")
    probs = cv_predict(best_hr, hfolds_raw)
    cal, thr, _ = calibrate("hybrid_nofe", oof_from_folds(n, folds, probs), y)
    xgb_hr = fit_xgb(best_hr, H.hybrid_train_matrix(X, oof_raw), y)
    ModelBundle("Hybrid without feature engineering", "hybrid", model=xgb_hr, fe=oof_raw["fe_full"],
                calibrator=cal, threshold=thr, transformer_file="transformer_raw.pt",
                info={"best_params": best_hr}).save("hybrid_nofe.joblib")
    summary["models"]["Hybrid without feature engineering"] = {
        "file": "hybrid_nofe.joblib", "group": "ablation", "cv": cv_summary(y, probs, folds),
        "best_params": best_hr, "threshold": thr}

    print("\n[extra] Pre-test model without memory/daily-life scores (src/pretest.py)")
    from src import pretest
    pretest.train_no_score_model(xgb_trials)

    summary["training_minutes"] = round((time.time() - t0) / 60, 1)
    (config.REPORTS_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nDone in {summary['training_minutes']} min. CV summary:")
    rows = [(k, v["cv"]["accuracy"]["mean"], v["cv"]["accuracy"]["std"]) for k, v in summary["models"].items()]
    print(pd.DataFrame(rows, columns=["model", "cv_acc_mean", "cv_acc_std"]).round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="tiny budgets for a smoke test")
    main(ap.parse_args().fast)
