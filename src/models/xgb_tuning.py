"""
XGBoost hyper-parameter tuning with Optuna.

The objective is the mean 5-fold cross-validated log-loss on the TRAINING set
(log-loss is smoother than accuracy, so the search is more stable). The class
imbalance strategy is itself a hyper-parameter:
    none / scale_pos_weight / SMOTE (SMOTE is applied to the training part of
    each fold only, never to the validation fold).
"""
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, log_loss

from src import config

optuna.logging.set_verbosity(optuna.logging.WARNING)

# One fold = (X_train, y_train, X_valid, y_valid)
Fold = tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]


def make_xgb(params: dict, y_train) -> xgb.XGBClassifier:
    """Build an XGBClassifier from tuned parameters (handles the imbalance option)."""
    p = {k: v for k, v in params.items() if k != "imbalance"}
    if params.get("imbalance") == "scale_pos_weight":
        y_train = np.asarray(y_train)
        p["scale_pos_weight"] = (y_train == 0).sum() / (y_train == 1).sum()
    return xgb.XGBClassifier(**p, eval_metric="logloss", random_state=config.SEED,
                             n_jobs=4, verbosity=0)


def fit_xgb(params: dict, X, y) -> xgb.XGBClassifier:
    """Fit on (X, y), applying SMOTE to these rows first if that option was chosen."""
    if params.get("imbalance") == "smote":
        X, y = SMOTE(random_state=config.SEED).fit_resample(X, y)
    return make_xgb(params, y).fit(X, y)


def cv_predict(params: dict, folds: list[Fold]) -> list[np.ndarray]:
    """Validation-fold probabilities for each fold."""
    return [fit_xgb(params, Xtr, ytr).predict_proba(Xva)[:, 1] for Xtr, ytr, Xva, _ in folds]


def suggest_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 20, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "imbalance": trial.suggest_categorical("imbalance", ["none", "scale_pos_weight", "smote"]),
    }


def tune_xgb(folds: list[Fold], n_trials: int = config.XGB_OPTUNA_TRIALS,
             name: str = "xgb") -> tuple[dict, optuna.Study]:
    """Run the Optuna search and return the best parameters and the study."""

    def objective(trial):
        params = suggest_params(trial)
        probs = cv_predict(params, folds)
        losses = [log_loss(f[3], p) for f, p in zip(folds, probs)]
        accs = [accuracy_score(f[3], p >= 0.5) for f, p in zip(folds, probs)]
        trial.set_user_attr("cv_accuracy", float(np.mean(accs)))
        return float(np.mean(losses))

    study = optuna.create_study(direction="minimize", study_name=name,
                                sampler=optuna.samplers.TPESampler(seed=config.SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_trial
    print(f"  [{name}] best CV log-loss={best.value:.4f}  "
          f"CV acc={best.user_attrs['cv_accuracy']:.4f}  ({n_trials} trials)")
    return dict(best.params), study
