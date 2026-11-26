"""
Proposed hybrid: FT-Transformer embeddings + XGBoost.

Pipeline
--------
1. Tune the Transformer with Optuna (CV on the training set).
2. Train a "full" Transformer on the whole training set (a small inner
   validation split is used for early stopping only).
3. Out-of-fold (OOF) embeddings for the training patients: for each of the
   5 folds, train a Transformer on the other 4 folds and embed the held-out
   fold. So no training patient is embedded by a model that saw its label.
4. Alignment: every fold model learns its own embedding coordinate system.
   To make the columns mean the same thing for XGBoost, each fold's
   embeddings are mapped into the full model's coordinate system with an
   orthogonal Procrustes rotation (+ scale), fitted on that fold's TRAINING
   rows only.
5. XGBoost is tuned and trained on [engineered features + embedding +
   Transformer probability]. Test patients are embedded by the full model.
"""
import numpy as np
import optuna
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split

from src import config
from src.features import FeatureEngineer
from src.models.bundle import hybrid_matrix
from src.models.transformer import TransformerClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Transformer training helpers
# ---------------------------------------------------------------------------
def train_transformer(X_feat: pd.DataFrame, y, cat_cardinality: dict, params: dict) -> TransformerClassifier:
    """Train on X_feat with an inner stratified validation split for early stopping."""
    y = np.asarray(y)
    fit_idx, es_idx = train_test_split(np.arange(len(y)), test_size=config.TRANSFORMER_VAL_FRACTION,
                                       stratify=y, random_state=config.SEED)
    model = TransformerClassifier(cat_cardinality, **params)
    return model.fit(X_feat.iloc[fit_idx], y[fit_idx], X_feat.iloc[es_idx], y[es_idx])


def suggest_transformer_params(trial: optuna.Trial) -> dict:
    d = trial.suggest_categorical("d", [16, 32, 64])
    return {
        "d": d,
        "n_layers": trial.suggest_int("n_layers", 1, 4),
        "n_heads": trial.suggest_categorical("n_heads", [2, 4]),
        "ff_mult": trial.suggest_float("ff_mult", 1.0, 4.0),
        "attn_dropout": trial.suggest_float("attn_dropout", 0.0, 0.3),
        "ff_dropout": trial.suggest_float("ff_dropout", 0.0, 0.3),
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "class_weight": trial.suggest_categorical("class_weight", [False, True]),
    }


def tune_transformer(X: pd.DataFrame, y: pd.Series, fe_kwargs: dict,
                     n_trials: int = config.TRANSFORMER_OPTUNA_TRIALS) -> tuple[dict, optuna.Study]:
    """Optuna search, objective = mean validation log-loss over 3 training-set folds."""
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=config.TRANSFORMER_TUNING_FOLDS, shuffle=True, random_state=config.SEED)
    folds = []  # feature engineering is fitted once per fold and cached
    for tr, va in skf.split(X, y):
        fe = FeatureEngineer(**fe_kwargs).fit(X.iloc[tr], y[tr])
        folds.append((fe.transform(X.iloc[tr]), y[tr], fe.transform(X.iloc[va]), y[va],
                      fe.categorical_cardinality()))

    def objective(trial):
        params = suggest_transformer_params(trial)
        losses = []
        for Xtr, ytr, Xva, yva, card in folds:
            m = train_transformer(Xtr, ytr, card, params)
            losses.append(log_loss(yva, np.clip(m.predict_proba(Xva), 1e-7, 1 - 1e-7)))
        return float(np.mean(losses))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=config.SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  [transformer] best CV log-loss={study.best_value:.4f} ({n_trials} trials)")
    return dict(study.best_params), study


# ---------------------------------------------------------------------------
# Embedding alignment
# ---------------------------------------------------------------------------
class EmbeddingAligner:
    """Similarity transform (centre, rotate, scale) from one embedding space to another."""

    def fit(self, source: np.ndarray, target: np.ndarray):
        self.mu_s_, self.mu_t_ = source.mean(0), target.mean(0)
        A, B = source - self.mu_s_, target - self.mu_t_
        self.R_, sv_sum = orthogonal_procrustes(A, B)
        self.scale_ = sv_sum / (A ** 2).sum()
        self.r2_ = 1 - ((A @ self.R_ * self.scale_ - B) ** 2).sum() / (B ** 2).sum()
        return self

    def transform(self, E: np.ndarray) -> np.ndarray:
        return (E - self.mu_s_) @ self.R_ * self.scale_ + self.mu_t_


# ---------------------------------------------------------------------------
# Out-of-fold Transformer run
# ---------------------------------------------------------------------------
def run_transformer_oof(X: pd.DataFrame, y: pd.Series, folds, fe_kwargs: dict, params: dict) -> dict:
    """Train the full Transformer and the K fold Transformers; return OOF outputs."""
    y = np.asarray(y)
    fe_full = FeatureEngineer(**fe_kwargs).fit(X, y)
    X_full = fe_full.transform(X)
    t_full = train_transformer(X_full, y, fe_full.categorical_cardinality(), params)

    d = params["d"]
    oof_prob, oof_emb = np.zeros(len(y)), np.zeros((len(y), d))
    fold_fes, histories, align_r2 = [], [], []
    for k, (tr, va) in enumerate(folds):
        fe_k = FeatureEngineer(**fe_kwargs).fit(X.iloc[tr], y[tr])
        Xtr_k, Xva_k = fe_k.transform(X.iloc[tr]), fe_k.transform(X.iloc[va])
        t_k = train_transformer(Xtr_k, y[tr], fe_k.categorical_cardinality(), params)
        oof_prob[va] = t_k.predict_proba(Xva_k)
        # align fold-k embedding space to the full model's space using fold-k TRAINING rows
        aligner = EmbeddingAligner().fit(t_k.embed(Xtr_k), t_full.embed(X_full.iloc[tr]))
        oof_emb[va] = aligner.transform(t_k.embed(Xva_k))
        fold_fes.append(fe_k)
        histories.append(t_k.history)
        align_r2.append(float(aligner.r2_))
        print(f"    fold {k + 1}: best epoch {t_k.best_epoch_}, alignment R2={aligner.r2_:.3f}")

    return {"fe_full": fe_full, "transformer_full": t_full, "oof_prob": oof_prob,
            "oof_emb": oof_emb, "fold_fes": fold_fes, "fold_histories": histories,
            "alignment_r2": align_r2}


def hybrid_folds(X: pd.DataFrame, y: pd.Series, folds, oof: dict) -> list:
    """CV folds for tuning the hybrid XGBoost (fold-specific feature engineering)."""
    y = np.asarray(y)
    out = []
    for (tr, va), fe_k in zip(folds, oof["fold_fes"]):
        Xtr = hybrid_matrix(fe_k.transform(X.iloc[tr]), oof["oof_emb"][tr], oof["oof_prob"][tr])
        Xva = hybrid_matrix(fe_k.transform(X.iloc[va]), oof["oof_emb"][va], oof["oof_prob"][va])
        out.append((Xtr, y[tr], Xva, y[va]))
    return out


def hybrid_train_matrix(X: pd.DataFrame, oof: dict) -> pd.DataFrame:
    """Full training matrix for the final hybrid XGBoost."""
    return hybrid_matrix(oof["fe_full"].transform(X), oof["oof_emb"], oof["oof_prob"])
