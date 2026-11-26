"""
Probability calibration and decision-threshold selection.

Both are fitted on OUT-OF-FOLD predictions from cross-validation on the
training set (every training patient is predicted by a model that did not see
it). The test set is never involved.

- Calibration: isotonic regression and Platt scaling (logistic) are both
  tried; the one with the lower cross-validated Brier score is kept.
- Threshold: the cut-off on the calibrated probability that maximises
  out-of-fold accuracy (ties are broken towards 0.5).
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold

from src import config


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


class Calibrator:
    """Maps a raw model probability to a calibrated probability."""

    def __init__(self, method: str):
        self.method = method

    def fit(self, p, y):
        if self.method == "isotonic":
            self.model_ = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)
        else:  # Platt scaling on the logit of the probability
            self.model_ = LogisticRegression(C=1e6).fit(_logit(p), y)
        return self

    def transform(self, p) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if self.method == "isotonic":
            return self.model_.predict(p)
        return self.model_.predict_proba(_logit(p))[:, 1]


def choose_calibrator(oof_prob, y) -> tuple[Calibrator, dict]:
    """Pick isotonic vs Platt by 5-fold CV Brier score on the OOF predictions."""
    oof_prob, y = np.asarray(oof_prob), np.asarray(y)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    scores = {}
    for method in ("isotonic", "platt"):
        cal = np.zeros_like(oof_prob)
        for tr, va in skf.split(oof_prob, y):
            cal[va] = Calibrator(method).fit(oof_prob[tr], y[tr]).transform(oof_prob[va])
        scores[method] = float(brier_score_loss(y, cal))
    scores["uncalibrated"] = float(brier_score_loss(y, oof_prob))
    best = min(("isotonic", "platt"), key=scores.get)
    return Calibrator(best).fit(oof_prob, y), scores


def choose_threshold(prob, y) -> float:
    """Threshold maximising accuracy on out-of-fold (calibrated) probabilities."""
    prob, y = np.asarray(prob), np.asarray(y)
    grid = np.round(np.arange(0.20, 0.801, 0.01), 2)
    accs = np.array([((prob >= t) == y).mean() for t in grid])
    best = np.flatnonzero(accs == accs.max())
    return float(grid[best[np.argmin(np.abs(grid[best] - 0.5))]])
