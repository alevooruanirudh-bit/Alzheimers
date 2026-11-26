"""
A single, uniform wrapper around every trained model.

Whatever the model type (sklearn baseline, tuned XGBoost, FT-Transformer or the
hybrid), a ModelBundle takes RAW patient features and returns probabilities,
so evaluate.py, explain.py and the web app all use the same code path:

    raw features -> [FeatureEngineer] -> model -> raw probability
                 -> [Calibrator] -> calibrated probability -> threshold -> 0/1

PyTorch models are saved as separate .pt files; the bundle stores only the file
name and loads the network lazily, so bundles can be pickled with joblib.
"""
import joblib
import numpy as np
import pandas as pd

from src import config
from src.models.transformer import TransformerClassifier


def hybrid_matrix(X_feat: pd.DataFrame, emb: np.ndarray, tx_prob: np.ndarray) -> pd.DataFrame:
    """Hybrid input = engineered features + [CLS] embedding + Transformer probability."""
    emb_df = pd.DataFrame(emb, columns=[f"tx_emb_{i}" for i in range(emb.shape[1])])
    out = pd.concat([X_feat.reset_index(drop=True), emb_df], axis=1)
    out["tx_prob"] = tx_prob
    return out


class ModelBundle:
    def __init__(self, name: str, kind: str, model=None, fe=None, calibrator=None,
                 threshold: float = 0.5, transformer_file: str | None = None, info: dict | None = None):
        self.name = name
        self.kind = kind            # "sklearn" | "transformer" | "hybrid"
        self.model = model          # sklearn / XGBoost estimator (None for "transformer")
        self.fe = fe                # FeatureEngineer or None (raw features)
        self.calibrator = calibrator
        self.threshold = threshold
        self.transformer_file = transformer_file
        self.info = info or {}
        self._transformer = None

    # -- lazy loading of the PyTorch part -------------------------------------
    @property
    def transformer(self) -> TransformerClassifier:
        if self._transformer is None:
            self._transformer = TransformerClassifier.load(config.MODELS_DIR / self.transformer_file)
        return self._transformer

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_transformer"] = None  # never pickle the network into the joblib file
        return state

    # -- inference -----------------------------------------------------------
    def features(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        """The exact matrix that the final estimator sees."""
        X = X_raw[config.RAW_FEATURES]
        Xf = self.fe.transform(X) if self.fe is not None else X
        if self.kind == "hybrid":
            t = self.transformer
            return hybrid_matrix(Xf, t.embed(Xf), t.predict_proba(Xf))
        return Xf

    def predict_proba_raw(self, X_raw: pd.DataFrame) -> np.ndarray:
        Xf = self.features(X_raw)
        if self.kind == "transformer":
            return self.transformer.predict_proba(Xf)
        return self.model.predict_proba(Xf)[:, 1]

    def predict_proba(self, X_raw: pd.DataFrame) -> np.ndarray:
        p = self.predict_proba_raw(X_raw)
        return self.calibrator.transform(p) if self.calibrator is not None else p

    def predict(self, X_raw: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X_raw) >= self.threshold).astype(int)

    # -- persistence -----------------------------------------------------------
    def save(self, filename: str) -> None:
        joblib.dump(self, config.MODELS_DIR / filename)

    @staticmethod
    def load(filename: str) -> "ModelBundle":
        return joblib.load(config.MODELS_DIR / filename)
