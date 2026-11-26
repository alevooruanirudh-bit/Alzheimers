"""
Inference + explanation service used by the web app, the Digital Twin and the
report generator.

    RiskPredictor().predict(patient_dict)  ->  probability, risk level,
                                               plain-language SHAP factors,
                                               Transformer attention

SHAP values come from TreeExplainer on the final (hybrid) XGBoost. The hybrid
has one column per embedding dimension (16 in the tuned model) that mean nothing to a clinician, so their SHAP
values are summed into one "Transformer embedding" contribution (SHAP values
are additive, so summing a group is exact).
"""
import io
import threading

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src import config
from src.data import get_train
from src.features import label_for
from src.models.bundle import ModelBundle

EMB_GROUP = "Transformer embedding"
TXP_GROUP = "Transformer risk score"
BINARY_FLAG_TEXT = {  # engineered flags -> wording when present / absent
    "MMSE_Impaired": ("Memory and thinking score in the impaired range (below 24 of 30)",
                      "Memory and thinking score in the normal range (24 or more of 30)"),
    "FA_Low": ("Everyday tasks score below 5 of 10", "Everyday tasks score of 5 or more of 10"),
    "ADL_Low": ("Self-care score below 5 of 10", "Self-care score of 5 or more of 10"),
}
_PLOT_LOCK = threading.Lock()  # matplotlib is not thread-safe


def _lower_first(text: str) -> str:
    """'Self-care score (ADL)' -> 'self-care score (ADL)' (keeps abbreviations)."""
    return text[:1].lower() + text[1:] if not text[:2].isupper() else text
# engineered features computed from optional measurements
DERIVED_FROM = {"LDL_HDL_Ratio": ["CholesterolLDL", "CholesterolHDL"],
                "PulsePressure": ["SystolicBP", "DiastolicBP"],
                "CardioRiskCount": ["CholesterolTotal"]}


class RiskPredictor:
    def __init__(self):
        self.hybrid = ModelBundle.load("hybrid.joblib")
        self.plain = ModelBundle.load("baseline_xgboost.joblib")
        self.explainer = shap.TreeExplainer(self.hybrid.model)
        self.plain_explainer = shap.TreeExplainer(self.plain.model)
        self.model_version = self.hybrid.info.get("model_version", config.MODEL_VERSION)
        # training-set statistics: quartiles describe values as low / typical / high, and
        # medians fill in optional measurements the user did not provide (training data only)
        X_train, _ = get_train()
        Xf = self.hybrid.fe.transform(X_train)
        self.q25, self.q75 = Xf.quantile(0.25), Xf.quantile(0.75)
        self.medians = X_train[config.RAW_FEATURES].median()
        # training range: values outside it (e.g. age 45, BMI 44) are clamped before reaching the model
        self.lo, self.hi = X_train[config.RAW_FEATURES].min(), X_train[config.RAW_FEATURES].max()

    # ------------------------------------------------------------------ helpers
    def missing(self, patient: dict) -> list[str]:
        return [f for f in config.RAW_FEATURES if patient.get(f) is None]

    def to_frame(self, patient: dict) -> pd.DataFrame:
        """Model input row: missing optional values -> training median; values outside the
        training range -> nearest edge of that range (the model has never seen beyond it)."""
        row = {}
        for f in config.RAW_FEATURES:
            v = patient.get(f)
            v = self.medians[f] if v is None else float(v)
            row[f] = float(min(max(v, self.lo[f]), self.hi[f]))
        return pd.DataFrame([row])

    def adjustments(self, patient: dict) -> list[dict]:
        """Inputs outside the research data range, and the value the model used instead."""
        out = []
        for f in config.NUMERICAL_FEATURES:
            v = patient.get(f)
            tol = 0.01 * (self.hi[f] - self.lo[f])  # ignore negligible clamps (e.g. 0 vs 0.002)
            if v is not None and not (self.lo[f] - tol <= float(v) <= self.hi[f] + tol):
                used = self.lo[f] if float(v) < self.lo[f] else self.hi[f]
                out.append({"feature": f, "label": label_for(f), "given": float(v), "used": round(float(used), 1)})
        return out

    @staticmethod
    def _uses_missing(feature: str, missing: set) -> bool:
        """Does this (possibly engineered) feature depend on a value that was not provided?"""
        sources = {f for part in feature.split("__x__") for f in DERIVED_FROM.get(part, [part])}
        return bool(sources & missing)

    def _describe(self, feature: str, value: float, shap_value: float, missing: set = frozenset()) -> str:
        """One plain-English sentence, e.g. 'Low MMSE score (12.3) increased the risk'."""
        text = self._describe_core(feature, value, shap_value)
        if missing and self._uses_missing(feature, missing):
            text += " (not provided, so a typical value was used)"
        return text

    def _describe_core(self, feature: str, value: float, shap_value: float) -> str:
        effect = "increased the risk" if shap_value > 0 else "lowered the risk"
        if feature == EMB_GROUP:
            return f"The Transformer's learned pattern across all features {effect}"
        if feature == TXP_GROUP:
            return f"The Transformer's own risk estimate ({value:.0%}) {effect}"
        if feature in BINARY_FLAG_TEXT:
            text = BINARY_FLAG_TEXT[feature][0 if value >= 0.5 else 1]
            return f"{text} {effect}"
        if "__x__" in feature:
            return f"The combination of {_lower_first(label_for(feature).replace(' x ', ' and '))} {effect}"
        label = label_for(feature)
        if feature in config.CATEGORICAL_CARDINALITY and config.CATEGORICAL_CARDINALITY[feature] == 2:
            return f"{'Presence' if value >= 0.5 else 'Absence'} of {_lower_first(label)} {effect}"
        if feature in self.q25.index:
            level = "Low" if value < self.q25[feature] else "High" if value > self.q75[feature] else "Typical"
            return f"{level} {_lower_first(label)} ({value:.1f}) {effect}"
        return f"{label} ({value:.2f}) {effect}"

    def _grouped_shap(self, Xh: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
        """Collapse embedding columns into one group; returns per-feature value + SHAP."""
        emb_cols = [c for c in Xh.columns if c.startswith("tx_emb_")]
        rows = [{"feature": c, "value": float(Xh[c].iloc[0]), "shap": float(v)}
                for c, v in zip(Xh.columns, values) if c not in emb_cols and c != "tx_prob"]
        rows.append({"feature": EMB_GROUP, "value": float("nan"),
                     "shap": float(values[[Xh.columns.get_loc(c) for c in emb_cols]].sum())})
        rows.append({"feature": TXP_GROUP, "value": float(Xh["tx_prob"].iloc[0]),
                     "shap": float(values[Xh.columns.get_loc("tx_prob")])})
        df = pd.DataFrame(rows)
        df["label"] = [label_for(f) for f in df["feature"]]
        return df.reindex(df["shap"].abs().sort_values(ascending=False).index).reset_index(drop=True)

    # ------------------------------------------------------------------ main API
    def predict(self, patient: dict, explain: bool = True) -> dict:
        X = self.to_frame(patient)
        missing = self.missing(patient)
        prob = float(self.hybrid.predict_proba(X)[0])
        adjusted = self.adjustments(patient)
        out = {"inputs": dict(patient), "not_provided": missing, "adjusted": adjusted,
               "probability": prob, "risk_level": config.risk_level(prob),
               "predicted_label": int(prob >= self.hybrid.threshold),
               "threshold": self.hybrid.threshold, "model_version": self.model_version}
        if not explain:
            return out
        Xh = self.hybrid.features(X)
        sv = self.explainer(Xh)
        g = self._grouped_shap(Xh, sv.values[0])
        g["text"] = [self._describe(f, v, s, set(missing)) for f, v, s in zip(g["feature"], g["value"], g["shap"])]
        for a in adjusted:  # say plainly when a value was outside the research data range
            g.loc[g["feature"] == a["feature"], "text"] = (
                f"{a['label']} of {a['given']:g} is outside the research data, so {a['used']:g} was used; this "
                + ("increased" if float(g.loc[g['feature'] == a['feature'], 'shap'].sum()) > 0 else "lowered") + " the risk")
        # plain-language top factors only use information the person actually gave
        known = g[[not self._uses_missing(f, set(missing)) for f in g["feature"]]]
        up = known[known["shap"] > 0].head(5)
        down = known[known["shap"] < 0].head(5)
        out.update({
            "base_value": float(np.ravel(sv.base_values)[0]),
            "raw_probability": float(self.hybrid.predict_proba_raw(X)[0]),
            "factors": g.to_dict(orient="records"),
            "top_increasing": up[["feature", "label", "shap", "text"]].to_dict(orient="records"),
            "top_decreasing": down[["feature", "label", "shap", "text"]].to_dict(orient="records"),
        })
        return out

    def top_factors_json(self, result: dict) -> list[dict]:
        """Compact form stored in the Digital Twin database."""
        return [{"feature": f["feature"], "label": f["label"], "shap": round(f["shap"], 4), "text": f["text"]}
                for f in (result["top_increasing"] + result["top_decreasing"])]

    def attention(self, patient: dict) -> list[dict]:
        """[CLS] -> feature attention of the Transformer for one patient."""
        t = self.hybrid.transformer
        Xf = self.hybrid.fe.transform(self.to_frame(patient))
        att = t.cls_attention(Xf).iloc[0].sort_values(ascending=False)
        return [{"feature": f, "label": label_for(f), "attention": float(a)} for f, a in att.items()]

    def plain_xgb_shap(self, patient: dict, k: int = 8) -> list[dict]:
        """SHAP of the plain XGBoost on the ORIGINAL features (for comparison)."""
        X = self.to_frame(patient)
        sv = self.plain_explainer(X).values[0]
        order = np.argsort(-np.abs(sv))[:k]
        return [{"feature": X.columns[i], "label": label_for(X.columns[i]), "value": float(X.iloc[0, i]),
                 "shap": float(sv[i]),
                 "text": self._describe(X.columns[i], float(X.iloc[0, i]), float(sv[i]), set(self.missing(patient)))}
                for i in order]

    # ------------------------------------------------------------------ plots
    def _explanation(self, result: dict, max_display: int) -> shap.Explanation:
        f = pd.DataFrame(result["factors"])
        # the embedding group has no single value, so show its size instead of "nan"
        data = np.array([f"{self.hybrid.transformer.params['d']} dims"
                         if feat == EMB_GROUP else round(float(v), 2)
                         for feat, v in zip(f["feature"], f["value"])], dtype=object)
        return shap.Explanation(values=f["shap"].values, base_values=result["base_value"],
                                data=data, feature_names=list(f["label"]))

    def waterfall_png(self, result: dict, max_display: int = 12) -> bytes:
        with _PLOT_LOCK:
            plt.figure()
            shap.plots.waterfall(self._explanation(result, max_display), max_display=max_display, show=False)
            fig = plt.gcf()
            fig.set_size_inches(9, 6.5)
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
            plt.close("all")
        return buf.getvalue()

    def force_png(self, result: dict, k: int = 8) -> bytes:
        f = pd.DataFrame(result["factors"]).head(k)
        rest = sum(r["shap"] for r in result["factors"][k:])
        values = np.append(f["shap"].values, rest)
        names = list(f["label"]) + ["All other features"]
        data = np.array([("" if feat == EMB_GROUP else round(float(v), 2)) for feat, v in zip(f["feature"], f["value"])]
                        + [""], dtype=object)
        with _PLOT_LOCK:
            shap.force_plot(result["base_value"], values, data, feature_names=names,
                            matplotlib=True, show=False, figsize=(16, 3.4), text_rotation=12)
            buf = io.BytesIO()
            plt.gcf().savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close("all")
        return buf.getvalue()


_instance: RiskPredictor | None = None


def get_predictor() -> RiskPredictor:
    """Load models once per process."""
    global _instance
    if _instance is None:
        _instance = RiskPredictor()
    return _instance
