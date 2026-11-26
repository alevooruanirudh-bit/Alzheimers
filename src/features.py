"""
Feature engineering and feature selection.

Two kinds of features are added:

1. Fixed clinical features (row-wise formulas, nothing is learned, so they
   cannot leak information):
     - CognitiveFunctionalComposite : average of MMSE/30, FunctionalAssessment/10, ADL/10
     - SymptomCount                 : number of the 7 cognitive/behavioural symptoms
     - CardioRiskCount              : smoking + CVD + diabetes + hypertension + high cholesterol
     - LDL_HDL_Ratio                : CholesterolLDL / CholesterolHDL
     - PulsePressure                : SystolicBP - DiastolicBP
     - MMSE_Impaired                : MMSE < 24 (standard clinical cut-off for cognitive impairment)
     - FA_Low, ADL_Low              : score below the scale midpoint (5 out of 10)
     - ImpairmentDomainCount        : FA_Low + ADL_Low + MemoryComplaints + BehavioralProblems

2. Learned features, fitted on TRAINING rows only:
     - pairwise interaction terms (products) between the top-k most important
       features, where importance comes from an XGBoost fitted on the training rows;
     - optional feature selection: a feature is kept only if its importance is
       higher than that of random "noise probe" columns (a light Boruta idea).

`FeatureEngineer` is an sklearn transformer, so in cross-validation it is
re-fitted inside every fold and never sees the validation fold.
"""
from itertools import combinations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin

from src import config

# Engineered binary flags are categorical tokens for the Transformer.
ENGINEERED_CATEGORICAL = {"MMSE_Impaired": 2, "FA_Low": 2, "ADL_Low": 2}

# Human-readable names, used by the explanation layer and the web app.
FEATURE_LABELS = {
    "CognitiveFunctionalComposite": "Cognitive/functional composite score",
    "SymptomCount": "Number of cognitive/behavioural symptoms",
    "CardioRiskCount": "Number of cardiovascular risk factors",
    "LDL_HDL_Ratio": "LDL/HDL cholesterol ratio",
    "PulsePressure": "Pulse pressure",
    "MMSE_Impaired": "MMSE below 24 (impaired range)",
    "FA_Low": "Functional assessment below 5",
    "ADL_Low": "Daily-living (ADL) score below 5",
    "ImpairmentDomainCount": "Number of impaired functional/behavioural domains",
    "MMSE": "Memory and thinking score (MMSE)",
    "FunctionalAssessment": "Everyday tasks score (functional assessment)",
    "ADL": "Self-care score (ADL)",
    "MemoryComplaints": "Memory complaints",
    "BehavioralProblems": "Behavioural problems",
    "FamilyHistoryAlzheimers": "Family history of Alzheimer's",
    "CardiovascularDisease": "Cardiovascular disease",
    "PhysicalActivity": "Physical activity",
    "SleepQuality": "Sleep quality",
    "DietQuality": "Diet quality",
    "AlcoholConsumption": "Alcohol consumption",
    "EducationLevel": "Education level",
    "SystolicBP": "Systolic blood pressure",
    "DiastolicBP": "Diastolic blood pressure",
    "CholesterolTotal": "Total cholesterol",
    "CholesterolLDL": "LDL cholesterol",
    "CholesterolHDL": "HDL cholesterol",
    "CholesterolTriglycerides": "Triglycerides",
    "DifficultyCompletingTasks": "Difficulty completing tasks",
    "PersonalityChanges": "Personality changes",
    "HeadInjury": "Head injury history",
}


def label_for(feature: str) -> str:
    """Readable label for any feature, including interaction terms."""
    if "__x__" in feature:
        a, b = feature.split("__x__")
        return f"{label_for(a)} x {label_for(b)}"
    return FEATURE_LABELS.get(feature, feature)


def add_clinical_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add the fixed, formula-based clinical features (no fitting involved)."""
    X = X.copy()
    X["CognitiveFunctionalComposite"] = (X["MMSE"] / 30 + X["FunctionalAssessment"] / 10 + X["ADL"] / 10) / 3
    X["SymptomCount"] = X[config.SYMPTOM_FEATURES].sum(axis=1)
    X["CardioRiskCount"] = (X["Smoking"] + X["CardiovascularDisease"] + X["Diabetes"]
                            + X["Hypertension"] + (X["CholesterolTotal"] >= 240).astype(int))
    X["LDL_HDL_Ratio"] = X["CholesterolLDL"] / X["CholesterolHDL"].clip(lower=1e-6)
    X["PulsePressure"] = X["SystolicBP"] - X["DiastolicBP"]
    X["MMSE_Impaired"] = (X["MMSE"] < 24).astype(int)
    X["FA_Low"] = (X["FunctionalAssessment"] < 5).astype(int)
    X["ADL_Low"] = (X["ADL"] < 5).astype(int)
    X["ImpairmentDomainCount"] = X["FA_Low"] + X["ADL_Low"] + X["MemoryComplaints"] + X["BehavioralProblems"]
    return X


def _importance(X: pd.DataFrame, y, seed: int) -> pd.Series:
    """XGBoost gain importance of every column, fitted on the given (training) rows."""
    model = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                              subsample=0.9, colsample_bytree=0.8,
                              random_state=seed, n_jobs=4, verbosity=0)
    model.fit(X, y)
    gain = model.get_booster().get_score(importance_type="gain")
    return pd.Series({c: gain.get(c, 0.0) for c in X.columns}).sort_values(ascending=False)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Parameters
    ----------
    engineer : add clinical features and interaction terms (False = raw features only)
    select   : keep only features more important than random noise probes
    n_top    : number of top features combined into pairwise interactions
    """

    def __init__(self, engineer: bool = True, select: bool = False,
                 n_top: int = config.N_TOP_FEATURES_FOR_INTERACTIONS, seed: int = config.SEED):
        self.engineer = engineer
        self.select = select
        self.n_top = n_top
        self.seed = seed

    # -- fitting (training rows only) ------------------------------------
    def fit(self, X: pd.DataFrame, y=None):
        X = X[config.RAW_FEATURES]
        self.interactions_: list[tuple[str, str]] = []
        if self.engineer:
            base = add_clinical_features(X)
            # pick the top-k raw/clinical features and multiply them pairwise
            top = list(_importance(base, y, self.seed).index[: self.n_top])
            self.interactions_ = list(combinations(top, 2))
        full = self._build(X)
        self.all_features_ = list(full.columns)

        if self.select:
            # Noise probes: shuffled copies of real columns have no relation to y.
            rng = np.random.default_rng(self.seed)
            probes = pd.DataFrame({f"_probe{i}": rng.permutation(full.iloc[:, i % full.shape[1]].values)
                                   for i in range(5)}, index=full.index)
            imp = _importance(pd.concat([full, probes], axis=1), y, self.seed)
            cutoff = imp[probes.columns].max()
            self.importance_ = imp.drop(probes.columns)
            self.selected_features_ = [c for c in self.all_features_ if self.importance_[c] > cutoff]
        else:
            self.selected_features_ = list(self.all_features_)
        return self

    # -- transforming ------------------------------------------------------
    def _build(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[config.RAW_FEATURES].astype(float)
        if not self.engineer:
            return X
        X = add_clinical_features(X)
        for a, b in self.interactions_:
            X[f"{a}__x__{b}"] = X[a] * X[b]
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._build(X)[self.selected_features_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.selected_features_)

    # -- helpers for the Transformer tokeniser --------------------------
    def categorical_cardinality(self) -> dict[str, int]:
        """Which output columns are categorical (and how many levels each has)."""
        allc = {**config.CATEGORICAL_CARDINALITY, **ENGINEERED_CATEGORICAL}
        return {c: allc[c] for c in self.selected_features_ if c in allc}
