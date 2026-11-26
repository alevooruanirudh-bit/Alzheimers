"""
Baseline models: Logistic Regression, SVM, Random Forest and plain XGBoost.

All baselines use the RAW clinical features (no feature engineering), so the
gain from the proposed pipeline is easy to see. Linear models get standard
scaling and one-hot encoding of the two multi-level categorical features.
LR and SVM have a small hyper-parameter grid searched with cross-validation
inside the training data only.
"""
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from src import config

MULTI_LEVEL = ["Ethnicity", "EducationLevel"]
BINARY = [c for c in config.CATEGORICAL_FEATURES if c not in MULTI_LEVEL]


def _linear_preprocessor() -> ColumnTransformer:
    """Scale numbers, one-hot multi-level categories, pass binaries through."""
    return ColumnTransformer([
        ("num", StandardScaler(), config.NUMERICAL_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), MULTI_LEVEL),
        ("bin", "passthrough", BINARY),
    ])


def _inner_cv():
    return StratifiedKFold(n_splits=3, shuffle=True, random_state=config.SEED)


def logistic_regression():
    pipe = Pipeline([("prep", _linear_preprocessor()),
                     ("clf", LogisticRegression(max_iter=5000, random_state=config.SEED))])
    return GridSearchCV(pipe, {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=_inner_cv(), scoring="accuracy")


def svm():
    pipe = Pipeline([("prep", _linear_preprocessor()),
                     ("clf", SVC(kernel="rbf", probability=True, random_state=config.SEED))])
    grid = {"clf__C": [0.3, 1, 3, 10, 30], "clf__gamma": ["scale", 0.01, 0.03]}
    return GridSearchCV(pipe, grid, cv=_inner_cv(), scoring="accuracy")


def random_forest():
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=-1,
                                  random_state=config.SEED)


def plain_xgboost():
    """XGBoost with library-default settings (no tuning, no feature engineering)."""
    return xgb.XGBClassifier(random_state=config.SEED, n_jobs=4, verbosity=0)


BASELINES = {
    "Logistic Regression": logistic_regression,
    "SVM (RBF)": svm,
    "Random Forest": random_forest,
    "XGBoost (plain)": plain_xgboost,
}
