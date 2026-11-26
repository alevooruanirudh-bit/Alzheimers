"""
Central configuration for the whole project.

Everything that a reviewer might want to change (paths, random seed, split
size, risk thresholds, tuning budgets, feature groups) lives in this one file
so that no "magic numbers" are hidden inside the other modules.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DATABASE_DIR = ROOT_DIR / "database"

RAW_CSV = DATA_RAW_DIR / "alzheimers_disease_data.csv"
SPLIT_FILE = DATA_PROCESSED_DIR / "split_indices.json"
DATABASE_URL = f"sqlite:///{DATABASE_DIR / 'digital_twin.db'}"

KAGGLE_DATASET = "rabieelkharoua/alzheimers-disease-dataset"

for _d in (DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, DATABASE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Reproducibility and evaluation protocol
# --------------------------------------------------------------------------
SEED = 42            # one fixed seed, never searched over
TEST_SIZE = 0.20     # stratified 80/20 split, made once and saved
N_FOLDS = 5          # stratified K-fold on the TRAINING set only

# --------------------------------------------------------------------------
# Dataset schema
# --------------------------------------------------------------------------
TARGET = "Diagnosis"
ID_COLUMNS = ["PatientID", "DoctorInCharge"]  # identifiers, dropped

# Categorical features and their number of levels (from the dataset documentation).
CATEGORICAL_CARDINALITY = {
    "Gender": 2,
    "Ethnicity": 4,
    "EducationLevel": 4,
    "Smoking": 2,
    "FamilyHistoryAlzheimers": 2,
    "CardiovascularDisease": 2,
    "Diabetes": 2,
    "Depression": 2,
    "HeadInjury": 2,
    "Hypertension": 2,
    "MemoryComplaints": 2,
    "BehavioralProblems": 2,
    "Confusion": 2,
    "Disorientation": 2,
    "PersonalityChanges": 2,
    "DifficultyCompletingTasks": 2,
    "Forgetfulness": 2,
}
CATEGORICAL_FEATURES = list(CATEGORICAL_CARDINALITY)

NUMERICAL_FEATURES = [
    "Age", "BMI", "AlcoholConsumption", "PhysicalActivity", "DietQuality",
    "SleepQuality", "SystolicBP", "DiastolicBP", "CholesterolTotal",
    "CholesterolLDL", "CholesterolHDL", "CholesterolTriglycerides", "MMSE",
    "FunctionalAssessment", "ADL",
]

RAW_FEATURES = [
    "Age", "Gender", "Ethnicity", "EducationLevel", "BMI", "Smoking",
    "AlcoholConsumption", "PhysicalActivity", "DietQuality", "SleepQuality",
    "FamilyHistoryAlzheimers", "CardiovascularDisease", "Diabetes",
    "Depression", "HeadInjury", "Hypertension", "SystolicBP", "DiastolicBP",
    "CholesterolTotal", "CholesterolLDL", "CholesterolHDL",
    "CholesterolTriglycerides", "MMSE", "FunctionalAssessment",
    "MemoryComplaints", "BehavioralProblems", "ADL", "Confusion",
    "Disorientation", "PersonalityChanges", "DifficultyCompletingTasks",
    "Forgetfulness",
]

SYMPTOM_FEATURES = [
    "MemoryComplaints", "BehavioralProblems", "Confusion", "Disorientation",
    "PersonalityChanges", "DifficultyCompletingTasks", "Forgetfulness",
]

# --------------------------------------------------------------------------
# Risk levels (applied to the CALIBRATED probability of Alzheimer's)
# --------------------------------------------------------------------------
RISK_LOW_MAX = 0.35    # p < 0.35          -> Low
RISK_HIGH_MIN = 0.70   # 0.35 <= p <= 0.70 -> Medium, p > 0.70 -> High


def risk_level(probability: float) -> str:
    """Map a calibrated probability to a Low / Medium / High risk label."""
    if probability < RISK_LOW_MAX:
        return "Low"
    if probability <= RISK_HIGH_MIN:
        return "Medium"
    return "High"


# --------------------------------------------------------------------------
# Training budgets
# --------------------------------------------------------------------------
XGB_OPTUNA_TRIALS = 100          # per tuned XGBoost model (spec: >= 100)
TRANSFORMER_OPTUNA_TRIALS = 20   # Transformer tuning budget
TRANSFORMER_TUNING_FOLDS = 3     # folds used inside each Transformer trial
TRANSFORMER_MAX_EPOCHS = 200
TRANSFORMER_PATIENCE = 20        # early-stopping patience (epochs)
TRANSFORMER_VAL_FRACTION = 0.15  # inner validation split for early stopping

# Number of interaction terms built from the top training-set features
N_TOP_FEATURES_FOR_INTERACTIONS = 4

MODEL_VERSION = "hybrid-ftt-xgb-v1"

# --------------------------------------------------------------------------
# Digital Twin what-if simulation: which inputs a patient can change
# --------------------------------------------------------------------------
CHANGEABLE_FEATURES = [
    "MMSE", "FunctionalAssessment", "ADL", "PhysicalActivity", "SleepQuality",
    "DietQuality", "AlcoholConsumption", "BMI", "SystolicBP", "DiastolicBP",
    "CholesterolTotal", "CholesterolLDL", "CholesterolHDL",
    "CholesterolTriglycerides", "Smoking",
]
FIXED_FEATURES = ["Age", "Gender", "Ethnicity", "EducationLevel", "FamilyHistoryAlzheimers"]
