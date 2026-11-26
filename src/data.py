"""
Data loading and the train/test split.

The split is created ONCE with a fixed seed and saved to
data/processed/split_indices.json. Every other script reads the saved indices,
so all models are trained and evaluated on exactly the same patients.

Run `python -m src.data` to (re)create the split and print a summary.
"""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src import config


def ensure_dataset() -> None:
    """Download the Kaggle CSV if it is missing. Never generates fake data."""
    if config.RAW_CSV.exists():
        return
    print(f"Dataset not found at {config.RAW_CSV}. Trying the Kaggle API ...")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", config.KAGGLE_DATASET,
             "-p", str(config.DATA_RAW_DIR), "--unzip"],
            check=True,
        )
    except Exception as exc:  # noqa: BLE001 - we want to show any failure
        sys.exit(
            f"Kaggle download failed ({exc}).\n"
            f"Please download '{config.KAGGLE_DATASET}' manually and place "
            f"alzheimers_disease_data.csv in {config.DATA_RAW_DIR}"
        )


def load_raw() -> pd.DataFrame:
    """Load the CSV and drop identifier columns (no predictive meaning)."""
    ensure_dataset()
    df = pd.read_csv(config.RAW_CSV)
    df = df.drop(columns=[c for c in config.ID_COLUMNS if c in df.columns])
    missing = set(config.RAW_FEATURES + [config.TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")
    return df[config.RAW_FEATURES + [config.TARGET]].reset_index(drop=True)


def make_split(df: pd.DataFrame) -> dict:
    """Stratified 80/20 split with the fixed seed; indices saved to JSON."""
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=config.TEST_SIZE, stratify=df[config.TARGET],
        random_state=config.SEED,
    )
    split = {"train": sorted(int(i) for i in train_idx),
             "test": sorted(int(i) for i in test_idx),
             "seed": config.SEED, "test_size": config.TEST_SIZE}
    config.SPLIT_FILE.write_text(json.dumps(split))
    return split


def load_split() -> dict:
    """Read the saved split (create it the first time)."""
    if not config.SPLIT_FILE.exists():
        return make_split(load_raw())
    return json.loads(config.SPLIT_FILE.read_text())


def get_train() -> tuple[pd.DataFrame, pd.Series]:
    """Training features and labels. This is all that training code may see."""
    df = load_raw()
    rows = df.iloc[load_split()["train"]].reset_index(drop=True)
    return rows[config.RAW_FEATURES], rows[config.TARGET]


def get_test() -> tuple[pd.DataFrame, pd.Series]:
    """Held-out test features and labels. Used ONLY by evaluate/explain/seed."""
    df = load_raw()
    rows = df.iloc[load_split()["test"]].reset_index(drop=True)
    return rows[config.RAW_FEATURES], rows[config.TARGET]


def get_test_original_index() -> list[int]:
    """Row numbers of the test patients in the original CSV (for traceability)."""
    return load_split()["test"]


def cv_folds(y: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    """The single, fixed set of stratified CV folds used by every model."""
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    return list(skf.split(np.zeros(len(y)), y))


if __name__ == "__main__":
    data = load_raw()
    s = make_split(data)
    y = data[config.TARGET]
    print(f"Rows: {len(data)}  features: {len(config.RAW_FEATURES)}")
    print(f"Train: {len(s['train'])}  Test: {len(s['test'])}")
    print(f"Positive rate  train={y.iloc[s['train']].mean():.3f}  test={y.iloc[s['test']].mean():.3f}")
    print(f"Saved split to {config.SPLIT_FILE}")
