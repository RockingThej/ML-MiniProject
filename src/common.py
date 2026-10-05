"""Shared config, data loading and preprocessing for the MI-risk project."""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "heart.csv"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
RESULTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

TARGET = "output"          # 1 = high MI risk, 0 = low MI risk
RANDOM_STATE = 42

CONTINUOUS = ["age", "trtbps", "chol", "thalachh", "oldpeak"]
NOMINAL = ["cp", "restecg", "slp", "thall"]   # unordered categories -> one-hot
PASSTHROUGH = ["sex", "fbs", "exng", "caa"]   # binary / small ordinal count

# Some copies of the dataset use the original UCI column names.
RENAME = {"target": "output", "trestbps": "trtbps", "thalach": "thalachh",
          "exang": "exng", "slope": "slp", "ca": "caa", "thal": "thall"}


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Download heart.csv from the Kaggle dataset "
            "'Heart Attack Analysis & Prediction Dataset' and place it in data/.")
    df = pd.read_csv(path).rename(columns=RENAME)
    return df.drop_duplicates().reset_index(drop=True)


def make_preprocessor() -> ColumnTransformer:
    """Scaling + one-hot encoding. Always used inside a Pipeline so it is
    fitted on training folds only (no data leakage)."""
    return ColumnTransformer([
        ("num", StandardScaler(), CONTINUOUS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), NOMINAL),
        ("pass", "passthrough", PASSTHROUGH),
    ])


def get_xy(df: pd.DataFrame):
    return df[CONTINUOUS + NOMINAL + PASSTHROUGH], df[TARGET]


def train_test(df: pd.DataFrame, test_size: float = 0.2):
    X, y = get_xy(df)
    return train_test_split(X, y, test_size=test_size, stratify=y,
                            random_state=RANDOM_STATE)
