"""
src/preprocess.py
-----------------
Loads and preprocesses the RTOS health CSV dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

FEATURE_COLS = [
    "cpu_load",
    "queue_level",
    "queue_dropped",
    "process_jitter",
    "process_deadline_miss",
    "process_exec_time",
    "process_stack_left",
    "sensor_deadline_miss",
    "comm_deadline_miss",
]
LABEL_COL = "health_label"
LABEL_MAP = {0: "Healthy", 1: "Warning", 2: "Critical"}


def load_data(csv_path: str) -> pd.DataFrame:
    """Load and do basic sanity checks on the CSV."""
    df = pd.read_csv(csv_path)
    print(f"[preprocess] Loaded {len(df)} rows, {df.shape[1]} columns")

    missing = df.isnull().sum()
    if missing.any():
        print("[preprocess] Missing values detected – filling with column median")
        df.fillna(df.median(numeric_only=True), inplace=True)

    # Clip obviously impossible values
    df["cpu_load"] = df["cpu_load"].clip(0, 100)
    df["process_stack_left"] = df["process_stack_left"].clip(0, None)

    return df


def split_and_scale(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Returns
    -------
    X_train, X_test, y_train, y_test : numpy arrays (scaled features)
    scaler                            : fitted StandardScaler
    feature_names                     : list[str]
    """
    X = df[FEATURE_COLS].values
    y = df[LABEL_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, FEATURE_COLS


def save_scaler(scaler, path: str = "models/scaler.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)
    print(f"[preprocess] Scaler saved → {path}")


def load_scaler(path: str = "models/scaler.pkl"):
    return joblib.load(path)
