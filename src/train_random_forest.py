"""
src/train_random_forest.py
--------------------------
Trains a Random Forest classifier on RTOS health data.
Saves the model and prints accuracy + confusion matrix.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
import joblib

from src.preprocess import load_data, split_and_scale, save_scaler, LABEL_MAP

MODEL_PATH = "models/random_forest.pkl"
CM_PLOT    = "plots/confusion_matrix_rf.png"


def train(csv_path: str = "data/rtos_health_data.csv"):
    # ── 1. Load & preprocess ────────────────────────────────────────────────
    df = load_data(csv_path)
    X_train, X_test, y_train, y_test, scaler, feat_names = split_and_scale(df)
    save_scaler(scaler)

    # ── 2. Train ─────────────────────────────────────────────────────────────
    print("\n[RF] Training Random Forest …")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # ── 3. Evaluate ──────────────────────────────────────────────────────────
    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n[RF] Accuracy: {acc * 100:.2f}%")
    print("\n[RF] Classification Report:")
    target_names = [LABEL_MAP[k] for k in sorted(LABEL_MAP)]
    print(classification_report(y_test, y_pred, target_names=target_names))

    # ── 4. Confusion matrix plot ─────────────────────────────────────────────
    os.makedirs("plots", exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=target_names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Random Forest – Confusion Matrix\nAccuracy: {acc*100:.1f}%",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(CM_PLOT, dpi=150)
    plt.close()
    print(f"[RF] Confusion matrix saved → {CM_PLOT}")

    # ── 5. Save model ────────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"[RF] Model saved → {MODEL_PATH}")

    return clf, feat_names, acc


if __name__ == "__main__":
    train()
