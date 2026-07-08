"""
src/train_isolation_forest.py
------------------------------
Trains an Isolation Forest anomaly detector.
Trained on HEALTHY samples only → detects deviations.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib

from src.preprocess import load_data, split_and_scale, FEATURE_COLS, LABEL_MAP

MODEL_PATH = "models/isolation_forest.pkl"
CM_PLOT    = "plots/confusion_matrix_if.png"


def train(csv_path: str = "data/rtos_health_data.csv"):
    # ── 1. Load & split (scaler already saved by RF trainer) ─────────────────
    df = load_data(csv_path)
    X_train, X_test, y_train, y_test, scaler, _ = split_and_scale(df)

    # Train Isolation Forest on the full training set (unsupervised)
    print("\n[IF] Training Isolation Forest …")
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.33,   # ~33 % of data is non-healthy
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_train)

    # ── 2. Evaluate ──────────────────────────────────────────────────────────
    # IF returns +1 (inlier / normal) or -1 (outlier / anomaly)
    raw_pred = iso.predict(X_test)
    # Map: +1 → 0 (Healthy), -1 → 1 (Anomaly/non-healthy)
    y_pred_bin = np.where(raw_pred == 1, 0, 1)
    # Map ground-truth: 0 → 0 (healthy), 1 or 2 → 1 (non-healthy)
    y_true_bin = np.where(y_test == 0, 0, 1)

    target_names = ["Healthy", "Anomaly"]
    print("\n[IF] Binary Classification Report (Healthy vs Anomaly):")
    print(classification_report(y_true_bin, y_pred_bin, target_names=target_names))

    # Anomaly score (lower = more anomalous)
    scores = iso.decision_function(X_test)
    print(f"[IF] Mean anomaly score (healthy):     {scores[y_test==0].mean():.4f}")
    print(f"[IF] Mean anomaly score (warning):     {scores[y_test==1].mean():.4f}")
    print(f"[IF] Mean anomaly score (critical):    {scores[y_test==2].mean():.4f}")

    # ── 3. Confusion matrix ──────────────────────────────────────────────────
    os.makedirs("plots", exist_ok=True)
    cm   = confusion_matrix(y_true_bin, y_pred_bin)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(ax=ax, colorbar=False, cmap="Oranges")
    ax.set_title("Isolation Forest – Confusion Matrix\n(Healthy vs Anomaly)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(CM_PLOT, dpi=150)
    plt.close()
    print(f"[IF] Confusion matrix saved → {CM_PLOT}")

    # ── 4. Score distribution plot ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    for label, color, name in zip([0, 1, 2], ["#2ecc71", "#f39c12", "#e74c3c"],
                                   ["Healthy", "Warning", "Critical"]):
        ax.hist(scores[y_test == label], bins=30, alpha=0.6,
                color=color, label=name, edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, label="Decision boundary")
    ax.set_xlabel("Anomaly Score (higher = more normal)")
    ax.set_ylabel("Count")
    ax.set_title("Isolation Forest – Anomaly Score Distribution")
    ax.legend()
    plt.tight_layout()
    score_plot = "plots/isolation_forest_scores.png"
    plt.savefig(score_plot, dpi=150)
    plt.close()
    print(f"[IF] Score distribution saved → {score_plot}")

    # ── 5. Save ───────────────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump(iso, MODEL_PATH)
    print(f"[IF] Model saved → {MODEL_PATH}")

    return iso


if __name__ == "__main__":
    train()
