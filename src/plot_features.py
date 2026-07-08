"""
src/plot_features.py
---------------------
Generates feature importance + correlation plots.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pandas as pd

from src.preprocess import load_data, FEATURE_COLS, LABEL_MAP


def plot_feature_importance(rf_model, feature_names: list):
    importances = rf_model.feature_importances_
    idx = np.argsort(importances)[::-1]

    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(feature_names)))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(feature_names)),
                  importances[idx],
                  color=colors,
                  edgecolor="white",
                  linewidth=0.8)
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels([feature_names[i] for i in idx],
                       rotation=40, ha="right", fontsize=10)
    ax.set_ylabel("Importance (Gini)", fontsize=11)
    ax.set_title("Random Forest – Feature Importance\n(RTOS Health Prediction)",
                 fontsize=13, fontweight="bold")
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#ffffff")

    for bar, val in zip(bars, importances[idx]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = "plots/feature_importance.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[plot] Feature importance saved → {out}")


def plot_correlation_matrix(csv_path: str = "data/rtos_health_data.csv"):
    df = load_data(csv_path)
    corr = df[FEATURE_COLS + ["health_label"]].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0,
                linewidths=0.4, ax=ax,
                annot_kws={"fontsize": 8})
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = "plots/correlation_matrix.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[plot] Correlation matrix saved → {out}")


def plot_label_distribution(csv_path: str = "data/rtos_health_data.csv"):
    df = load_data(csv_path)
    counts = df["health_label"].value_counts().sort_index()
    labels = [LABEL_MAP[k] for k in counts.index]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 3,
                str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Sample Count", fontsize=11)
    ax.set_title("Dataset – Health Label Distribution", fontsize=13, fontweight="bold")
    ax.set_facecolor("#f8f9fa")
    plt.tight_layout()
    out = "plots/label_distribution.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[plot] Label distribution saved → {out}")


def plot_boxplots(csv_path: str = "data/rtos_health_data.csv"):
    df = load_data(csv_path)
    df["health_label_str"] = df["health_label"].map(LABEL_MAP)
    palette = {"Healthy": "#2ecc71", "Warning": "#f39c12", "Critical": "#e74c3c"}
    order   = ["Healthy", "Warning", "Critical"]

    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    axes = axes.flatten()

    for ax, feat in zip(axes, FEATURE_COLS):
        sns.boxplot(data=df, x="health_label_str", y=feat,
                    order=order, palette=palette, hue="health_label_str",
                    hue_order=order, legend=False, ax=ax,
                    linewidth=0.8, fliersize=3)
        ax.set_title(feat, fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=8)

    fig.suptitle("Feature Distribution per Health Label",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = "plots/feature_boxplots.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Boxplots saved → {out}")


if __name__ == "__main__":
    rf = joblib.load("models/random_forest.pkl")
    plot_feature_importance(rf, FEATURE_COLS)
    plot_correlation_matrix()
    plot_label_distribution()
    plot_boxplots()
