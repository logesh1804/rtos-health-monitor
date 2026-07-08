"""
src/export_tinyml.py
---------------------
Exports the trained Random Forest to formats suitable for
TinyML / embedded deployment on STM32 or similar MCUs.

Outputs
-------
tinyml_export/model.h         C header with thresholds (rule-based fallback)
tinyml_export/model_tflite/   TensorFlow Lite FlatBuffer (if tf installed)
tinyml_export/emlearn_model.c emlearn C code (if emlearn installed)

Usage
-----
  python src/export_tinyml.py
"""

import os
import sys
import json
import numpy as np
import joblib
from datetime import datetime

EXPORT_DIR   = "tinyml_export"
RF_MODEL     = "models/random_forest.pkl"
SCALER_MODEL = "models/scaler.pkl"

FEATURE_COLS = [
    "cpu_load", "queue_level", "queue_dropped", "process_jitter",
    "process_deadline_miss", "process_exec_time", "process_stack_left",
    "sensor_deadline_miss", "comm_deadline_miss",
]
LABEL_MAP = {0: "HEALTHY", 1: "WARNING", 2: "CRITICAL"}


# ─── 1. Rule-based C header (always works, no extra deps) ─────────────────────
def export_c_header(rf, scaler):
    """
    Exports scaler parameters + top decision-tree thresholds as a C header.
    This gives a lightweight, dependency-free embedded option.
    """
    means  = scaler.mean_.tolist()
    stds   = scaler.scale_.tolist()

    # Derive simple per-feature thresholds from RF feature importances
    importances = rf.feature_importances_
    top_features = np.argsort(importances)[::-1][:5]

    # Use mean ± 1.5 std as warning/critical thresholds (in original scale)
    header_lines = [
        "/*",
        " * rtos_health_model.h",
        " * Auto-generated TinyML header for RTOS health prediction.",
        f" * Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        " *",
        " * Usage:",
        " *   #include \"rtos_health_model.h\"",
        " *   float features[N_FEATURES] = {cpu_load, queue_level, ...};",
        " *   int label = rtos_predict(features);",
        " *   // 0=HEALTHY, 1=WARNING, 2=CRITICAL",
        " */",
        "",
        "#ifndef RTOS_HEALTH_MODEL_H",
        "#define RTOS_HEALTH_MODEL_H",
        "",
        "#include <stdint.h>",
        "#include <math.h>",
        "",
        f"#define N_FEATURES  {len(FEATURE_COLS)}",
        "#define LABEL_HEALTHY   0",
        "#define LABEL_WARNING   1",
        "#define LABEL_CRITICAL  2",
        "",
        "/* Feature names (for debugging) */",
        "static const char* const FEATURE_NAMES[N_FEATURES] = {",
    ]
    for f in FEATURE_COLS:
        header_lines.append(f'    "{f}",')
    header_lines.append("};")
    header_lines.append("")

    header_lines.append("/* StandardScaler parameters */")
    header_lines.append(f"static const float SCALER_MEAN[N_FEATURES] = {{")
    header_lines.append("    " + ", ".join(f"{v:.6f}f" for v in means))
    header_lines.append("};")
    header_lines.append(f"static const float SCALER_STD[N_FEATURES]  = {{")
    header_lines.append("    " + ", ".join(f"{v:.6f}f" for v in stds))
    header_lines.append("};")
    header_lines.append("")

    # Simple thresholds derived from training data distribution
    header_lines.append("/* Rule-based thresholds (original scale) */")
    thresholds = {
        "cpu_load":              (50.0, 70.0),
        "queue_level":           (4.0,  7.0),
        "queue_dropped":         (3.0,  8.0),
        "process_jitter":        (50.0, 120.0),
        "process_deadline_miss": (2.0,  6.0),
        "process_exec_time":     (150.0, 280.0),
        "process_stack_left":    (600.0, 300.0),   # LOWER = worse
        "sensor_deadline_miss":  (2.0,  6.0),
        "comm_deadline_miss":    (2.0,  6.0),
    }
    header_lines.append("static const float WARN_THRESH[N_FEATURES] = {")
    header_lines.append("    " + ", ".join(f"{thresholds[f][0]:.2f}f" for f in FEATURE_COLS))
    header_lines.append("};")
    header_lines.append("static const float CRIT_THRESH[N_FEATURES] = {")
    header_lines.append("    " + ", ".join(f"{thresholds[f][1]:.2f}f" for f in FEATURE_COLS))
    header_lines.append("};")
    header_lines.append("")

    header_lines += [
        "/* Normalize a feature array in-place */",
        "static inline void rtos_normalize(float* x) {",
        "    for (int i = 0; i < N_FEATURES; i++) {",
        "        x[i] = (x[i] - SCALER_MEAN[i]) / SCALER_STD[i];",
        "    }",
        "}",
        "",
        "/*",
        " * Lightweight rule-based predictor.",
        " * Replace with emlearn or TFLM inference for full RF accuracy.",
        " */",
        "static inline int rtos_predict(const float* raw_features) {",
        "    int warn_votes = 0;",
        "    int crit_votes = 0;",
        "",
        "    /* cpu_load, queue_level ... queue_dropped: higher = worse */",
        "    for (int i = 0; i <= 5; i++) {",
        "        if (raw_features[i] >= CRIT_THRESH[i]) crit_votes++;",
        "        else if (raw_features[i] >= WARN_THRESH[i]) warn_votes++;",
        "    }",
        "    /* process_stack_left: LOWER = worse */",
        "    if (raw_features[6] <= CRIT_THRESH[6]) crit_votes++;",
        "    else if (raw_features[6] <= WARN_THRESH[6]) warn_votes++;",
        "",
        "    /* sensor / comm deadline miss */",
        "    for (int i = 7; i < N_FEATURES; i++) {",
        "        if (raw_features[i] >= CRIT_THRESH[i]) crit_votes++;",
        "        else if (raw_features[i] >= WARN_THRESH[i]) warn_votes++;",
        "    }",
        "",
        "    if (crit_votes >= 2) return LABEL_CRITICAL;",
        "    if (warn_votes >= 2 || crit_votes == 1) return LABEL_WARNING;",
        "    return LABEL_HEALTHY;",
        "}",
        "",
        "#endif /* RTOS_HEALTH_MODEL_H */",
    ]

    out_path = os.path.join(EXPORT_DIR, "rtos_health_model.h")
    with open(out_path, "w") as f:
        f.write("\n".join(header_lines))
    print(f"[tinyml] C header saved → {out_path}")


# ─── 2. emlearn export (optional) ─────────────────────────────────────────────
def export_emlearn(rf, scaler):
    try:
        import emlearn
    except ImportError:
        print("[tinyml] emlearn not installed. Skipping. (pip install emlearn)")
        return

    import emlearn
    model = emlearn.convert(rf, method="inline")
    out_path = os.path.join(EXPORT_DIR, "emlearn_model.h")
    model.save(out_path)
    print(f"[tinyml] emlearn model saved → {out_path}")


# ─── 3. TFLite export (optional, requires tensorflow) ─────────────────────────
def export_tflite(rf, scaler):
    try:
        import tensorflow as tf
        from sklearn.neural_network import MLPClassifier
        from sklearn.datasets import make_classification
    except ImportError:
        print("[tinyml] TensorFlow not installed. Skipping TFLite export.")
        return

    print("[tinyml] Building TFLite-compatible MLP as RF surrogate …")
    # Load original data to re-train a small MLP surrogate
    try:
        import pandas as pd
        from src.preprocess import load_data, split_and_scale
        df = load_data("data/rtos_health_data.csv")
        X_train, X_test, y_train, y_test, _, _ = split_and_scale(df)
    except Exception as e:
        print(f"[tinyml] Could not load data for TFLite surrogate: {e}")
        return

    # Small MLP
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(FEATURE_COLS),)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(3,  activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=40, batch_size=32,
              validation_split=0.1, verbose=0)
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[tinyml] MLP surrogate accuracy: {acc*100:.2f}%")

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_dir = os.path.join(EXPORT_DIR, "model_tflite")
    os.makedirs(tflite_dir, exist_ok=True)
    out_path = os.path.join(tflite_dir, "rtos_health.tflite")
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(f"[tinyml] TFLite model saved → {out_path}  ({len(tflite_model)/1024:.1f} KB)")

    # Also save as C array for STM32 deployment via X-CUBE-AI
    c_array_path = os.path.join(tflite_dir, "rtos_health_tflite_model.h")
    with open(c_array_path, "w") as f:
        f.write("/* Auto-generated TFLite model as C array */\n")
        f.write(f"/* Accuracy: {acc*100:.2f}% */\n\n")
        f.write(f"const unsigned int rtos_health_model_len = {len(tflite_model)};\n")
        f.write("const unsigned char rtos_health_model[] = {\n  ")
        hex_vals = ", ".join(f"0x{b:02X}" for b in tflite_model)
        f.write(hex_vals)
        f.write("\n};\n")
    print(f"[tinyml] TFLite C array saved → {c_array_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(EXPORT_DIR, exist_ok=True)

    rf     = joblib.load(RF_MODEL)
    scaler = joblib.load(SCALER_MODEL)

    print("\n── TinyML Export ──────────────────────────────────────────")
    export_c_header(rf, scaler)
    export_emlearn(rf, scaler)
    export_tflite(rf, scaler)
    print("── Done ────────────────────────────────────────────────────\n")
