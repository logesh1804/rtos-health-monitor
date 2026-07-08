"""
train_pipeline.py
-----------------
Master script: runs the full training pipeline in one command.

  python train_pipeline.py [--csv PATH]

Steps
-----
1. Generate synthetic dataset (if CSV not found)
2. Preprocess
3. Train Random Forest
4. Train Isolation Forest
5. Generate all plots
6. Export TinyML artefacts
"""

import argparse
import os
import sys

# Allow running from project root without installing the package
sys.path.insert(0, os.path.dirname(__file__))

CSV_DEFAULT = "data/rtos_health_data.csv"


def banner(text: str):
    width = 60
    print(f"\n{'─'*width}")
    print(f"  {text}")
    print(f"{'─'*width}")


def main(csv_path: str):
    # ── Step 0: dataset ──────────────────────────────────────────────────────
    if not os.path.exists(csv_path):
        banner("Step 0 – Generating synthetic dataset")
        import importlib.util, runpy
        runpy.run_path("data/generate_dataset.py")

    # ── Step 1: Random Forest ────────────────────────────────────────────────
    banner("Step 1 – Training Random Forest Classifier")
    from src.train_random_forest import train as train_rf
    rf, feat_names, rf_acc = train_rf(csv_path)

    # ── Step 2: Isolation Forest ─────────────────────────────────────────────
    banner("Step 2 – Training Isolation Forest Anomaly Detector")
    from src.train_isolation_forest import train as train_if
    iso = train_if(csv_path)

    # ── Step 3: Plots ────────────────────────────────────────────────────────
    banner("Step 3 – Generating plots")
    from src.plot_features import (
        plot_feature_importance,
        plot_correlation_matrix,
        plot_label_distribution,
        plot_boxplots,
    )
    plot_feature_importance(rf, feat_names)
    plot_correlation_matrix(csv_path)
    plot_label_distribution(csv_path)
    plot_boxplots(csv_path)

    # ── Step 4: TinyML export ────────────────────────────────────────────────
    banner("Step 4 – Exporting TinyML artefacts")
    from src.export_tinyml import export_c_header, export_emlearn
    import joblib
    scaler = joblib.load("models/scaler.pkl")
    os.makedirs("tinyml_export", exist_ok=True)
    export_c_header(rf, scaler)
    export_emlearn(rf, scaler)

    # ── Summary ──────────────────────────────────────────────────────────────
    banner("✓ Pipeline complete")
    print(f"  Random Forest accuracy : {rf_acc*100:.2f}%")
    print(f"  Models saved in        : models/")
    print(f"  Plots saved in         : plots/")
    print(f"  TinyML artefacts in    : tinyml_export/")
    print()
    print("  Run a single prediction:")
    print('  python src/predict_health.py --single \'{"cpu_load":75,"queue_level":9,')
    print('    "queue_dropped":14,"process_jitter":200,"process_deadline_miss":8,')
    print('    "process_exec_time":350,"process_stack_left":180,')
    print('    "sensor_deadline_miss":6,"comm_deadline_miss":7}\'')
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RTOS Health Predictor – Training Pipeline")
    parser.add_argument("--csv", default=CSV_DEFAULT,
                        help=f"Path to dataset CSV (default: {CSV_DEFAULT})")
    args = parser.parse_args()
    main(args.csv)
