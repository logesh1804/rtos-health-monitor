"""
label_data.py
-------------
Automatically labels real_capture.csv based on metric thresholds.
No Excel needed. Run after each capture session.

Usage:
    python label_data.py --input data/real_capture.csv --label 0
    python label_data.py --input data/real_capture.csv --label 1
    python label_data.py --input data/real_capture.csv --label 2

All rows in the input file get the label you specify.
Results are APPENDED to data/real_capture_labelled.csv
"""

import argparse
import os
import pandas as pd

OUT_FILE = "data/real_capture_labelled.csv"

LABEL_NAMES = {
    0: "Healthy",
    1: "Warning",
    2: "Critical"
}

FEATURE_COLS = [
    "cpu_load", "queue_level", "queue_dropped", "process_jitter",
    "process_deadline_miss", "process_exec_time", "process_stack_left",
    "sensor_deadline_miss", "comm_deadline_miss",
]


def label_file(input_path: str, label: int):
    if not os.path.exists(input_path):
        print(f"[label] ERROR: File not found: {input_path}")
        return

    df = pd.read_csv(input_path)

    # Keep only feature columns
    df = df[FEATURE_COLS].copy()

    # Assign label
    df["health_label"] = label

    row_count = len(df)

    # Show summary of what we are labelling
    print(f"\n[label] File     : {input_path}")
    print(f"[label] Rows     : {row_count}")
    print(f"[label] Label    : {label} = {LABEL_NAMES[label]}")
    print(f"\n[label] Value summary:")
    print(df[FEATURE_COLS].describe().round(1).to_string())

    # Append to labelled file
    header_needed = not os.path.exists(OUT_FILE)
    df.to_csv(OUT_FILE, mode="a", index=False, header=header_needed)

    print(f"\n[label] ✓ {row_count} rows labelled as {label} ({LABEL_NAMES[label]})")
    print(f"[label] ✓ Appended to {OUT_FILE}")

    # Show current totals
    if os.path.exists(OUT_FILE):
        full = pd.read_csv(OUT_FILE)
        counts = full["health_label"].value_counts().sort_index()
        print(f"\n[label] Dataset totals so far:")
        for lbl, cnt in counts.items():
            bar = "█" * (cnt // 5)
            print(f"         {LABEL_NAMES[int(lbl)]:10} (label={int(lbl)}): {cnt:>4} rows  {bar}")
        print(f"         {'TOTAL':10}          : {len(full):>4} rows")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label captured UART CSV data")
    parser.add_argument("--input",  default="data/real_capture.csv",
                        help="Input CSV file to label")
    parser.add_argument("--label",  type=int, required=True, choices=[0, 1, 2],
                        help="Health label: 0=Healthy  1=Warning  2=Critical")
    args = parser.parse_args()

    label_file(args.input, args.label)