"""
capture_uart.py
---------------
Captures real UART data from STM32 into a CSV file for retraining.

Usage:
    python capture_uart.py --port COM13 --samples 100

After capturing, open data/real_capture.csv, add the health_label column
(0=Healthy, 1=Warning, 2=Critical), then retrain:
    python train_pipeline.py --csv data/real_capture_labelled.csv
"""

import argparse
import csv
import os
import sys
import time

FEATURE_COLS = [
    "cpu_load", "queue_level", "queue_dropped", "process_jitter",
    "process_deadline_miss", "process_exec_time", "process_stack_left",
    "sensor_deadline_miss", "comm_deadline_miss",
]

OUT_FILE = "data/real_capture.csv"


def capture(port: str, baud: int, n_samples: int):
    try:
        import serial
    except ImportError:
        print("pyserial not installed. Run: pip install pyserial")
        sys.exit(1)

    os.makedirs("data", exist_ok=True)

    print(f"\n[capture] Opening {port} @ {baud} baud …")
    print(f"[capture] Collecting {n_samples} samples → {OUT_FILE}")
    print("[capture] Press Ctrl-C to stop early\n")

    collected = 0

    with open(OUT_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write header — NO health_label yet (you add it manually after)
        writer.writerow(FEATURE_COLS + ["health_label"])

        with serial.Serial(port, baud, timeout=5) as ser:
            while collected < n_samples:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                parts = line.split(",")
                if len(parts) < 9:
                    print(f"[capture] Skipping malformed line: {line!r}")
                    continue

                try:
                    vals = [float(p) for p in parts[:9]]
                except ValueError:
                    print(f"[capture] Skipping non-numeric line: {line!r}")
                    continue

                # Write row with empty label — fill in manually afterwards
                writer.writerow(vals + [""])
                csvfile.flush()
                collected += 1

                # Print live summary
                print(f"  [{collected:>3}/{n_samples}]  "
                      f"cpu={vals[0]:.0f}%  "
                      f"jitter={vals[3]:.0f}ms  "
                      f"dl_miss={vals[4]:.0f}  "
                      f"stack={vals[6]:.0f}")

    print(f"\n[capture] Done. {collected} rows saved to {OUT_FILE}")
    print("\n── Next steps ──────────────────────────────────────────────")
    print("1. Open data/real_capture.csv in Excel or VS Code")
    print("2. Fill in the 'health_label' column for each row:")
    print("     0 = Healthy   (system running normally)")
    print("     1 = Warning   (elevated but not failing)")
    print("     2 = Critical  (faults / overload / near failure)")
    print("3. Save as: data/real_capture_labelled.csv")
    print("4. Retrain: python train_pipeline.py --csv data/real_capture_labelled.csv")
    print("────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture STM32 UART data to CSV")
    parser.add_argument("--port",    default="COM13",  help="Serial port (e.g. COM13)")
    parser.add_argument("--baud",    type=int, default=115200)
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of samples to collect (default: 100)")
    args = parser.parse_args()
    capture(args.port, args.baud, args.samples)
    