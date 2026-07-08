"""
generate_dataset.py
-------------------
Generates a synthetic but realistic RTOS health dataset
based on the STM32F411 + FreeRTOS system described in main.c.

Run this if you don't yet have real hardware data:
    python data/generate_dataset.py

Output: data/rtos_health_data.csv
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N_HEALTHY  = 400
N_WARNING  = 300
N_CRITICAL = 200

def healthy():
    return dict(
        cpu_load             = rng.uniform(20, 45),
        queue_level          = rng.integers(0, 3),
        queue_dropped        = rng.integers(0, 2),
        process_jitter       = rng.uniform(0, 30),
        process_deadline_miss= rng.integers(0, 1),
        process_exec_time    = rng.uniform(50, 120),
        process_stack_left   = rng.integers(900, 2048),
        sensor_deadline_miss = rng.integers(0, 1),
        comm_deadline_miss   = rng.integers(0, 1),
        health_label         = 0,
    )

def warning():
    return dict(
        cpu_load             = rng.uniform(45, 70),
        queue_level          = rng.integers(3, 7),
        queue_dropped        = rng.integers(2, 8),
        process_jitter       = rng.uniform(30, 100),
        process_deadline_miss= rng.integers(1, 5),
        process_exec_time    = rng.uniform(120, 250),
        process_stack_left   = rng.integers(400, 900),
        sensor_deadline_miss = rng.integers(1, 4),
        comm_deadline_miss   = rng.integers(1, 4),
        health_label         = 1,
    )

def critical():
    return dict(
        cpu_load             = rng.uniform(70, 99),
        queue_level          = rng.integers(7, 11),
        queue_dropped        = rng.integers(8, 30),
        process_jitter       = rng.uniform(100, 400),
        process_deadline_miss= rng.integers(5, 20),
        process_exec_time    = rng.uniform(250, 600),
        process_stack_left   = rng.integers(40, 400),
        sensor_deadline_miss = rng.integers(4, 15),
        comm_deadline_miss   = rng.integers(4, 15),
        health_label         = 2,
    )

rows = (
    [healthy()  for _ in range(N_HEALTHY)]  +
    [warning()  for _ in range(N_WARNING)]  +
    [critical() for _ in range(N_CRITICAL)]
)

df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
df = df.round(2)

out = "data/rtos_health_data.csv"
df.to_csv(out, index=False)
print(f"Dataset saved → {out}  ({len(df)} rows)")
print(df["health_label"].value_counts().sort_index().rename({0:"Healthy",1:"Warning",2:"Critical"}))
