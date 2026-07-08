"""
predict_single.py  (updated for real hardware + derived features)
Edit the values below and run:  python predict_single.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import joblib

RF_MODEL  = "models/random_forest.pkl"
SCALER    = "models/scaler.pkl"
FEAT_FILE = "models/feature_names.json"

LABEL_MAP    = {0: "Healthy",  1: "Warning",  2: "Critical"}
LABEL_COLORS = {0: "\033[92m", 1: "\033[93m", 2: "\033[91m"}
RESET        = "\033[0m"

# ── Edit your STM32 readings here ─────────────────────────────────────────────
raw = {
    "cpu_load"             : 5,
    "queue_level"          : 0,
    "queue_dropped"        : 0,
    "process_jitter"       : 382,
    "process_deadline_miss": 10,
    "process_exec_time"    : 382,
    "process_stack_left"   : 384,
    "sensor_deadline_miss" : 5,
    "comm_deadline_miss"   : 4,
}
# ──────────────────────────────────────────────────────────────────────────────

def add_derived(r):
    r = dict(r)
    r["jitter_ratio"]        = r["process_jitter"] / 383.0
    r["total_deadline_miss"] = (r["process_deadline_miss"] +
                                 r["sensor_deadline_miss"] +
                                 r["comm_deadline_miss"])
    r["queue_pressure"]      = r["queue_level"] + r["queue_dropped"] * 2
    r["stack_danger"]        = 1.0 / (r["process_stack_left"] + 1)
    r["cpu_jitter_stress"]   = r["cpu_load"] * r["process_jitter"] / 1000.0
    return r

rf     = joblib.load(RF_MODEL)
scaler = joblib.load(SCALER)
with open(FEAT_FILE) as f:
    feat_names = json.load(f)

sample = add_derived(raw)
x      = np.array([[sample[feat] for feat in feat_names]])
x_sc   = scaler.transform(x)

rf_class = int(rf.predict(x_sc)[0])
rf_proba = rf.predict_proba(x_sc)[0]
risk_pct = round(float(1 - rf_proba[0]) * 100, 1)
color    = LABEL_COLORS[rf_class]
label    = LABEL_MAP[rf_class]
border   = "=" * 52

print()
print("+" + border + "+")
print("|" + "  RTOS RUNTIME HEALTH REPORT".center(52) + "|")
print("+" + border + "+")
print(f"|  {'Status':<20} {color}{label:<29}{RESET}|")
print(f"|  {'RF Confidence':<20} {rf_proba[rf_class]*100:>5.1f}%{'':24}|")
print(f"|  {'Risk Score':<20} {risk_pct:>5.1f}%{'':24}|")
print("+" + border + "+")
print("|  Probability breakdown:                                    |")
for lbl, pct in zip(["Healthy","Warning","Critical"], rf_proba * 100):
    bar = chr(9608) * int(pct / 5)
    print(f"|    {lbl:<10} {pct:>5.1f}%  {bar:<17}|")
print("+" + border + "+")
print("|  Raw input metrics:                                        |")
for k, v in raw.items():
    print(f"|    {k:<28} {str(v):>20}|")
print("+" + border + "+")
print()
