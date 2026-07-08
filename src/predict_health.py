import argparse
import json
import sys
import os
import threading
import asyncio
import numpy as np
import joblib
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

RF_MODEL_PATH = "models/random_forest.pkl"
IF_MODEL_PATH = "models/isolation_forest.pkl"
SCALER_PATH   = "models/scaler.pkl"
FEAT_FILE     = "models/feature_names.json"

RAW_COLS = [
    "cpu_load", "queue_level", "queue_dropped", "process_jitter",
    "process_deadline_miss", "process_exec_time", "process_stack_left",
    "sensor_deadline_miss", "comm_deadline_miss",
]

LABEL_MAP = {0: "Healthy", 1: "Warning", 2: "Critical"}

# --- THREAD-SAFE SHARED STATE ---
connected_clients = set()
latest_telemetry_payload = None  # Holds the latest processed dataset
state_lock = threading.Lock()

def load_models():
    try:
        rf     = joblib.load(RF_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        iso    = joblib.load(IF_MODEL_PATH) if os.path.exists(IF_MODEL_PATH) else None
        if os.path.exists(FEAT_FILE):
            with open(FEAT_FILE) as f: feat_names = json.load(f)
        else:
            feat_names = RAW_COLS
        return rf, iso, scaler, feat_names
    except Exception:
        return None, None, None, RAW_COLS

rf, iso, scaler, feat_names = load_models()

def add_derived(r: dict) -> dict:
    r = dict(r)
    r["jitter_ratio"]        = r["process_jitter"] / 383.0
    r["total_deadline_miss"] = (r["process_deadline_miss"] + r["sensor_deadline_miss"] + r["comm_deadline_miss"])
    r["queue_pressure"]      = r["queue_level"] + r["queue_dropped"] * 2
    r["stack_danger"]        = 1.0 / (r["process_stack_left"] + 1)
    r["cpu_jitter_stress"]   = r["cpu_load"] * r["process_jitter"] / 1000.0
    return r

def compute_per_task_status(raw: dict) -> dict:
    sensor_status, sensor_risk, sensor_reason = "Healthy", "Low", "Operating normally"
    if raw["sensor_deadline_miss"] > 50 or raw["process_jitter"] > 200:
        sensor_status, sensor_risk, sensor_reason = "Warning", "Medium", "Jitter variations shifting scheduling offsets"

    comm_status, comm_risk, comm_reason = "Healthy", "Low", "Queue levels stable"
    if raw["queue_level"] > 6 or raw["queue_dropped"] > 0:
        comm_status, comm_risk, comm_reason = "Warning", "Medium", "Queue wait time increasing"
    if raw["queue_dropped"] > 5:
        comm_status, comm_risk, comm_reason = "Critical", "High", "Queue frame dropped buffers full"

    proc_status, proc_risk, proc_reason, proc_fail = "Healthy", "Low", "Execution window safe", "None"
    if raw["process_stack_left"] < 150 or raw["process_exec_time"] > 250:
        if raw["process_stack_left"] < 80 or raw["process_deadline_miss"] > 100:
            proc_status, proc_risk, proc_reason = "Critical", "High", "Stack remaining decreasing + execution time increasing"
            proc_fail = "Stack overflow / deadline miss"
        else:
            proc_status, proc_risk, proc_reason = "Warning", "Medium", "Stack sizing compression bounds observed"

    return {
        "Sensor_Task": {"status": sensor_status, "risk": sensor_risk, "reason": sensor_reason},
        "Comm_Task": {"status": comm_status, "risk": comm_risk, "reason": comm_reason},
        "Processing_Task": {"status": proc_status, "risk": proc_risk, "reason": proc_reason, "possible_failure": proc_fail}
    }

def print_custom_terminal_report(report_idx: int, task_data: dict):
    print("\n" + "="*40)
    print(f"RTOS HEALTH REPORT [Sequence #{report_idx}]")
    print("="*40)
    print(f"Task Sensor_Task:\n  Status: {task_data['Sensor_Task']['status']}\n  Risk: {task_data['Sensor_Task']['risk']}")
    print(f"\nTask Comm_Task:\n  Status: {task_data['Comm_Task']['status']}\n  Reason: {task_data['Comm_Task']['reason']}\n  Predicted Risk: {task_data['Comm_Task']['risk']}")
    print(f"\nTask Processing_Task:\n  Status: {task_data['Processing_Task']['status']}")
    print(f"  Reason: {task_data['Processing_Task']['reason']}\n  Predicted Risk: {task_data['Processing_Task']['risk']}")
    if task_data['Processing_Task']['status'] == "Critical":
        print(f"  Possible Failure: {task_data['Processing_Task']['possible_failure']}")
    print("="*40 + "\n")

def predict_sample_internal(raw: dict) -> dict:
    sample = add_derived(raw)
    if rf and scaler:
        x = np.array([[sample[f] for f in feat_names]])
        x_sc = scaler.transform(x)
        rf_class = int(rf.predict(x_sc)[0])
        rf_proba = rf.predict_proba(x_sc)[0]
        risk_pct = round(float(1 - rf_proba[0]) * 100, 1)
        rf_label = LABEL_MAP[rf_class]
    else:
        risk_pct = 4.0 if raw['process_stack_left'] > 100 else 92.5
        rf_class = 0 if risk_pct < 15 else 2
        rf_label = LABEL_MAP[rf_class]

    return {
        "raw": raw,
        "prediction": {
            "rf_label": rf_label,
            "rf_class": rf_class,
            "risk_pct": risk_pct,
            "task_breakdown": compute_per_task_status(raw)
        }
    }

def run_serial_stream_loop(port: str, baud: int):
    """ Reads UART data independent of any async loop constraints """
    global latest_telemetry_payload
    import serial
    print(f"[UART Receiver] Listening for microcontrollers on {port} @ {baud}...")
    report_count = 0
    
    try:
        with serial.Serial(port, baud, timeout=3) as ser:
            while True:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line: continue
                parts = line.split(",")
                if len(parts) < 9: continue
                try:
                    vals = [float(p) for p in parts[:9]]
                    raw = dict(zip(RAW_COLS, vals))
                    
                    payload_results = predict_sample_internal(raw)
                    report_count += 1
                    
                    # Print to terminal instantly
                    print_custom_terminal_report(report_count, payload_results["prediction"]["task_breakdown"])
                    
                    # Update global payload safely using a thread lock
                    with state_lock:
                        latest_telemetry_payload = payload_results
                except ValueError:
                    continue
    except Exception as e:
        print(f"[Serial Offline Error] {e}")

# --- FASTAPI ACTIVE BROADCASTER LOOP ---
async def web_broadcast_worker():
    """ Runs completely on Uvicorn's internal loop, polling for fresh data updates """
    global latest_telemetry_payload
    last_sent_payload = None
    while True:
        await asyncio.sleep(0.05)  # Quick 50ms check interval
        if latest_telemetry_payload is not None and latest_telemetry_payload != last_sent_payload:
            with state_lock:
                current_payload = latest_telemetry_payload
                last_sent_payload = current_payload
                
            if connected_clients:
                msg = json.dumps(current_payload)
                await asyncio.gather(*[client.send_text(msg) for client in connected_clients], return_exceptions=True)

@app.on_event("startup")
async def startup_event():
    # Fire off our broadcaster worker straight inside Uvicorn's active loop
    asyncio.create_task(web_broadcast_worker())

@app.get("/")
async def get_dashboard_html():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connected_clients.remove(websocket)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RTOS Web & Console Sync Telemetry Predictor Engine")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--uart", metavar="PORT", help="Live UART port e.g. COM15")
    group.add_argument("--csv",  metavar="FILE", help="Batch CSV execution source track")
    parser.add_argument("--baud", type=int, default=115200)
    args = parser.parse_args()

    if args.uart:
        # Start pure background worker thread for data processing
        serial_worker = threading.Thread(target=run_serial_stream_loop, args=(args.uart, args.baud), daemon=True)
        serial_worker.start()
        
        # Start web interface engine
        uvicorn.run(app, host="0.0.0.0", port=8000)