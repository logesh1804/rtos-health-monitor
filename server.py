import argparse
import json
import os
import threading
import asyncio
import numpy as np
import joblib
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

# ML Model Paths matching your project directory layout
RF_MODEL_PATH = "models/random_forest.pkl"
IF_MODEL_PATH = "models/isolation_forest.pkl"
SCALER_PATH = "models/scaler.pkl"
FEAT_FILE = "models/feature_names.json"

RAW_COLS = [
    "cpu_load", "queue_level", "queue_dropped", "process_jitter",
    "process_deadline_miss", "process_exec_time", "process_stack_left",
    "sensor_deadline_miss", "comm_deadline_miss",
]

LABEL_MAP = {0: "Healthy", 1: "Warning", 2: "Critical"}
connected_clients = set()
connected_clients_lock = threading.Lock()
reports_received_count = 0
live_source_args = None
listener_thread = None


def load_models():
    """ Loads your trained ML parameters safely """
    try:
        rf = joblib.load(RF_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        iso = joblib.load(IF_MODEL_PATH) if os.path.exists(IF_MODEL_PATH) else None
        if os.path.exists(FEAT_FILE):
            with open(FEAT_FILE) as f:
                feat_names = json.load(f)
        else:
            feat_names = RAW_COLS
        print("[ML] Successfully loaded trained Random Forest and Scaler configuration models!")
        return rf, iso, scaler, feat_names
    except Exception as e:
        print(f"[ML Warning] Could not load pickled pipelines ({e}). Operating in template demo simulation.")
        return None, None, None, RAW_COLS


rf, iso, scaler, feat_names = load_models()


def add_derived(r: dict) -> dict:
    r = dict(r)
    r["jitter_ratio"] = r["process_jitter"] / 383.0
    r["total_deadline_miss"] = (r["process_deadline_miss"] + r["sensor_deadline_miss"] + r["comm_deadline_miss"])
    r["queue_pressure"] = r["queue_level"] + r["queue_dropped"] * 2
    r["stack_danger"] = 1.0 / (r["process_stack_left"] + 1)
    r["cpu_jitter_stress"] = r["cpu_load"] * r["process_jitter"] / 1000.0
    return r


def generate_reason_template(raw: dict, risk_pct: float, rf_class: int) -> list:
    """ Computes structured textual diagnostic logic output based on features """
    reasons = []
    if rf_class == 0:
        reasons.append("All RTOS runtime pipelines operating normally within stable real-time constraint limits.")
        return reasons

    if raw["process_stack_left"] < 100:
        reasons.append(f"CRITICAL STACK DEGRADATION: ProcessTask stack space falling low ({int(raw['process_stack_left'])} words remaining). Potential overflow hazard.")
    if isinstance(raw["queue_level"], (int, float)) and (raw["queue_dropped"] > 0 or raw["queue_level"] > 7):
        reasons.append(f"QUEUE PIPELINE CONGESTION: Queue level pressure high. CommTask dropped {int(raw['queue_dropped'])} items due to transmission buffer blocks.")
    elif raw["queue_dropped"] > 0:
        reasons.append(f"QUEUE PIPELINE CONGESTION: CommTask dropped {int(raw['queue_dropped'])} items due to transmission buffer blocks. (queue_level not available via this data source)")
    if raw["process_jitter"] > 250:
        reasons.append(f"TIMING JITTER BREACH: High scheduler execution variance detected ({int(raw['process_jitter'])} ms). Watch for priority inversion.")
    if raw["process_deadline_miss"] > 200 or raw["sensor_deadline_miss"] > 100:
        reasons.append("DEADLINE EXCEEDED: Real-time task loop missed strict execution window configurations.")
    if raw["cpu_load"] > 80:
        reasons.append(f"RESOURCE STARVATION: High CPU utilization load processing profile ({int(raw['cpu_load'])}%).")

    if not reasons:
        reasons.append("Multi-variable configuration pattern profile flagged by Random Forest model matching failure signatures.")
    return reasons


def _task_status_from_risk(risk_score: float) -> str:
    """Maps a 0-100 per-task risk score onto the same 3-class scheme as the overall model."""
    if risk_score >= 60:
        return "Critical"
    if risk_score >= 25:
        return "Warning"
    return "Healthy"


def generate_task_breakdown(raw: dict, rf_class: int) -> dict:
    """
    Rule-based per-task health breakdown (SensorTask / CommTask / ProcessTask).

    This is an additive reasoning layer on top of the existing single
    Random Forest verdict. It does NOT retrain or replace the model.
    It inspects the same raw Health fields already available and applies
    simple, explainable thresholds per task, matching the report format:

        Task Sensor_Task: Status / Risk
        Task Comm_Task:   Status / Risk / Reason
        Task Processing_Task: Status / Risk / Reason / Possible Failure

    Keys match what index.html's JS expects: pred.task_breakdown with
    Sensor_Task / Comm_Task / Processing_Task, each holding
    status / risk / reason / possible_failure.
    """
    queue_level_numeric = raw["queue_level"] if isinstance(raw["queue_level"], (int, float)) else None

    sensor_risk = 0
    sensor_reason = "Sampling and queue submission within expected timing bounds."
    sensor_failure = "None"

    if raw["sensor_deadline_miss"] > 100:
        sensor_risk += 50
        sensor_reason = "Deadline miss count elevated for 1000 ms sampling period."
        sensor_failure = "Missed sensor sampling window"
    elif raw["sensor_deadline_miss"] > 30:
        sensor_risk += 25
        sensor_reason = "Deadline miss count creeping upward."

    if raw["process_jitter"] > 250:
        sensor_risk += 15

    sensor_status = _task_status_from_risk(sensor_risk)

    comm_risk = 0
    comm_reason = "Queue wait time and transmission load nominal."
    comm_failure = "None"

    if raw["queue_dropped"] > 0:
        comm_risk += 40
        comm_reason = f"CommTask dropped {int(raw['queue_dropped'])} queued item(s) due to buffer pressure."
        comm_failure = "Queue overflow / message loss"
    if queue_level_numeric is not None and queue_level_numeric > 7:
        comm_risk += 30
        comm_reason = "Queue wait time increasing - buffer nearing capacity."
        if comm_failure == "None":
            comm_failure = "Queue saturation risk"
    if raw["comm_deadline_miss"] > 100:
        comm_risk += 20
        comm_reason = "Deadline miss count elevated for 2000 ms comm period."

    comm_status = _task_status_from_risk(comm_risk)

    process_risk = 0
    process_reason = "Execution time and stack headroom within safe limits."
    process_failure = "None"

    if raw["process_stack_left"] < 100:
        process_risk += 55
        process_reason = "Stack remaining critically low for ProcessTask."
        process_failure = "Stack overflow"
    elif raw["process_stack_left"] < 200:
        process_risk += 25
        process_reason = "Stack remaining decreasing."

    if raw["process_deadline_miss"] > 200:
        process_risk += 30
        process_reason = (process_reason + " Execution time also exceeding deadline window."
                           if process_risk > 30 else
                           "Execution time exceeding the 600 ms deadline window.")
        if process_failure == "None":
            process_failure = "Deadline miss"
        else:
            process_failure = "Stack overflow / deadline miss"
    elif raw["process_deadline_miss"] > 80:
        process_risk += 15

    process_status = _task_status_from_risk(process_risk)

    if rf_class == 2 and process_status == "Healthy" and sensor_status == "Healthy" and comm_status == "Healthy":
        process_status = "Warning"
        if process_reason == "Execution time and stack headroom within safe limits.":
            process_reason = "Random Forest model flagged overall Critical risk from combined feature pattern."

    return {
        "Sensor_Task": {
            "status": sensor_status,
            "risk": sensor_status,
            "reason": sensor_reason,
            "possible_failure": sensor_failure,
        },
        "Comm_Task": {
            "status": comm_status,
            "risk": comm_status,
            "reason": comm_reason,
            "possible_failure": comm_failure,
        },
        "Processing_Task": {
            "status": process_status,
            "risk": process_status,
            "reason": process_reason,
            "possible_failure": process_failure,
        },
    }


def predict_sample(raw: dict) -> dict:
    global reports_received_count
    reports_received_count += 1

    model_input = dict(raw)
    # queue_level is now always numeric (read from Health.QueueLevel in firmware)
    if not isinstance(model_input.get("queue_level"), (int, float)):
        model_input["queue_level"] = 0  # defensive fallback for UART path only

    sample = add_derived(model_input)

    if rf and scaler:
        x = np.array([[sample[f] for f in feat_names]])
        x_sc = scaler.transform(x)
        rf_class = int(rf.predict(x_sc)[0])
        rf_proba = rf.predict_proba(x_sc)[0]
        risk_pct = round(float(1 - rf_proba[0]) * 100, 1)
        rf_label = LABEL_MAP[rf_class]
        rf_conf = round(float(rf_proba[rf_class]) * 100, 1)
    else:
        risk_pct = 4.0 if raw['cpu_load'] < 20 else (52.0 if raw['cpu_load'] < 75 else 91.4)
        rf_class = 0 if risk_pct < 15 else (1 if risk_pct < 60 else 2)
        rf_label = LABEL_MAP[rf_class]
        rf_conf = 94.5

    reasons = generate_reason_template(raw, risk_pct, rf_class)
    task_breakdown = generate_task_breakdown(raw, rf_class)

    return {
        "raw": raw,
        "prediction": {
            "rf_label": rf_label,
            "rf_class": rf_class,
            "rf_confidence": rf_conf,
            "risk_pct": risk_pct,
            "reports_received": reports_received_count,
            "reasons": reasons,
            "task_breakdown": task_breakdown
        }
    }


async def broadcast_payload(data: dict):
    payload = json.dumps(data)
    with connected_clients_lock:
        clients = list(connected_clients)

    print(f"[WebSocket] Broadcasting to {len(clients)} clients")
    if not clients:
        return

    send_results = await asyncio.gather(
        *[client.send_text(payload) for client in clients],
        return_exceptions=True,
    )

    failed_clients = []
    for client, result in zip(clients, send_results):
        if isinstance(result, Exception):
            print(f"[WebSocket] Payload failed: {result}")
            failed_clients.append(client)
        else:
            print("[WebSocket] Payload sent")

    if failed_clients:
        with connected_clients_lock:
            for client in failed_clients:
                connected_clients.discard(client)


def _log_broadcast_future(future):
    try:
        future.result()
    except Exception as e:
        print(f"[WebSocket] Payload failed: {e}")


def serial_listener(port, baud, loop):
    import serial
    print(f"[UART Receiver Task] Active. Monitoring live telemetry streaming from {port}...")
    try:
        with serial.Serial(port, baud, timeout=3) as ser:
            while True:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 9:
                    continue
                try:
                    vals = [float(p) for p in parts[:9]]
                    raw = dict(zip(RAW_COLS, vals))

                    result = predict_sample(raw)
                    print(
                        f"[UART] Prediction generated: "
                        f"{result['prediction']['rf_label']} "
                        f"({result['prediction']['rf_confidence']}%)"
                    )

                    future = asyncio.run_coroutine_threadsafe(broadcast_payload(result), loop)
                    future.add_done_callback(_log_broadcast_future)
                except ValueError:
                    continue
    except Exception as e:
        print(f"\n[UART Error] Port access failed: {e}. Dashboard running in simulator validation mode.\n")


OPENOCD_HEALTH_FIELD_ORDER = [
    "CpuLoad", "ProcessExecTime", "ProcessJitter", "QueueDropped",
    "ProcessDeadlineMiss", "SensorDeadlineMiss", "CommDeadlineMiss",
    "ProcessStackLeft", "QueueLevel",   # 9th word — added in main.c V2.2
]
OPENOCD_DEFAULT_ADDRESS = 0x2000007C
OPENOCD_DEFAULT_WORDS = 9              # was 8 — now reads QueueLevel too


def _openocd_read_until_prompt(sock, timeout=5.0, debug=False):
    buf = b""
    sock.settimeout(timeout)
    while True:
        chunk = sock.recv(4096)
        if debug:
            print(f"    [trace] received {len(chunk)} bytes: {chunk!r}")
        if not chunk:
            raise ConnectionError("OpenOCD closed the connection")
        buf += chunk
        if b"> " in buf:
            break
    return buf


def _openocd_send_command(sock, command, timeout=5.0, debug=False):
    if debug:
        print(f"    [trace] sending: {command!r}")
    sock.sendall((command + "\n").encode("utf-8"))
    raw = _openocd_read_until_prompt(sock, timeout, debug=debug)
    text = raw.decode("utf-8", errors="ignore")
    text = text.replace(command, "", 1).replace("> ", "")
    return text.strip()


def _openocd_parse_mdw(response, expected_count):
    values = []
    for line in response.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        _, _, hex_part = line.partition(":")
        for token in hex_part.strip().split():
            try:
                values.append(int(token, 16))
            except ValueError:
                continue
    if len(values) < expected_count:
        raise ConnectionError(
            f"Health structure length mismatch: expected {expected_count} words, got {len(values)}"
        )
    return values[:expected_count]


def openocd_listener(host, port, address, words, period, loop):
    """
    Live dashboard feed sourced from OpenOCD instead of UART.
    Reads the RTOS_Health_t struct directly from target RAM and feeds
    the same prediction/broadcast pipeline serial_listener() uses.
    """
    import socket
    import time

    print(f"[OpenOCD Receiver Task] Active. Monitoring live telemetry from {host}:{port} "
          f"@ 0x{address:08X} ({words} words)...")

    while True:
        sock = None
        try:
            print("[trace] connecting socket...")
            sock = socket.create_connection((host, port), timeout=5.0)
            print("[trace] socket connected, draining banner...")
            _openocd_read_until_prompt(sock, debug=True)
            print("[OpenOCD Receiver Task] Connected.")

            while True:
                response = _openocd_send_command(sock, f"mdw 0x{address:08X} {words}", debug=True)
                decoded = _openocd_parse_mdw(response, expected_count=words)
                health = dict(zip(OPENOCD_HEALTH_FIELD_ORDER, decoded))

                raw = {
                    "cpu_load": health["CpuLoad"],
                    "queue_level": health["QueueLevel"],   # live from Health struct
                    "queue_dropped": health["QueueDropped"],
                    "process_jitter": health["ProcessJitter"],
                    "process_deadline_miss": health["ProcessDeadlineMiss"],
                    "process_exec_time": health["ProcessExecTime"],
                    "process_stack_left": health["ProcessStackLeft"],
                    "sensor_deadline_miss": health["SensorDeadlineMiss"],
                    "comm_deadline_miss": health["CommDeadlineMiss"],
                }

                result = predict_sample(raw)
                pred   = result["prediction"]
                tb     = pred["task_breakdown"]

                with connected_clients_lock:
                    client_count = len(connected_clients)

                # Queue dropped status
                qd = raw["queue_dropped"]
                if qd >= 5:
                    qd_status = "Critical"
                elif qd > 0:
                    qd_status = "Warning"
                else:
                    qd_status = "Healthy"

                # ── RTOS HEALTH REPORT (terminal) ─────────────────────────
                print("=" * 60)
                print("  RTOS HEALTH REPORT")
                print("=" * 60)

                print("  Task Sensor_Task:")
                print(f"    Status          : {tb['Sensor_Task']['status']}")
                print(f"    Risk            : {tb['Sensor_Task']['risk']}")
                print(f"    Reason          : {tb['Sensor_Task']['reason']}")
                if tb['Sensor_Task']['possible_failure'] != "None":
                    print(f"    Possible Failure: {tb['Sensor_Task']['possible_failure']}")
                print(f"    Deadline Misses : {int(raw['sensor_deadline_miss'])}")
                print(f"    Jitter          : {int(raw['process_jitter'])} ms")

                print("  Task Comm_Task:")
                print(f"    Status          : {tb['Comm_Task']['status']}")
                print(f"    Risk            : {tb['Comm_Task']['risk']}")
                print(f"    Reason          : {tb['Comm_Task']['reason']}")
                if tb['Comm_Task']['possible_failure'] != "None":
                    print(f"    Possible Failure: {tb['Comm_Task']['possible_failure']}")
                print(f"    Deadline Misses : {int(raw['comm_deadline_miss'])}")
                print(f"    Queue Level     : {int(raw['queue_level'])}/10  (live)")
                print(f"    Queue Dropped   : {int(qd)}  [{qd_status}]")

                print("  Task Processing_Task:")
                print(f"    Status          : {tb['Processing_Task']['status']}")
                print(f"    Risk            : {tb['Processing_Task']['risk']}")
                print(f"    Reason          : {tb['Processing_Task']['reason']}")
                if tb['Processing_Task']['possible_failure'] != "None":
                    print(f"    Possible Failure: {tb['Processing_Task']['possible_failure']}")
                print(f"    Exec Time       : {int(raw['process_exec_time'])} ms")
                print(f"    Stack Left      : {int(raw['process_stack_left'])} words")
                print(f"    Deadline Misses : {int(raw['process_deadline_miss'])}")

                print(f"  Overall  : {pred['rf_label']}  "
                      f"({pred['rf_confidence']}% confidence)  |  "
                      f"CPU: {int(raw['cpu_load'])}%  |  "
                      f"Risk: {pred['risk_pct']}%  |  "
                      f"clients={client_count}")
                print("-" * 60)

                future = asyncio.run_coroutine_threadsafe(broadcast_payload(result), loop)
                future.add_done_callback(_log_broadcast_future)

                time.sleep(period)

        except Exception as e:
            print(f"\n[OpenOCD Error] {e}. Retrying in 2s...\n")
            if sock:
                try:
                    sock.close()
                except OSError:
                    pass
            time.sleep(2.0)


@app.on_event("startup")
async def start_live_listener():
    """
    FIX: Start the UART/OpenOCD reader after FastAPI is running so the
    background thread receives Uvicorn's real asyncio event loop.
    """
    global listener_thread

    if listener_thread and listener_thread.is_alive():
        return

    if live_source_args is None:
        return

    loop = asyncio.get_running_loop()
    args = live_source_args

    if args.source == "openocd":
        ocd_address = int(args.ocd_address, 16) if args.ocd_address.lower().startswith("0x") else int(args.ocd_address)
        listener_thread = threading.Thread(
            target=openocd_listener,
            args=(args.ocd_host, args.ocd_port, ocd_address, args.ocd_words, args.ocd_period, loop),
            daemon=True,
        )
    else:
        listener_thread = threading.Thread(
            target=serial_listener,
            args=(args.uart, args.baud, loop),
            daemon=True,
        )

    print(f"[FastAPI] Starting {args.source.upper()} listener on Uvicorn event loop")
    listener_thread.start()


@app.get("/")
async def get():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    with connected_clients_lock:
        connected_clients.add(websocket)
        client_count = len(connected_clients)

    print(f"[WebSocket] Browser connected. clients={client_count}")

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        with connected_clients_lock:
            connected_clients.discard(websocket)
            client_count = len(connected_clients)

        print(f"[WebSocket] Browser disconnected. clients={client_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["uart", "openocd"], default="uart",
                        help="Live data source for the dashboard (default: uart)")
    parser.add_argument("--uart", default="COM15")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--ocd-host", default="localhost", help="OpenOCD telnet host (--source openocd)")
    parser.add_argument("--ocd-port", type=int, default=4444, help="OpenOCD telnet port (--source openocd)")
    parser.add_argument("--ocd-address", default="0x2000007C", help="Health struct address (--source openocd)")
    parser.add_argument("--ocd-words", type=int, default=8, help="Health struct word count (--source openocd)")
    parser.add_argument("--ocd-period", type=float, default=1.0, help="Sampling period in seconds (--source openocd)")
    args = parser.parse_args()

    # FIX: The live listener is started in FastAPI startup, where
    # asyncio.get_running_loop() returns Uvicorn's active event loop.
    live_source_args = args

    uvicorn.run(app, host="0.0.0.0", port=8000)