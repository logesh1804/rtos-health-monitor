# AI-Assisted FreeRTOS Task Health Predictor

Runtime health monitoring and failure prediction for FreeRTOS tasks on an STM32F411 microcontroller. Traditional RTOS validation (stack overflow hooks, watchdog timers, deadline checks) only flags problems *after* they happen. This project collects live runtime metrics from FreeRTOS tasks and applies machine learning to predict task health — Healthy / Warning / Critical — **before** a stack overflow, deadline miss, or overload condition actually occurs.

---

## Overview

- **Target hardware:** STM32F411 (FreeRTOS)
- **Tasks monitored:** SensorTask, CommTask, ProcessTask, LoggerTask
- **Data link:** UART @ 115200 baud, CSV health metrics streamed every 3 seconds
- **ML models:** Random Forest classifier (Healthy/Warning/Critical) + Isolation Forest anomaly detector
- **Live visualization:** Local HTTP dashboard with per-task health cards, risk bars, and event log
- **Bonus path:** TinyML export for on-device inference back on the STM32

### Project scope

| Stage | Description |
|---|---|
| **Basic** | Rule-based / threshold RTOS health monitor |
| **Advanced** | AI-assisted trend-based failure prediction (this project) |

---

## How it works

1. **Firmware** on the STM32F411 runs four FreeRTOS tasks and continuously tracks stack high-water mark, execution time, jitter, missed deadlines, queue wait/drops, and CPU load.
2. **LoggerTask** streams this data as a 10-column CSV line over UART every 3 seconds.
3. A Python script on the PC captures and labels this data across six fault scenarios (normal operation plus five injected fault conditions).
4. Five derived features are engineered from the raw metrics to improve model expressiveness.
5. A Random Forest classifier is trained to predict task health state, alongside an Isolation Forest for unsupervised anomaly detection.
6. A local dashboard visualizes live predictions, reasons, and risk levels per task.

---

## Repository structure

```
rtos_health_predictor/
├── src/                        # Core source (firmware interface / shared modules)
├── data/                       # Collected & labeled UART session data
├── models/                     # Trained Random Forest / Isolation Forest models
├── plots/                      # Training/evaluation plots
├── tinyml_export/              # On-device inference export (C header)
│
├── capture_uart.py             # Captures raw UART data from the STM32
├── label_data.py               # Programmatically labels captured sessions
├── fix_and_retrain.py          # Corrects cumulative-counter overlap between sessions, retrains
├── train_pipeline.py           # Full training pipeline (feature engineering → RF + Isolation Forest)
├── predict_single.py           # Run inference on a single sample/session
├── openocd_logger_v2.py        # UART/OpenOCD-based live logger
├── diagnose_openocd.py         # OpenOCD connection diagnostics
├── server.py                   # Local dashboard HTTP server (port 8765)
├── index.html                  # Dashboard frontend
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Hardware setup

| Metric | Source | Purpose |
|---|---|---|
| Stack high-water mark | `uxTaskGetStackHighWaterMark()` | Predict stack overflow risk |
| Task execution time | Tick delta around task body | Detect task overload |
| Task period jitter | Deviation from expected period | Detect scheduling instability |
| Missed deadline count | Period > expected threshold | Detect real-time failure risk |
| Queue wait / drops | `uxQueueMessagesWaiting()` | Detect communication delay/congestion |
| CPU load | Idle task tick counting | Detect system-wide overload |

**Tasks:**
- `SensorTask` — reads sensor data, pushes to queue (1 s period)
- `CommTask` — simulates UART/communication load (2 s period)
- `ProcessTask` — consumes queue, computation-heavy (500 ms period)
- `LoggerTask` — aggregates metrics, transmits CSV over UART (3 s period)

---

## Fault conditions simulated

Six labeled sessions were used for training data, covering both normal operation and five distinct fault types:

| Fault | Injection method |
|---|---|
| Normal | Baseline operation |
| CPU overload | Heavy computation loop added |
| Deadline miss | Artificial delay in high-priority task |
| Stack risk | Reduced task stack size |
| Queue congestion | Producer rate > consumer rate |
| Task starvation | Skewed task priority configuration |

---

## ML pipeline

- **Features:** 9 raw metrics + 5 derived features (`jitter_ratio`, `total_deadline_miss`, `queue_pressure`, `stack_danger`, `cpu_jitter_stress`) = 14 total
- **Classifier:** Random Forest (Healthy / Warning / Critical) — ~78–79% test accuracy, Warning class at 100% precision
- **Anomaly detector:** Isolation Forest — retrained whenever the feature set changes
- **Validation:** Cross-validation accuracy ~71%

Run training:
```bash
python train_pipeline.py
```

Run inference on a single sample:
```bash
python predict_single.py
```

If cumulative counters overlap across capture sessions, correct and retrain with:
```bash
python fix_and_retrain.py
```

---

## Live dashboard

Start the local dashboard server:
```bash
python server.py
```
Then open `http://localhost:8765` in a browser. The dashboard shows, per task:
- Current status (Healthy / Warning / Critical)
- Risk level and reason text
- Live metric trends
- Predicted failure type
- Event log of state transitions

---

## Data collection workflow

```bash
# 1. Capture raw UART session data
python capture_uart.py

# 2. Label the captured session by fault type
python label_data.py

# 3. Train / retrain the models
python train_pipeline.py
```

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell
pip install -r requirements.txt
```

Connect the STM32F411 via UART (historically enumerated as COM13–COM15 on Windows) at 115200 baud.

---

## Key findings

- Multi-feature pattern recognition across all metrics simultaneously outperforms single-threshold checks — this is the core advantage of the ML approach over classic rule-based validation.
- Real hardware data is noisier than simulated data; cumulative UART counters can overlap between capture sessions and need explicit correction before retraining.
- Labeled fault diversity (six distinct conditions) was necessary to get meaningful separation between Warning and Critical classes.
- Feature engineering (the 5 derived features) measurably improved model expressiveness over raw metrics alone.

---

## Roadmap

- [ ] Improve cross-validation accuracy beyond ~71%
- [ ] TinyML export refinement for on-device inference back on STM32CubeIDE
- [ ] Additional fault scenario coverage
- [ ] Documentation polish for final submission

---

## Author

**Logesh** — ECE student
Project: *AI-Assisted FreeRTOS Task Health Prediction*
