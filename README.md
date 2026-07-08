# AI-Assisted FreeRTOS Runtime Health Predictor

Predicts the runtime health of FreeRTOS tasks on **STM32F411** using a **Random Forest** classifier and **Isolation Forest** anomaly detector — before failure actually occurs.

---

## Project Structure

```
rtos_health_predictor/
├── data/
│   ├── generate_dataset.py     ← synthetic data generator (dev / testing)
│   └── rtos_health_data.csv    ← your real or generated dataset
├── models/
│   ├── random_forest.pkl       ← trained RF classifier
│   ├── isolation_forest.pkl    ← trained anomaly detector
│   └── scaler.pkl              ← StandardScaler parameters
├── plots/
│   ├── confusion_matrix_rf.png
│   ├── confusion_matrix_if.png
│   ├── feature_importance.png
│   ├── correlation_matrix.png
│   ├── label_distribution.png
│   ├── feature_boxplots.png
│   └── isolation_forest_scores.png
├── src/
│   ├── preprocess.py           ← data loading, scaling, splitting
│   ├── train_random_forest.py  ← RF training + evaluation
│   ├── train_isolation_forest.py ← IF training + evaluation
│   ├── plot_features.py        ← all visualisations
│   ├── predict_health.py       ← inference (single / batch / UART)
│   └── export_tinyml.py        ← C header, emlearn, TFLite export
├── tinyml_export/
│   ├── rtos_health_model.h     ← drop-in C header for STM32
│   ├── emlearn_model.h         ← emlearn C code (if emlearn installed)
│   └── model_tflite/           ← TFLite FlatBuffer + C array
├── .vscode/
│   ├── settings.json
│   └── launch.json             ← 8 pre-configured run configs
├── train_pipeline.py           ← master script – runs everything
└── requirements.txt
```

---

## Dataset Columns

| Column | Source in `main.c` | Description |
|---|---|---|
| `cpu_load` | `CpuLoad` | CPU busy percentage |
| `queue_level` | `uxQueueMessagesWaiting` | Items currently in SensorQueue |
| `queue_dropped` | `QueueDropped` | Messages lost due to full queue |
| `process_jitter` | `ProcessJitter` | Scheduling jitter of ProcessTask (ms) |
| `process_deadline_miss` | `ProcessDeadlineMiss` | Deadline misses for ProcessTask |
| `process_exec_time` | `ProcessExecTime` | Execution time of ProcessTask (ms) |
| `process_stack_left` | `uxTaskGetStackHighWaterMark` | Remaining stack words |
| `sensor_deadline_miss` | `SensorDeadlineMiss` | Deadline misses for SensorTask |
| `comm_deadline_miss` | `CommDeadlineMiss` | Deadline misses for CommTask |
| `health_label` | ground truth | **0** Healthy · **1** Warning · **2** Critical |

---

## Quick Start

```bash
# 1. Clone / open folder in VS Code
cd rtos_health_predictor

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline (generates data if CSV not found)
python train_pipeline.py
```

All models, plots, and TinyML artefacts are produced automatically.

---

## Prediction Modes

### Single sample (JSON)
```bash
python src/predict_health.py --single \
  '{"cpu_load":75,"queue_level":9,"queue_dropped":14,
    "process_jitter":200,"process_deadline_miss":8,
    "process_exec_time":350,"process_stack_left":180,
    "sensor_deadline_miss":6,"comm_deadline_miss":7}'
```

### Batch prediction from CSV
```bash
python src/predict_health.py --csv data/rtos_health_data.csv
# Output: data/predictions.csv
```

### Live UART stream from STM32
```bash
# Linux / Mac
python src/predict_health.py --uart /dev/ttyUSB0 --baud 115200

# Windows
python src/predict_health.py --uart COM3 --baud 115200
```

The UART parser expects the `LoggerTask` format from `main.c`:
```
cpu_load,queue_level,queue_dropped,process_jitter,process_deadline_miss,
process_exec_time,process_stack_left,sensor_deadline_miss,comm_deadline_miss
```

---

## Sample Output

```
╔════════════════════════════════════════════════════╗
║              RTOS RUNTIME HEALTH REPORT            ║
╠════════════════════════════════════════════════════╣
║  Status               Critical                     ║
║  RF Confidence          91.5%                      ║
║  Risk Score             92.0%                      ║
║  Anomaly Detector     Anomaly                      ║
║  IF Score             -0.1823                      ║
╠════════════════════════════════════════════════════╣
║  Probability breakdown:                            ║
║    Healthy      3.0%  ██                           ║
║    Warning      5.5%  █                            ║
║    Critical    91.5%  ██████████████████           ║
╚════════════════════════════════════════════════════╝
```

---

## TinyML Embedded Deployment

### Option A – Rule-based C header (no library needed)
Copy `tinyml_export/rtos_health_model.h` into your STM32 CubeIDE project:

```c
#include "rtos_health_model.h"

float features[N_FEATURES] = {
    cpu_load, queue_level, queue_dropped,
    process_jitter, process_deadline_miss, process_exec_time,
    process_stack_left, sensor_deadline_miss, comm_deadline_miss
};

int label = rtos_predict(features);
// LABEL_HEALTHY=0, LABEL_WARNING=1, LABEL_CRITICAL=2
```

### Option B – emlearn (full RF, ~2 KB flash)
```bash
pip install emlearn
python src/export_tinyml.py
# Copy tinyml_export/emlearn_model.h to STM32 project
```

### Option C – TensorFlow Lite Micro (X-CUBE-AI compatible)
```bash
pip install tensorflow
python src/export_tinyml.py
# Use tinyml_export/model_tflite/rtos_health.tflite with X-CUBE-AI
```

---

## Extending to Real Hardware Data

1. Flash `main.c` to your STM32F411 board.
2. Open a serial terminal at 115200 baud.
3. Collect CSV logs and replace `data/rtos_health_data.csv`.
4. Add ground-truth labels (manual or via fault injection described in the project doc).
5. Re-run `python train_pipeline.py` — everything updates automatically.

---

## Hardware

- **MCU**: STM32F411 (Black Pill / Nucleo-F411RE)
- **RTOS**: FreeRTOS via CMSIS-RTOS v2
- **Interface**: USART2 @ 115200 baud (PA2/PA3)
- **Toolchain**: STM32CubeIDE + HAL drivers
