/*
 * rtos_health_model.h
 * Auto-generated TinyML header for RTOS health prediction.
 * Generated: 2026-06-27 15:20:57
 *
 * Usage:
 *   #include "rtos_health_model.h"
 *   float features[N_FEATURES] = {cpu_load, queue_level, ...};
 *   int label = rtos_predict(features);
 *   // 0=HEALTHY, 1=WARNING, 2=CRITICAL
 */

#ifndef RTOS_HEALTH_MODEL_H
#define RTOS_HEALTH_MODEL_H

#include <stdint.h>
#include <math.h>

#define N_FEATURES  9
#define LABEL_HEALTHY   0
#define LABEL_WARNING   1
#define LABEL_CRITICAL  2

/* Feature names (for debugging) */
static const char* const FEATURE_NAMES[N_FEATURES] = {
    "cpu_load",
    "queue_level",
    "queue_dropped",
    "process_jitter",
    "process_deadline_miss",
    "process_exec_time",
    "process_stack_left",
    "sensor_deadline_miss",
    "comm_deadline_miss",
};

/* StandardScaler parameters */
static const float SCALER_MEAN[N_FEATURES] = {
    6.878125f, 1.953125f, 216.081250f, 840.306250f, 157.937500f, 889.018750f, 344.237500f, 52.312500f, 58.490625f
};
static const float SCALER_STD[N_FEATURES]  = {
    6.278994f, 3.417006f, 601.617761f, 1236.250564f, 102.091668f, 1207.887443f, 99.141155f, 80.159114f, 77.456762f
};

/* Rule-based thresholds (original scale) */
static const float WARN_THRESH[N_FEATURES] = {
    50.00f, 4.00f, 3.00f, 50.00f, 2.00f, 150.00f, 600.00f, 2.00f, 2.00f
};
static const float CRIT_THRESH[N_FEATURES] = {
    70.00f, 7.00f, 8.00f, 120.00f, 6.00f, 280.00f, 300.00f, 6.00f, 6.00f
};

/* Normalize a feature array in-place */
static inline void rtos_normalize(float* x) {
    for (int i = 0; i < N_FEATURES; i++) {
        x[i] = (x[i] - SCALER_MEAN[i]) / SCALER_STD[i];
    }
}

/*
 * Lightweight rule-based predictor.
 * Replace with emlearn or TFLM inference for full RF accuracy.
 */
static inline int rtos_predict(const float* raw_features) {
    int warn_votes = 0;
    int crit_votes = 0;

    /* cpu_load, queue_level ... queue_dropped: higher = worse */
    for (int i = 0; i <= 5; i++) {
        if (raw_features[i] >= CRIT_THRESH[i]) crit_votes++;
        else if (raw_features[i] >= WARN_THRESH[i]) warn_votes++;
    }
    /* process_stack_left: LOWER = worse */
    if (raw_features[6] <= CRIT_THRESH[6]) crit_votes++;
    else if (raw_features[6] <= WARN_THRESH[6]) warn_votes++;

    /* sensor / comm deadline miss */
    for (int i = 7; i < N_FEATURES; i++) {
        if (raw_features[i] >= CRIT_THRESH[i]) crit_votes++;
        else if (raw_features[i] >= WARN_THRESH[i]) warn_votes++;
    }

    if (crit_votes >= 2) return LABEL_CRITICAL;
    if (warn_votes >= 2 || crit_votes == 1) return LABEL_WARNING;
    return LABEL_HEALTHY;
}

#endif /* RTOS_HEALTH_MODEL_H */