#!/usr/bin/env python3
import asyncio
import logging
import math
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import requests
import tensorflow as tf
from fastapi import FastAPI, Response
from kubernetes import client, config
from prometheus_client import Gauge, CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel
from tensorflow.keras.models import load_model

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)

MODEL_DIR   = "model"
NAMESPACE   = "thesis"
DEPLOY_NAME = "cpu-mem-demo"

FORECAST_HORIZON = 10 * 60
POLL_INTERVAL    = 60
LOOKBACK         = 10

EMA_ALPHA       = float(os.getenv("EMA_ALPHA", "0.2"))
TARGET_FACTOR   = float(os.getenv("TARGET_FACTOR", "1.3"))
avg_rps_per_pod = 50.0

IDLE_RPS      = float(os.getenv("IDLE_RPS", "1.0"))
IDLE_MINUTES  = int(os.getenv("IDLE_MINUTES", "5"))    
IDLE_CAP_PODS = int(os.getenv("IDLE_CAP_PODS", "2"))

PROM_URL = os.getenv(
    "PROM_URL",
    "http://prometheus-kube-prometheus-prometheus.monitoring.svc:9090",
)

QUERIES = {
    "rps": 'sum(rate(http_requests_total{pod=~"cpu-mem-demo-.*",path="/"}[30s])) or vector(0)',

    "cpu": 'sum(rate(container_cpu_usage_seconds_total{namespace="thesis",pod=~"cpu-mem-demo-.*"}[30s])) or vector(0)',
    "mem": 'avg(container_memory_working_set_bytes{namespace="thesis",pod=~"cpu-mem-demo-.*"}) or vector(0)',
}

g_ratio = Gauge(
    "custom_autoscaler_desired_replicas",
    "Predicted ÷ current replica ratio (HPA-Input)",
    ["namespace", "deployment"],
)
g_conf = Gauge(
    "custom_autoscaler_confidence",
    "Heuristic confidence (0–1)",
    ["namespace", "deployment"],
)

DEBUG_RPS   = Gauge("autoscaler_rps",       "Polled RPS",               ["namespace","deployment"])
DEBUG_CPU   = Gauge("autoscaler_cpu",       "Polled CPU cores",         ["namespace","deployment"])
DEBUG_MEM   = Gauge("autoscaler_mem",       "Polled Working-Set MiB",   ["namespace","deployment"])
DEBUG_PRED  = Gauge("autoscaler_predicted", "Raw model prediction",     ["namespace","deployment"])
DEBUG_CAP   = Gauge("autoscaler_cap",       "Reality CAP (# Pods)",     ["namespace","deployment"])
DEBUG_DIV   = Gauge("autoscaler_divisor",   "Adaptive divisor RPS/Pod", ["namespace","deployment"])
DEBUG_IDLE  = Gauge("autoscaler_idle_ticks","Consecutive idle intervals",["namespace","deployment"])

for g in (g_ratio, g_conf,
          DEBUG_RPS, DEBUG_CPU, DEBUG_MEM, DEBUG_PRED, DEBUG_CAP, DEBUG_DIV, DEBUG_IDLE):
    g.labels(NAMESPACE, DEPLOY_NAME).set(0)

app = FastAPI(title="ML autoscaler", version="0.8.1")

logging.info("Loading ML model …")
try:
    model = load_model(f"{MODEL_DIR}/best_model.keras", compile=False)
    logging.info("✓  best_model.keras geladen")
except Exception as exc:
    logging.warning("⚠  .keras-Load fehlgeschlagen (%s) – fallback auf HDF5", exc)
    model = load_model(f"{MODEL_DIR}/best.h5", compile=False)

x_scal = joblib.load(f"{MODEL_DIR}/x_scaler.joblib")
y_scal = joblib.load(f"{MODEL_DIR}/y_scaler.joblib")
logging.info("✓  Scaler geladen")

history    = np.zeros(LOOKBACK * 5, dtype="float32")
idle_ticks = 0 

def time_feats() -> tuple[float, float]:
    h = datetime.now().hour + datetime.now().minute / 60
    return math.sin(2 * math.pi * h / 24), math.cos(2 * math.pi * h / 24)

def current_replicas() -> int:
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()
    dep = client.AppsV1Api().read_namespaced_deployment(DEPLOY_NAME, NAMESPACE)
    return max(dep.spec.replicas or 1, 1)

def prom(query: str) -> float:
    try:
        resp = requests.get(
            f"{PROM_URL}/api/v1/query",
            params={"query": query, "time": datetime.now(timezone.utc).timestamp()},
            timeout=3,
        ).json()
        result = resp.get("data", {}).get("result", [])
        return float(result[0]["value"][1]) if result else 0.0
    except Exception as exc:
        logging.warning("Prom-Query-Fehler: %s  (q=%s)", exc, query)
        return 0.0

# ─── Kernlogik ────────────────────────────────────────────────────────────────
def run_inference(rps: float, cpu: float, mem_mib: float) -> dict:
    global history, avg_rps_per_pod, idle_ticks

    sin_h, cos_h = time_feats()
    history = np.concatenate(([rps, cpu, mem_mib, sin_h, cos_h], history[:-5]))

    x    = x_scal.transform(history.reshape(1, -1))
    pred = int(round(y_scal.inverse_transform(model.predict(x, verbose=0)
                                              .reshape(-1, 1)).item()))
    pred = max(1, min(10, pred))

    curr = current_replicas()
    if curr > 0:
        inst_rps_per_pod = rps / curr
        avg_rps_per_pod  = EMA_ALPHA * inst_rps_per_pod + (1 - EMA_ALPHA) * avg_rps_per_pod

    cap_divisor = max(10, avg_rps_per_pod * TARGET_FACTOR)
    traffic_cap = max(1, math.ceil(rps / cap_divisor))
    pred        = min(pred, traffic_cap + 1)

    idle_required_ticks = max(1, int(round((IDLE_MINUTES * 60) / max(POLL_INTERVAL, 1))))
    if rps < IDLE_RPS:
        idle_ticks += 1
    else:
        idle_ticks = 0
    idle_active = (idle_ticks >= idle_required_ticks)

    conf = max(0.1, 1 - abs(pred - 5) / 5)  # 0.1 … 1.0

    if idle_active:
        desired_idle = max(1, IDLE_CAP_PODS)
        pred = min(pred, max(curr, desired_idle)) if curr <= desired_idle else min(pred, desired_idle)
        if curr > desired_idle:
            ratio = desired_idle / curr
        else:
            ratio = 1.0 
    else:
        if pred > curr:  
            ratio = pred / curr if conf >= 0.5 else 1.0
        elif pred < curr:  
            safe_to_down = (curr - pred) >= 1 and (curr - 1) > traffic_cap \
                           and rps / max(curr - 1, 1) < cap_divisor
            ratio = pred / curr if (conf >= 0.3 and safe_to_down) else 1.0
        else:
            ratio = 1.0

    DEBUG_RPS .labels(NAMESPACE, DEPLOY_NAME).set(rps)
    DEBUG_CPU .labels(NAMESPACE, DEPLOY_NAME).set(cpu)
    DEBUG_MEM .labels(NAMESPACE, DEPLOY_NAME).set(mem_mib)
    DEBUG_PRED.labels(NAMESPACE, DEPLOY_NAME).set(pred)
    DEBUG_CAP .labels(NAMESPACE, DEPLOY_NAME).set(traffic_cap)
    DEBUG_DIV .labels(NAMESPACE, DEPLOY_NAME).set(cap_divisor)
    DEBUG_IDLE.labels(NAMESPACE, DEPLOY_NAME).set(idle_ticks)

    g_ratio.labels(NAMESPACE, DEPLOY_NAME).set(ratio)
    g_conf .labels(NAMESPACE, DEPLOY_NAME).set(conf)

    logging.info(
        "rps=%6.1f  pred=%2d  cap=%2d  curr=%2d  ratio=%4.2f  conf=%.2f  div=%5.1f  idle=%d%s",
        rps, pred, traffic_cap, curr, ratio, conf, cap_divisor, idle_ticks,
        " IDLE" if idle_active else "",
    )

    return {
        "predicted_replicas": pred,
        "current_replicas":  curr,
        "ratio":             ratio,
        "confidence":        conf,
        "traffic_cap":       traffic_cap,
        "divisor":           cap_divisor,
        "idle_ticks":        idle_ticks,
        "idle_active":       idle_active,
    }

class UpdatePayload(BaseModel):
    rps: float
    cpu: float
    memory: float 

@app.post("/update")
async def manual_update(p: UpdatePayload):
    return run_inference(p.rps, p.cpu, p.memory)

@app.post("/reset_history")
async def reset_history():
    global history, avg_rps_per_pod, idle_ticks
    history = np.zeros_like(history)
    avg_rps_per_pod = 50.0
    idle_ticks = 0
    for g in (g_ratio, g_conf):
        g.labels(NAMESPACE, DEPLOY_NAME).set(1.0)
    return {"status": "history reset"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/")
def health():
    return {
        "status": "ok",
        "tf_version": tf.__version__,
        "lookback":   LOOKBACK,
        "model_loaded": True,
    }

@app.on_event("startup")
async def start_poller():
    async def loop():
        while True:
            rps = prom(QUERIES["rps"])
            cpu = prom(QUERIES["cpu"])
            mem = prom(QUERIES["mem"]) / 1024 / 1024
            run_inference(rps, cpu, mem)
            await asyncio.sleep(POLL_INTERVAL)
    asyncio.create_task(loop())

