#!/usr/bin/env python3
"""
pilot_collect.py
────────────────
Download a short window of KPI time-series from Prometheus and save them
as a tidy CSV (one timestamp index, one column per signal).

Environment variables you may override:
  PROM_URL      – Prometheus base URL (default http://localhost:9090)
  APP_NS        – namespace of the workload (default thesis)
  APP_LABEL     – app label (default cpu-mem-demo)
  RUN_MINUTES   – how many minutes back to collect (default 10)
  STEP          – PromQL resolution (default 30s)
"""

import os
from datetime import datetime, timedelta, timezone
import requests
import pandas as pd
import matplotlib.pyplot as plt


PROM      = os.getenv("PROM_URL",   "http://localhost:9090")
NS        = os.getenv("APP_NS",     "thesis")
APP       = os.getenv("APP_LABEL",  "cpu-mem-demo")
WIN_MIN   = int(os.getenv("RUN_MINUTES", "10"))
STEP      = os.getenv("STEP", "30s")

def prom_df(query: str, start: float, end: float, col: str, step: str = STEP) -> pd.DataFrame:
    """Return a 1-column DF (UTC index) for a PromQL range query."""
    resp = requests.get(
        f"{PROM}/api/v1/query_range",
        params=dict(query=query, start=start, end=end, step=step),
        timeout=10,
    ).json()

    if resp["status"] != "success":
        raise RuntimeError(resp)

    result = resp["data"]["result"]
    if not result:
        raise ValueError(f"no series returned for:\n{query}")

    ts = result[0]["values"]                      # [[unix, "val"], …]
    df = pd.DataFrame(ts, columns=["ts", col])
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df[col]  = df[col].astype(float)
    return df.set_index("ts")


end_dt   = datetime.now(timezone.utc)
start_dt = end_dt - timedelta(minutes=WIN_MIN)
start, end = start_dt.timestamp(), end_dt.timestamp()

print(f"⏱  {WIN_MIN} min  |  {start_dt:%Y-%m-%d %H:%M:%S} UTC  →  {end_dt:%H:%M:%S}")


LBL = f'namespace="{NS}",job="{APP}"'
Q = dict(
    rps   = f'sum(rate(http_requests_total{{{LBL},path="/"}}[1m]))',
    p99   = (
        'histogram_quantile(0.99, '
        f'sum by(le) (rate(request_duration_seconds_bucket{{{LBL},path="/"}}[1m])))'
    ),
    cpu   = f'sum(rate(container_cpu_usage_seconds_total{{namespace="{NS}",pod=~"{APP}.*"}}[30s]))',
    mem   = f'sum(container_memory_working_set_bytes{{namespace="{NS}",pod=~"{APP}.*"}}) / 1024 / 1024',
    repl  = f'kube_deployment_status_replicas{{deployment="{APP}",namespace="{NS}"}}',
)


df = (
    prom_df(Q["rps"],  start, end, "rps/s")
      .join(prom_df(Q["p99"], start, end, "p99_lat_s"))
      .join(prom_df(Q["cpu"], start, end, "cpu_cores"))
      .join(prom_df(Q["mem"], start, end, "mem_MiB"))
      .join(prom_df(Q["repl"],start, end, "replicas"))
)

print("\nhead():\n", df.head())
print("\nNaNs per column (should all be 0):\n", df.isna().sum())

fig, ax1 = plt.subplots(figsize=(12,4))
df["rps/s"].plot(ax=ax1, color="tab:blue", label="req/s")
ax2 = ax1.twinx()
df["p99_lat_s"].plot(ax=ax2, ls="--", color="tab:red", label="p99 latency (s)")
ax1.set_ylabel("req/s"); ax2.set_ylabel("seconds")
fig.legend(loc="upper right"); plt.title(f"Pilot {WIN_MIN}-min window")
plt.tight_layout(); plt.show()

fname = f"pilot_{WIN_MIN}m_metrics.csv"
df.to_csv(fname)
print(f"\n✅  saved →  {fname}  ({len(df)} rows)")

