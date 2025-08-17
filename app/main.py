import os, time
from fastapi import FastAPI, Response, Request
from prometheus_client import Counter, Histogram, generate_latest

CPU_MS = int(os.getenv("CPU_MS", "50"))
MEM_MB = int(os.getenv("MEM_MB", "0"))

REQ_CNT = Counter(
    "http_requests_total", "HTTP requests", ["path", "code"]
)
REQ_LAT = Histogram(
    "request_duration_seconds", "Response latency in seconds", ["path"]
)

app = FastAPI()

def burn_cpu(ms: int):
    stop = time.perf_counter() + ms / 1000
    while time.perf_counter() < stop:
        pass

def burn_mem(mb: int):
    if mb > 0:
        _ = bytearray(mb * 1024 * 1024)

@app.middleware("http")
async def observe(request: Request, call_next):
    start = time.perf_counter()
    resp  = await call_next(request)
    dur_s = time.perf_counter() - start
    REQ_CNT.labels(path=request.url.path, code=resp.status_code).inc()
    REQ_LAT.labels(path=request.url.path).observe(dur_s)
    return resp

@app.get("/")
def root() -> Response:
    burn_cpu(CPU_MS); burn_mem(MEM_MB)
    return {"cpu_ms": CPU_MS, "mem_mb": MEM_MB}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")

