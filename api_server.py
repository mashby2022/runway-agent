"""
Sidecar API server — port 8081
Exposes endpoints that Lovable calls directly (not via the NAT agent).
Shares state with the NAT server via data/mode.json.

Start alongside nat serve:
    python api_server.py &
"""

import importlib.util
import json
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

MODE_FILE = os.path.join(os.path.dirname(__file__), "data", "mode.json")

HAS_GPU: bool = importlib.util.find_spec("cudf") is not None

_COMPUTE_PROFILES = {
    "ONLINE":  {"source_compute": "NVIDIA A10G (Brev GPU)", "gpu_boost": "35x",  "latency_ms": 12},
    "OFFLINE": {"source_compute": "MacBook Local (CPU)",    "gpu_boost": "1x",   "latency_ms": 180},
}

app = FastAPI(title="Runway Inclusive — Sidecar API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _read_mode() -> str:
    try:
        with open(MODE_FILE) as f:
            return json.load(f).get("mode", "OFFLINE")
    except (FileNotFoundError, json.JSONDecodeError):
        return "OFFLINE"


def _write_mode(mode: str) -> None:
    os.makedirs(os.path.dirname(MODE_FILE), exist_ok=True)
    with open(MODE_FILE, "w") as f:
        json.dump({"mode": mode}, f)


class ModeRequest(BaseModel):
    mode: str  # "ONLINE" or "OFFLINE"


@app.get("/health")
def health():
    return {
        "status":       "alive",
        "engine":       _read_mode(),
        "gpu_detected": HAS_GPU,
    }


@app.get("/system_mode")
def get_mode():
    mode = _read_mode()
    return {"execution_mode": mode, **_COMPUTE_PROFILES[mode]}


@app.post("/toggle_system_mode")
def toggle_mode(body: ModeRequest):
    mode = body.mode.upper().strip()
    if mode not in _COMPUTE_PROFILES:
        return {"error": f"Unknown mode '{mode}'. Use 'ONLINE' or 'OFFLINE'."}
    _write_mode(mode)
    return {
        "status":         "mode_switched",
        "execution_mode": mode,
        **_COMPUTE_PROFILES[mode],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081, log_level="info")
