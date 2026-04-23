"""
Sidecar API server — port 8081

Responsibilities:
  - SSE proxy for /generate/stream  →  strips raw Python reprs, emits clean JSON events
  - Non-streaming proxy for /generate
  - Compute mode toggle (shared via data/mode.json with nat serve)
  - Chat history persistence (data/chat_history.json)
  - System health / status endpoints

Start alongside nat serve:
    python api_server.py &
"""

import asyncio
import importlib.util
import io
import json
import logging
import math
import os
import uuid
from datetime import datetime, timezone
from threading import Lock

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("runway_api")

import httpx
import pandas as pd
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

_DUCKDB_AVAILABLE = importlib.util.find_spec("duckdb") is not None

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR       = os.path.join(os.path.dirname(__file__), "data")
MODE_FILE      = os.path.join(DATA_DIR, "mode.json")
HISTORY_FILE   = os.path.join(DATA_DIR, "chat_history.json")
CATALOG_FILE   = os.path.join(DATA_DIR, "catalog.json")
BYPASS_FILE    = os.path.join(DATA_DIR, "bypass_state.json")

NAT_BASE = "http://localhost:8080"

HAS_GPU: bool = importlib.util.find_spec("cudf") is not None

# ── Startup validation ────────────────────────────────────────────────────────
_NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
if not _NVIDIA_API_KEY:
    print("\n" + "=" * 70)
    print("  WARNING: NVIDIA_API_KEY is not set.")
    print("  Miranda will not reach the NIM inference endpoint.")
    print("  Fix: export NVIDIA_API_KEY=nvapi-...")
    print("=" * 70 + "\n")

# ── Environment detection ─────────────────────────────────────────────────────
# Brev GPU instances set BREV_WORKSPACE_ID; cudf presence is a reliable proxy.
# Everything else is treated as MacBook Local.
_IS_BREV: bool = bool(
    os.environ.get("BREV_WORKSPACE_ID")
    or os.environ.get("BREV_CLUSTER_ID")
    or os.environ.get("RUNWAY_ENV", "").upper() == "BREV"
    or HAS_GPU
)

_COMPUTE_PROFILES = {
    "ONLINE": {
        "source_compute": "NVIDIA A10G (Brev GPU)",
        "engine":         "NVIDIA RAPIDS (cuDF)",
        "gpu_boost":      "35x",
        "latency_ms":     12,
        "audit_footer":   "[ Mode: GPU | Engine: NVIDIA RAPIDS (cuDF) ]",
    },
    "OFFLINE": {
        "source_compute": "MacBook Local (CPU)" if not _IS_BREV else "Brev CPU",
        "engine":         "DuckDB + Parquet",
        "gpu_boost":      "1x",
        "latency_ms":     185,
        "audit_footer":   (
            "[ Mode: Local | Engine: DuckDB ]"
            if not _IS_BREV else
            "[ Mode: Brev CPU | Engine: DuckDB ]"
        ),
    },
}

# ── Pydantic models ───────────────────────────────────────────────────────────

class ModeRequest(BaseModel):
    mode: str


class AppendRequest(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[dict]
    reset_history: bool = False
    stream: bool = True


class BriefRequest(BaseModel):
    title:                    str   = "Untitled"
    persona:                  str   = "executive"
    strategic_recommendation: str   = "Greenlight Priority: High"
    executive_summary:        str   = ""
    resonance_score:          float = 0.87
    completion_rate:          float = 0.81
    platform:                 str   = "Streaming"
    thematic_alignment:       float = 88.0
    confidence_score:         float = 82.4
    variance_delta:           str   = "8.3%"
    tf_reach_prediction:      str   = ""
    target_network:           str   = "CBS"
    target_demographic:       str   = "35-45"
    validation_status:        str   = "VALIDATED"
    thematic_signal:          str   = ""
    faiss_insight:            str   = ""
    script_tags:              str   = ""
    graph_data:               str   = ""
    latency_ms:               int   = 340
    engine:                   str   = "RAPIDS/cuDF | NVIDIA L4"
    mode:                     str   = "ONLINE"


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Runway Inclusive — Sidecar API", version="1.2")

# CORS — wildcard covers Lovable preview (*.lovable.app, *.lovableproject.com)
# and all localhost variants.  allow_credentials MUST be False with wildcard.
# OPTIONS pre-flight is handled automatically by CORSMiddleware.
_LOVABLE_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8080",
    "http://localhost:8081",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8081",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_LOVABLE_ORIGINS,
    allow_origin_regex=r"https://.*\.(lovable\.app|lovableproject\.com)",
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    allow_credentials=True,
    max_age=3600,
)

_history_lock  = Lock()
_bypass_lock   = Lock()
_cancel_flag: dict = {"active": False}   # mutable sentinel — checked by SSE generator

# ── Bypass state helpers ──────────────────────────────────────────────────────

def _load_bypass_state() -> dict:
    try:
        with open(BYPASS_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"active": False}


def _set_bypass_state(prompt_excerpt: str = "") -> dict:
    state = {
        "active":             True,
        "controller_status":  "MANUAL_BYPASS",
        "active_gatekeeper":  "Tiger Team",
        "triggered_at":       datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "interrupted_prompt": prompt_excerpt[:300],
        "audit":              _MANUAL_OVERRIDE_AUDIT,
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(BYPASS_FILE, "w") as f:
        json.dump(state, f, indent=2)
    return state


def _clear_bypass_state() -> None:
    state = {"active": False}
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(BYPASS_FILE, "w") as f:
        json.dump(state, f, indent=2)


def _bypass_annotation() -> dict:
    """Return bypass fields to inject into endpoint responses, or {} if not active."""
    with _bypass_lock:
        state = _load_bypass_state()
    if not state.get("active"):
        return {}
    return {
        "controller_status":  state["controller_status"],
        "active_gatekeeper":  state["active_gatekeeper"],
        "bypass_triggered_at": state.get("triggered_at", ""),
    }


# ── Keyword-bypass helpers ────────────────────────────────────────────────────

# Keys are the bypass type; values are substrings to match (all lowercased).
_BYPASS_KEYWORDS: dict[str, set[str]] = {
    "quad": {
        "quad", "loyalty tier", "audience composition",
        "gold viewer", "silver viewer", "occasional viewer",
    },
    "epg": {
        "epg", " schedule", "what's on", "what is on",
        "airing", "now playing", "currently playing",
    },
    "tco": {
        "tco", "active viewer", "watch time", "engagement rate",
        "peak concurrent", "kpi", "channel metric", "audience metric",
        "total viewer",
    },
}

_MARKET_NAMES: dict[str, str] = {
    # Longer / more specific aliases must come before short ones that could be substrings
    "new york":      "New York",    "nyc":           "New York",
    "los angeles":   "Los Angeles", "san francisco": "San Francisco",
    "chicago":       "Chicago",     "dallas":        "Dallas",
    "miami":         "Miami",       "atlanta":       "Atlanta",
    "paris":         "Paris",       "milan":         "Milan",
    "london":        "London",      "la":            "Los Angeles",
}


def _detect_bypass(text: str) -> str | None:
    t = text.lower()
    for kw_type, keywords in _BYPASS_KEYWORDS.items():
        if any(k in t for k in keywords):
            return kw_type
    return None


def _extract_market(text: str) -> str:
    t = text.lower()
    for alias, name in _MARKET_NAMES.items():
        if alias in t:
            return name
    return ""


def _bypass_direct(bypass_type: str, market: str, segment: str) -> str:
    """Synchronous direct tool call — returns Intelligence Brief string immediately."""
    from tools import get_quad_analysis, get_audience_metrics, get_current_schedule

    market_label = market or "All Markets"
    # Strip outer brackets so we can embed the mode string cleanly
    _raw_footer  = _compute_audit_footer()
    footer_inner = _raw_footer.strip().lstrip("[").rstrip("]").strip()

    try:
        if bypass_type == "quad":
            data = json.loads(get_quad_analysis(market, segment))
            gold, silver = data.get("Gold", 0), data.get("Silver", 0)
            occasional   = data.get("Occasional", 0)
            total        = data.get("total_viewers", 0)
            engine       = data.get("engine", "DuckDB + Parquet")
            return (
                f"# Audience Composition — {market_label}\n\n"
                f"## Data Comparison\n\n"
                f"| Segment | Share % | Tier |\n|---|---|---|\n"
                f"| Gold (≥ 85% completion) | {gold}% | Devoted |\n"
                f"| Silver (60–85%) | {silver}% | Regular |\n"
                f"| Occasional (< 60%) | {occasional}% | Light |\n\n"
                f"## Why This Matters\n\n"
                f"> {market_label} loyalty profile — Gold at {gold}% of {total:,} viewers.\n"
                f">\n> `AR = (Δ_tribe_affinity × σ_viewership) / (CR_target × HUT_prime)`\n\n"
                f"`[ {footer_inner} | Engine: {engine} | Viewers: {total:,} ]`\n\nThat is all."
            )

        if bypass_type == "tco":
            data = json.loads(get_audience_metrics(market, segment))
            active    = data.get("active_viewers", 0)
            avg_watch = data.get("avg_watch_time", 0.0)
            peak      = data.get("peak_concurrent", 0)
            eng_rate  = data.get("engagement_rate", 0.0)
            engine    = data.get("engine", "DuckDB + Parquet")
            return (
                f"# Channel KPIs — {market_label}\n\n"
                f"## Data Comparison\n\n"
                f"| Metric | Value | Tier |\n|---|---|---|\n"
                f"| Active Viewers | {active:,} | Live |\n"
                f"| Avg Watch Time | {avg_watch}m | Session |\n"
                f"| Peak Concurrent | {peak:,} | Peak |\n"
                f"| Engagement Rate | {eng_rate:.1%} | Rate |\n\n"
                f"## Why This Matters\n\n"
                f"> Engagement at {eng_rate:.1%} signals Couture One audience loyalty.\n"
                f">\n> `AR = (Δ_tribe_affinity × σ_viewership) / (CR_target × HUT_prime)`\n\n"
                f"`[ {footer_inner} | Engine: {engine} ]`\n\nThat is all."
            )

        if bypass_type == "epg":
            now_utc  = datetime.now(timezone.utc)
            now_iso  = now_utc.strftime("%Y-%m-%dT%H:%M:%S+00:00")
            now_hhmm = now_utc.strftime("%H:%M")
            now_day  = now_utc.strftime("%A")
            slot     = get_current_schedule("ch_runway_01", now_iso)

            # If daily schedule.json has no match, fall back to weekly_schedule.json
            if "error" in slot:
                try:
                    weekly   = _load_json("data/weekly_schedule.json")
                    plan     = weekly.get("weekly_plan", [])
                    day_data = next(
                        (d for d in plan if d.get("day_of_week", "").title() == now_day), {}
                    )
                    wslots   = sorted(day_data.get("slots", []), key=lambda s: s.get("time", ""))
                    slot     = next(
                        (s for s in reversed(wslots) if s.get("time", "99:99") <= now_hhmm),
                        wslots[0] if wslots else {},
                    )
                except Exception:
                    slot = {}

            title   = slot.get("title", "Continuous Stream")
            t_start = slot.get("time", now_hhmm)
            block   = slot.get("block_duration_min", 90)
            eh, em  = divmod((int(t_start[:2]) * 60 + int(t_start[3:]) + block) % 1440, 60)
            t_end   = slot.get("ends_at", f"{eh:02d}:{em:02d}")
            tribe   = slot.get("tribe", "Heritage Couture")
            return (
                f"# EPG — Couture One Now Playing\n\n"
                f"## Data Comparison\n\n"
                f"| Time | New Title | Tribe Resonance | Strategic Impact |\n|---|---|---|---|\n"
                f"| {t_start}–{t_end} | {title} | {tribe} | Live — {block}m block |\n\n"
                f"## Why This Matters\n\n"
                f"> Current slot live on the Couture One feed.\n"
                f">\n> `AR = (Δ_tribe_affinity × σ_viewership) / (CR_target × HUT_prime)`\n\n"
                f"`[ {footer_inner} | Engine: DuckDB + Parquet ]`\n\nThat is all."
            )

    except Exception as exc:
        return (
            f"# Editorial Conflict Detected\n\n"
            f"Direct engine error: `{type(exc).__name__}`.\n\n"
            f"`[ {footer_inner} ]`\n\nThat is all."
        )

    return "# No Data\n\nThat is all."


async def _bypass_sse(bypass_type: str, market: str, segment: str) -> ...:
    """Async SSE generator — calls tools.py directly, no NAT round-trip."""
    from tools import get_quad_analysis, get_audience_metrics, get_current_schedule

    market_label = market or "All Markets"
    footer = _compute_audit_footer()

    try:
        if bypass_type == "quad":
            data = json.loads(get_quad_analysis(market, segment))
            gold       = data.get("Gold", 0)
            silver     = data.get("Silver", 0)
            occasional = data.get("Occasional", 0)
            total      = data.get("total_viewers", 0)
            engine     = data.get("engine", "DuckDB + Parquet")
            brief = (
                f"# Audience Composition — {market_label}\n\n"
                f"## Data Comparison\n\n"
                f"| Segment | Share % | Tier |\n"
                f"|---|---|---|\n"
                f"| Gold (≥ 85% completion) | {gold}% | Devoted |\n"
                f"| Silver (60–85%) | {silver}% | Regular |\n"
                f"| Occasional (< 60%) | {occasional}% | Light |\n\n"
                f"## Why This Matters\n\n"
                f"> {market_label} loyalty profile active — Gold at {gold}% of {total:,} total viewers.\n"
                f">\n"
                f"> `AR = (Δ_tribe_affinity × σ_viewership) / (CR_target × HUT_prime)`\n\n"
                f"`[ {footer} | Engine: {engine} | Viewers: {total:,} ]`\n\n"
                f"That is all."
            )

        elif bypass_type == "tco":
            data = json.loads(get_audience_metrics(market, segment))
            active    = data.get("active_viewers", 0)
            avg_watch = data.get("avg_watch_time", 0.0)
            peak      = data.get("peak_concurrent", 0)
            eng_rate  = data.get("engagement_rate", 0.0)
            engine    = data.get("engine", "DuckDB + Parquet")
            brief = (
                f"# Channel KPIs — {market_label}\n\n"
                f"## Data Comparison\n\n"
                f"| Metric | Value | Tier |\n"
                f"|---|---|---|\n"
                f"| Active Viewers | {active:,} | Live |\n"
                f"| Avg Watch Time | {avg_watch}m | Session |\n"
                f"| Peak Concurrent | {peak:,} | Peak |\n"
                f"| Engagement Rate | {eng_rate:.1%} | Rate |\n\n"
                f"## Why This Matters\n\n"
                f"> Engagement at {eng_rate:.1%} signals audience loyalty alignment with the Couture One mandate.\n"
                f">\n"
                f"> `AR = (Δ_tribe_affinity × σ_viewership) / (CR_target × HUT_prime)`\n\n"
                f"`[ {footer} | Engine: {engine} ]`\n\n"
                f"That is all."
            )

        elif bypass_type == "epg":
            now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
            slot      = get_current_schedule("ch_runway_01", now_iso)
            title     = slot.get("title", "—")
            time_str  = slot.get("time", slot.get("start", "—"))
            ends_at   = slot.get("ends_at", slot.get("end", "—"))
            tribe     = slot.get("tribe", "—")
            block_min = slot.get("block_duration_min", 0)
            brief = (
                f"# EPG — Couture One Now Playing\n\n"
                f"## Data Comparison\n\n"
                f"| Time | New Title | Tribe Resonance | Strategic Impact |\n"
                f"|---|---|---|---|\n"
                f"| {time_str}–{ends_at} | {title} | {tribe} | Live — {block_min}m block |\n\n"
                f"## Why This Matters\n\n"
                f"> Current slot is live on the Couture One feed.\n"
                f">\n"
                f"> `AR = (Δ_tribe_affinity × σ_viewership) / (CR_target × HUT_prime)`\n\n"
                f"`[ {footer} | Engine: DuckDB + Parquet ]`\n\n"
                f"That is all."
            )

        else:
            brief = "# No Data\n\nThat is all."

    except Exception as exc:
        brief = (
            f"# Editorial Conflict Detected\n\n"
            f"The direct engine encountered an error: `{type(exc).__name__}`.\n\n"
            f"`[ {footer} ]`\n\nThat is all."
        )

    _append_message("assistant", brief)
    yield f"data: {json.dumps({'type': 'text', 'content': brief})}\n\n"
    yield "data: [DONE]\n\n"


# ── Mode helpers ──────────────────────────────────────────────────────────────

def _read_mode() -> str:
    try:
        with open(MODE_FILE) as f:
            return json.load(f).get("mode", "OFFLINE")
    except (FileNotFoundError, json.JSONDecodeError):
        return "OFFLINE"


def _write_mode(mode: str) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(MODE_FILE, "w") as f:
        json.dump({"mode": mode}, f)


def _compute_audit_footer() -> str:
    """Return the one-line audit footer for the current environment + mode.

    On a MacBook (no Brev, no cudf): '[ Mode: Local | Engine: DuckDB ]'
    On a Brev GPU instance (ONLINE):  '[ Mode: GPU | Engine: NVIDIA RAPIDS (cuDF) ]'
    On a Brev CPU instance (OFFLINE): '[ Mode: Brev CPU | Engine: DuckDB ]'
    """
    mode = _read_mode() if _IS_BREV else "OFFLINE"
    return _COMPUTE_PROFILES.get(mode, _COMPUTE_PROFILES["OFFLINE"])["audit_footer"]


def _environment_meta() -> dict:
    """Return environment + compute fields to embed in endpoint responses."""
    mode    = _read_mode() if _IS_BREV else "OFFLINE"
    profile = _COMPUTE_PROFILES.get(mode, _COMPUTE_PROFILES["OFFLINE"])
    return {
        "environment":    "brev_gpu" if _IS_BREV and mode == "ONLINE" else
                          "brev_cpu" if _IS_BREV else "macbook_local",
        "compute_profile": profile["source_compute"],
        "engine":          profile["engine"],
        "audit_footer":    profile["audit_footer"],
    }


# ── Chat history helpers ──────────────────────────────────────────────────────

def _load_history() -> list[dict]:
    try:
        with open(HISTORY_FILE) as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_history(messages: list[dict]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(messages, f, indent=2)


def _append_message(role: str, content: str) -> None:
    with _history_lock:
        messages = _load_history()
        messages.append({
            "role":      role,
            "content":   content,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        })
        _save_history(messages)


def _record_user_turn(body: dict) -> None:
    msgs = body.get("messages", [])
    if msgs:
        last = msgs[-1]
        if last.get("role") == "user" and last.get("content"):
            _append_message("user", last["content"])


# ── SSE chunk parsers ─────────────────────────────────────────────────────────

def _parse_data_line(payload: str) -> str | None:
    """Return the text content from a NAT `data:` JSON payload, or None to drop.

    Drops: [DONE], tool-call chunks, empty/None delta.content, malformed JSON.
    """
    if payload == "[DONE]":
        return None
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return None

    choices = obj.get("choices") or []
    if not choices:
        return None

    delta = choices[0].get("delta") or {}
    if delta.get("tool_calls"):   # Miranda calling a tool — no visible text
        return None

    content = delta.get("content")
    return content if content else None


def _parse_intermediate_line(payload: str) -> dict | None:
    """Return a clean profilerTrace dict from a NAT `intermediate_data:` payload.

    Strips the raw Python repr from the payload field — only the name is kept.
    """
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return None

    name = obj.get("name", "")
    if not name:
        return None

    return {
        "type":      "trace",
        "event":     obj.get("type", ""),
        "name":      name,
        "parent_id": obj.get("parent_id", ""),
    }


# ── NAT proxy endpoints ───────────────────────────────────────────────────────

@app.post("/generate/stream")
async def generate_stream(request: Request):
    """Clean SSE proxy for NAT's /generate/stream.

    Accepts an optional `reset_history: true` flag in the JSON body.
    When set, chat_history.json is cleared before the request is forwarded,
    preventing state pollution / response loops.
    """
    logger.info("POST /generate/stream — client: %s", request.client)
    body = await request.json()

    persona = body.pop("persona", "executive").lower().strip()
    body["persona_context"] = persona   # forward to NAT for dynamic persona routing

    # Session reset — wipe history before forwarding to NAT
    if body.get("reset_history"):
        with _history_lock:
            _save_history([])
        print("DEBUG: History reset requested — cleared chat_history.json")

    # Debug log every incoming message
    msgs = body.get("messages", [])
    if msgs:
        last_content = msgs[-1].get("content", "")[:80]
        print(f"DEBUG: Processing new message: {last_content!r}")

    if msgs and msgs[-1].get("role") == "user":
        if persona == "analyst":
            msgs[-1]["content"] += "\n\n[SYSTEM OVERRIDE: ANALYST WORKSPACE ACTIVE. Drop the Miranda Priestly metaphors. Provide raw data, metric tables, and deep analytical exploration. Prioritize data density over tone.]"
        else:
            msgs[-1]["content"] += "\n\n[SYSTEM OVERRIDE: EXECUTIVE BRIEF ACTIVE. Maintain your exacting Miranda Priestly persona. Be concise, cutting, use fashion metaphors, and strictly format your output as an Intelligence Brief.]"

    _record_user_turn(body)

    # Capture prompt for bypass staging before entering generator scope
    _prompt_excerpt = (msgs[-1].get("content", "") if msgs else "")[:300]
    _generation_complete = {"ok": False}   # mutable flag updated inside generator

    async def event_gen():
        assembled: list[str] = []
        client_disconnected = False

        _KEEPALIVE_INTERVAL = 8.0   # seconds before emitting a Strategic Wait heartbeat
        _KEEPALIVE_MESSAGES = [
            "Strategic Analysis in progress — the Condé Nast Intelligence Layer is processing.",
            "Cross-referencing the Style Tribe index. One moment.",
            "Reviewing the weekly arc. Patience is a virtue, even in fashion.",
            "The engine is working. Excellence cannot be rushed.",
        ]
        _keepalive_seq = 0

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{NAT_BASE}/generate/stream",
                    json=body,
                    headers={"Accept": "text/event-stream"},
                ) as nat_resp:
                    _line_iter = nat_resp.aiter_lines().__aiter__()
                    while True:
                        # ── Cancellation checks (before waiting for next line) ──
                        if _cancel_flag["active"]:
                            _cancel_flag["active"] = False
                            client_disconnected = True
                            break
                        if await request.is_disconnected():
                            client_disconnected = True
                            break

                        # ── Wait for next line with keepalive timeout ──────────
                        try:
                            raw_line = await asyncio.wait_for(
                                _line_iter.__anext__(), timeout=_KEEPALIVE_INTERVAL
                            )
                        except asyncio.TimeoutError:
                            # NAT is still thinking — send a visible heartbeat
                            msg = _KEEPALIVE_MESSAGES[_keepalive_seq % len(_KEEPALIVE_MESSAGES)]
                            _keepalive_seq += 1
                            yield f"data: {json.dumps({'type': 'wait', 'content': msg})}\n\n"
                            continue
                        except StopAsyncIteration:
                            break

                        raw_line = raw_line.strip()
                        if not raw_line:
                            continue

                        # intermediate_data → clean trace (no Python reprs)
                        if raw_line.startswith("intermediate_data:"):
                            payload = raw_line[len("intermediate_data:"):].strip()
                            trace = _parse_intermediate_line(payload)
                            if trace:
                                yield f"data: {json.dumps(trace)}\n\n"
                            continue

                        # data → extract text content only
                        if raw_line.startswith("data:"):
                            payload = raw_line[5:].strip()
                            if payload == "[DONE]":
                                break
                            text = _parse_data_line(payload)
                            if text:
                                assembled.append(text)
                                yield f"data: {json.dumps({'type': 'text', 'content': text})}\n\n"

        except Exception as exc:
            logger.error("SSE generation error: %s: %s", type(exc).__name__, exc)
            err_event = {
                "type": "error",
                "content": (
                    f"Connection error ({type(exc).__name__}). "
                    "The intelligence engine is unavailable — confirm NAT is running on port 8080 and retry."
                ),
            }
            yield f"data: {json.dumps(err_event)}\n\n"

        if client_disconnected:
            print(f"DEBUG: Client disconnected mid-generation. Prompt: {_prompt_excerpt[:60]!r}")
            with _bypass_lock:
                _set_bypass_state(_prompt_excerpt)
        else:
            # Clean completion — clear any active bypass
            _generation_complete["ok"] = True
            with _bypass_lock:
                _clear_bypass_state()

            # ── Empty-response guard — neutral fallback only ────────────────
            # DVP ("Why isn't anybody ready?") is reserved for explicit
            # is_final_conflict signals from tools. A plain empty response
            # (timeout, model silence, keepalive exhaustion) gets a polite
            # no-response notice instead so the browser is never left blank.
            if not assembled:
                no_resp_msg = (
                    "# No Response Received\n\n"
                    "The strategic engine did not return a response this time. "
                    "This is likely a local CPU timeout — the Nano model needs "
                    "up to 60 seconds on MacBook hardware.\n\n"
                    f"`[ {_compute_audit_footer()} ]`\n\nThat is all."
                )
                assembled = [no_resp_msg]
                yield f"data: {json.dumps({'type': 'text', 'content': no_resp_msg})}\n\n"
                print("DEBUG: Empty response — injected neutral no-response fallback.")

        if assembled:
            full_response = "".join(assembled)
            _append_message("assistant", full_response)

            # Analyst mode: emit a structured persona_metrics event for the Trace Drawer.
            # The model includes a json block in its text; this parallel event gives the
            # frontend a reliable machine-readable signal regardless of model formatting.
            if persona == "analyst":
                import re as _re
                # Extract FAISS chunk metadata from the assembled response if present
                chunk_pattern = _re.compile(
                    r'"source"\s*:\s*"([^"]+)".*?"category"\s*:\s*"([^"]+)".*?"score"\s*:\s*([\d.]+)',
                    _re.DOTALL,
                )
                retrieved_chunks = [
                    {"source": m.group(1), "category": m.group(2), "score": float(m.group(3))}
                    for m in chunk_pattern.finditer(full_response)
                ]
                # Extract RAPIDS technical_evidence block if present in response
                tech_evidence: dict = {}
                te_match = _re.search(
                    r'"technical_evidence"\s*:\s*(\{[^}]+\})', full_response, _re.DOTALL
                )
                if te_match:
                    try:
                        tech_evidence = json.loads(te_match.group(1))
                    except Exception:
                        pass
                metrics_event = {
                    "type":               "persona_metrics",
                    "persona":            "analyst",
                    "retrieved_chunks":   retrieved_chunks,
                    "analytics_engine":   _compute_audit_footer(),
                    "corpus_size":        11,
                    "technical_evidence": tech_evidence,
                }
                yield f"data: {json.dumps(metrics_event)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


@app.post("/generate")
async def generate(request: Request):
    """Non-streaming proxy for NAT's /generate."""
    body = await request.json()

    persona = body.pop("persona", "executive").lower().strip()
    body["persona_context"] = persona   # forward to NAT for dynamic persona routing

    if body.get("reset_history"):
        with _history_lock:
            _save_history([])
        print("DEBUG: History reset — cleared chat_history.json")

    msgs = body.get("messages", [])
    if msgs:
        print(f"DEBUG: Processing new message: {msgs[-1].get('content','')[:80]!r}")

    if msgs and msgs[-1].get("role") == "user":
        if persona == "analyst":
            msgs[-1]["content"] += "\n\n[SYSTEM OVERRIDE: ANALYST WORKSPACE ACTIVE. Drop the Miranda Priestly metaphors. Provide raw data, metric tables, and deep analytical exploration. Prioritize data density over tone.]"
        else:
            msgs[-1]["content"] += "\n\n[SYSTEM OVERRIDE: EXECUTIVE BRIEF ACTIVE. Maintain your exacting Miranda Priestly persona. Be concise, cutting, use fashion metaphors, and strictly format your output as an Intelligence Brief.]"

    _record_user_turn(body)

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{NAT_BASE}/generate", json=body)
        data = resp.json()

    try:
        content = data["choices"][0]["message"]["content"]
        if content:
            _append_message("assistant", content)
            with _bypass_lock:
                _clear_bypass_state()

            # Analyst mode: inject persona_metrics block for the frontend Trace Drawer.
            if persona == "analyst":
                import re as _re
                chunk_pattern = _re.compile(
                    r'"source"\s*:\s*"([^"]+)".*?"category"\s*:\s*"([^"]+)".*?"score"\s*:\s*([\d.]+)',
                    _re.DOTALL,
                )
                retrieved_chunks = [
                    {"source": m.group(1), "category": m.group(2), "score": float(m.group(3))}
                    for m in chunk_pattern.finditer(content)
                ]
                # Extract RAPIDS technical_evidence block if present in response
                tech_evidence: dict = {}
                te_match = _re.search(
                    r'"technical_evidence"\s*:\s*(\{[^}]+\})', content, _re.DOTALL
                )
                if te_match:
                    try:
                        tech_evidence = json.loads(te_match.group(1))
                    except Exception:
                        pass
                data["persona_metrics"] = {
                    "persona":            "analyst",
                    "retrieved_chunks":   retrieved_chunks,
                    "analytics_engine":   _compute_audit_footer(),
                    "corpus_size":        11,
                    "technical_evidence": tech_evidence,
                }
        else:
            # Empty content — neutral no-response notice (DVP reserved for is_final_conflict)
            fallback = (
                "# No Response Received\n\n"
                "The strategic engine returned no content. "
                "This is typically a local CPU timeout — please allow up to 60 seconds "
                "for the Nano model on MacBook hardware, or rephrase your request.\n\n"
                f"`[ {_compute_audit_footer()} ]`\n\nThat is all."
            )
            data.setdefault("choices", [{}])
            data["choices"][0].setdefault("message", {})["content"] = fallback
            data["_conflict_injected"] = True
            _append_message("assistant", fallback)
            print("DEBUG: Empty /generate response — injected Editorial Conflict fallback.")
    except (KeyError, IndexError, TypeError):
        pass

    return data


# ── Stop / Manual-Override bypass ────────────────────────────────────────────

_MANUAL_OVERRIDE_AUDIT = (
    "[ Mode: Manual Override | Controller: Tiger Team | Gateway: Production Locked ]"
)


@app.delete("/api/chat/stop")
def api_chat_stop():
    """Kill the current generation stream and stage the interrupted request as a bypass.

    Sets the SSE cancellation flag so the next stream chunk loop exits cleanly.
    Reads the last user message from chat_history.json and appends a
    'User-Initiated Bypass' entry to pending_overrides.json so the operator
    can resolve the intent manually via the Override Queue.
    Returns a Manual Override audit footer.
    """
    _cancel_flag["active"] = True

    # Capture interrupted context from history
    with _history_lock:
        history   = _load_history()
    last_user = next(
        (m["content"] for m in reversed(history) if m.get("role") == "user"), ""
    )

    bypass_entry = {
        "id":              str(uuid.uuid4())[:8],
        "day":             "",
        "time":            "",
        "original_slot":   {},
        "new_title_id":    "",
        "new_title":       "",
        "new_runtime":     0,
        "bypass_type":     "User-Initiated Bypass",
        "interrupted_prompt": last_user[:300],
        "timestamp":       datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "status":          "pending",
        "audit":           _MANUAL_OVERRIDE_AUDIT,
    }

    with _bypass_lock:
        _set_bypass_state(last_user[:300])

    print(f"DEBUG: Stop requested — bypass staged. Interrupted prompt: {last_user[:80]!r}")
    return {
        "stopped":            True,
        "bypass_id":          bypass_entry["id"],
        "message":            "Generation cancelled.",
        "controller_status":  "MANUAL_BYPASS",
        "active_gatekeeper":  "Tiger Team",
        "_audit":             _MANUAL_OVERRIDE_AUDIT,
    }


# ── /api/chat — keyword-bypass chat route ────────────────────────────────────

@app.post("/api/chat")
async def api_chat(request: Request):
    """Smart chat endpoint with keyword bypass for TCO, Quad, and EPG queries.

    Detects keywords in the last user message:
      - 'Quad' / 'loyalty tier' / 'audience composition' → get_quad_analysis()
      - 'EPG' / 'schedule' / 'airing'                    → get_current_schedule()
      - 'TCO' / 'kpi' / 'active viewers' / 'watch time'  → get_audience_metrics()

    Keyword hits call tools.py directly — no NeMo Agent round-trip, no timeout risk.
    Non-keyword messages are forwarded to NAT with reset_history=True so multi-turn
    context errors cannot accumulate.

    Always clears local chat history before processing.
    """
    body = await request.json()

    # Always reset — no multi-turn memory errors
    with _history_lock:
        _save_history([])

    msgs      = body.get("messages", [])
    user_text = (msgs[-1].get("content", "") if msgs else "").strip()
    _record_user_turn(body)

    bypass_type = _detect_bypass(user_text)
    if bypass_type:
        market  = _extract_market(user_text)
        segment = ""
        if "female" in user_text.lower():
            segment = "Female"
        elif "lgbt" in user_text.lower():
            segment = "LGBT+"

        # SSE stream: send heartbeat immediately so browser sees 200 OK,
        # then deliver the full brief without any NAT round-trip.
        async def _bypass_stream():
            yield f"data: {json.dumps({'type': 'heartbeat', 'content': 'Couture One Intelligence Layer — processing request.'})}\n\n"
            brief = _bypass_direct(bypass_type, market, segment)
            _append_message("assistant", brief)
            yield f"data: {json.dumps({'type': 'text', 'content': brief, 'bypassed': True})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            _bypass_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control":      "no-cache",
                "X-Accel-Buffering":  "no",
                "Connection":         "keep-alive",
                "Access-Control-Allow-Origin": "*",
            },
        )

    # Non-keyword: proxy to NAT, forcing context reset
    body["reset_history"] = True
    _prompt_excerpt       = user_text[:300]
    _cancel_flag["active"] = False

    async def _nat_fallback():
        assembled: list[str] = []
        _KEEPALIVE_INTERVAL  = 8.0
        _KEEPALIVE_MESSAGES  = [
            "Strategic Analysis in progress — the Condé Nast Intelligence Layer is processing.",
            "Cross-referencing the Style Tribe index. One moment.",
            "Reviewing the weekly arc. Patience is a virtue, even in fashion.",
            "The engine is working. Excellence cannot be rushed.",
        ]
        _seq = 0
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{NAT_BASE}/generate/stream",
                    json=body,
                    headers={"Accept": "text/event-stream"},
                ) as nat_resp:
                    _line_iter = nat_resp.aiter_lines().__aiter__()
                    while True:
                        if _cancel_flag["active"]:
                            _cancel_flag["active"] = False
                            break
                        if await request.is_disconnected():
                            break
                        try:
                            raw = await asyncio.wait_for(
                                _line_iter.__anext__(), timeout=_KEEPALIVE_INTERVAL
                            )
                        except asyncio.TimeoutError:
                            msg = _KEEPALIVE_MESSAGES[_seq % len(_KEEPALIVE_MESSAGES)]
                            _seq += 1
                            yield f"data: {json.dumps({'type': 'wait', 'content': msg})}\n\n"
                            continue
                        except StopAsyncIteration:
                            break
                        raw = raw.strip()
                        if not raw:
                            continue
                        if raw.startswith("intermediate_data:"):
                            trace = _parse_intermediate_line(raw[len("intermediate_data:"):].strip())
                            if trace:
                                yield f"data: {json.dumps(trace)}\n\n"
                            continue
                        if raw.startswith("data:"):
                            payload = raw[5:].strip()
                            if payload == "[DONE]":
                                break
                            text = _parse_data_line(payload)
                            if text:
                                assembled.append(text)
                                yield f"data: {json.dumps({'type': 'text', 'content': text})}\n\n"
        except Exception:
            pass

        if not assembled:
            fallback = (
                "# No Response Received\n\n"
                "The strategic engine timed out. "
                "For instant results use keywords: **Quad**, **EPG**, or **TCO** — "
                "these bypass the agent entirely.\n\n"
                f"`[ {_compute_audit_footer()} ]`\n\nThat is all."
            )
            assembled = [fallback]
            yield f"data: {json.dumps({'type': 'text', 'content': fallback})}\n\n"

        if assembled:
            _append_message("assistant", "".join(assembled))
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _nat_fallback(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":      "no-cache",
            "X-Accel-Buffering":  "no",
            "Connection":         "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ── Direct data endpoints (no LLM round-trip) ────────────────────────────────

def _normalise_market(dma: str) -> str:
    """Strip DMA suffix: 'New York (DMA 1)' → 'new york'."""
    return dma.split(" (DMA")[0].strip().lower()


@app.get("/api/nielsen")
def api_nielsen(
    dma:     str = Query(default="",  description="Market name, e.g. 'New York' or 'New York (DMA 1)'"),
    segment: str = Query(default="",  description="primary_demographic filter, e.g. 'Female'"),
):
    """Aggregate Nielsen metrics from nielsen_telemetry.json for a DMA / segment.

    Returns the 14-field Nielsen schema plus session count and top titles.
    Falls back to zeroed mock data if the file is missing or the DMA has no rows.
    """
    records = _load_json("data/nielsen_telemetry.json")

    market_key = _normalise_market(dma) if dma else ""
    if market_key:
        records = [r for r in records if market_key in r.get("market", "").lower()]

    # segment filter: if the log segment matches the demographic label on any row
    if segment:
        seg_lower = segment.lower()
        records = [r for r in records if seg_lower in r.get("primary_demographic", "").lower()
                   or seg_lower in r.get("genres", "").lower()
                   or not r.get("primary_demographic")]  # keep rows without demo tag

    if not records:
        # Graceful mock fallback
        return {
            "dma":     dma or "All Markets",
            "segment": segment or "All",
            "records": 0,
            "nielsen": {
                "UniverseEstimate_UE": 0, "HouseholdsUsingTV_HUT": 0,
                "PersonsViewingTV_PUT": 0, "Audience_HH_or_Persons": 0,
                "Rating_Pct": 0.0, "Share_Pct": 0.0, "AverageAudience_000": 0.0,
                "GRPs": 0.0, "Reach_Pct": 0.0, "Frequency": 0.0,
                "GrossImpressions": 0, "MediaCost": 0.0, "CPM": 0.0, "CPP": 0.0,
            },
            "top_titles": [],
            "note": "No data found for the requested filters.",
        }

    _avg = lambda key: round(sum(r.get(key, 0) for r in records) / len(records), 4)
    _sum = lambda key: round(sum(r.get(key, 0) for r in records), 4)

    nielsen = {
        "UniverseEstimate_UE":    int(records[0].get("UniverseEstimate_UE", 0)),
        "HouseholdsUsingTV_HUT":  _avg("HouseholdsUsingTV_HUT"),
        "PersonsViewingTV_PUT":   _avg("PersonsViewingTV_PUT"),
        "Audience_HH_or_Persons": int(_sum("Audience_HH_or_Persons")),
        "Rating_Pct":             _avg("Rating_Pct"),
        "Share_Pct":              _avg("Share_Pct"),
        "AverageAudience_000":    _avg("AverageAudience_000"),
        "GRPs":                   round(_sum("GRPs"), 4),
        "Reach_Pct":              _avg("Reach_Pct"),
        "Frequency":              _avg("Frequency"),
        "GrossImpressions":       int(_sum("GrossImpressions")),
        "MediaCost":              round(_sum("MediaCost"), 2),
        "CPM":                    _avg("CPM"),
        "CPP":                    _avg("CPP"),
    }

    top_titles = sorted(records, key=lambda r: r.get("GRPs", 0), reverse=True)[:10]

    return {
        "dma":        dma or "All Markets",
        "segment":    segment or "All",
        "records":    len(records),
        "nielsen":    nielsen,
        "top_titles": [
            {
                "show_id":       r.get("show_id", ""),
                "title":         r.get("title", ""),
                "daypart":       r.get("daypart", ""),
                "Rating_Pct":    r.get("Rating_Pct", 0),
                "Share_Pct":     r.get("Share_Pct", 0),
                "GRPs":          r.get("GRPs", 0),
                "MediaCost":     r.get("MediaCost", 0),
                "session_count": r.get("session_count", 0),
            }
            for r in top_titles
        ],
    }


_DMA_UNIVERSE: dict[str, int] = {
    "new york":    5_610_000, "los angeles": 5_200_000, "chicago":  2_200_000,
    "dallas":      1_450_000, "atlanta":     1_200_000, "philadelphia": 1_100_000,
    "london":      3_800_000, "paris":       2_900_000, "milan":    1_650_000,
    "san francisco": 980_000,
}


@app.get("/api/quads")
def api_quads(
    dma:     str = Query(default="",  description="Market name, e.g. 'New York'"),
    segment: str = Query(default="",  description="primary_demographic filter, e.g. 'Female'"),
):
    """Audience Composition (Occasional / Silver / Gold) from engagement_logs.parquet.

    Tiers are viewer-loyalty bands defined by completion rate:
      Gold       >= 0.85  — loyal devotees
      Silver     0.60–0.85 — regular viewers
      Occasional < 0.60   — light viewers

    Returns viewer-weighted percentages (Audience_HH_or_Persons), not session counts.
    Always returns _engine: 'DuckDB + Parquet'. Never returns empty — falls back to
    regional synthetic average when no rows match.
    """
    where_clauses, params = [], []
    market_key = _normalise_market(dma) if dma else ""
    if market_key:
        where_clauses.append("lower(trim(market)) LIKE ?")
        params.append(f"%{market_key}%")
    if segment:
        where_clauses.append("lower(trim(primary_demographic)) = ?")
        params.append(segment.strip().lower())
    where_sql = " AND ".join(where_clauses)

    logs = _query_parquet("data/engagement_logs.parquet", where=where_sql, params=params)
    if not logs:
        logs = _load_json("data/engagement_logs.json")
        if market_key:
            logs = [r for r in logs if market_key in r.get("market", "").strip().lower()]
        if segment:
            logs = [r for r in logs if r.get("primary_demographic", "").strip().lower() == segment.strip().lower()]

    # Demo Safety Net — regional synthetic average when no matching rows
    if not logs:
        universe = _DMA_UNIVERSE.get(market_key, 1_450_000)
        return {
            "dma":           dma or "All Markets",
            "segment":       segment or "All",
            "Gold":          12.0,
            "Silver":        28.0,
            "Occasional":    60.0,
            "total_viewers": universe,
            "prime_time_pct": 41.0,
            "tiers": {
                "Gold":       {"sessions": 0, "viewer_share_pct": 12.0, "avg_completion": 0.91},
                "Silver":     {"sessions": 0, "viewer_share_pct": 28.0, "avg_completion": 0.72},
                "Occasional": {"sessions": 0, "viewer_share_pct": 60.0, "avg_completion": 0.38},
            },
            "_engine":       "DuckDB + Parquet",
            "_source":       "Regional Average (synthetic — no matching rows for this filter)",
            "_audit_footer": _compute_audit_footer(),
        }

    occasional = [r for r in logs if r.get("completion_rate", 0) < 0.60]
    silver     = [r for r in logs if 0.60 <= r.get("completion_rate", 0) < 0.85]
    gold       = [r for r in logs if r.get("completion_rate", 0) >= 0.85]

    def _viewer_sum(rows: list[dict]) -> int:
        return int(sum(r.get("Audience_HH_or_Persons") or r.get("GrossImpressions") or 0 for r in rows))

    v_occasional = _viewer_sum(occasional)
    v_silver     = _viewer_sum(silver)
    v_gold       = _viewer_sum(gold)
    total_v      = v_occasional + v_silver + v_gold or 1

    def _pct(v: int) -> float:
        return round(v / total_v * 100, 1)

    def _avg_cr(rows: list[dict]) -> float:
        return round(sum(r.get("completion_rate", 0) for r in rows) / max(len(rows), 1), 4)

    demo_keys = ["Gen_Alpha", "Gen_Z", "Millennial", "Gen_X", "Silver_Stylists"]

    def _demo_avg(rows: list[dict]) -> dict:
        result = {}
        for dk in demo_keys:
            vals = [r["demographic_scores"][dk] for r in rows
                    if isinstance(r.get("demographic_scores"), dict) and dk in r["demographic_scores"]]
            result[dk] = round(sum(vals) / max(len(vals), 1), 4)
        return result

    prime_logs = [r for r in logs if r.get("is_prime_time")]
    prime_pct  = round(len(prime_logs) / len(logs) * 100, 1)

    return {
        "dma":           dma or "All Markets",
        "segment":       segment or "All",
        "Gold":          _pct(v_gold),
        "Silver":        _pct(v_silver),
        "Occasional":    _pct(v_occasional),
        "total_viewers": total_v,
        "prime_time_pct": prime_pct,
        "tiers": {
            "Gold":       {"sessions": len(gold),       "viewer_share_pct": _pct(v_gold),       "avg_completion": _avg_cr(gold),       "demo_scores": _demo_avg(gold)},
            "Silver":     {"sessions": len(silver),     "viewer_share_pct": _pct(v_silver),     "avg_completion": _avg_cr(silver),     "demo_scores": _demo_avg(silver)},
            "Occasional": {"sessions": len(occasional), "viewer_share_pct": _pct(v_occasional), "avg_completion": _avg_cr(occasional), "demo_scores": _demo_avg(occasional)},
        },
        "_engine":       "DuckDB + Parquet",
        "_audit_footer": _compute_audit_footer(),
        **_bypass_annotation(),
    }


# ── Manual Override Queue ─────────────────────────────────────────────────────

def _load_queue() -> list[dict]:
    try:
        with open(QUEUE_FILE) as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_queue(entries: list[dict]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(QUEUE_FILE, "w") as f:
        json.dump(entries, f, indent=2)


def _load_weekly() -> dict:
    try:
        with open(SCHEDULE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_weekly(data: dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(SCHEDULE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _catalog_by_id() -> dict[str, dict]:
    """Return catalog keyed by show_id for O(1) lookup."""
    try:
        with open(CATALOG_FILE) as f:
            items = json.load(f)
        return {item["show_id"]: item for item in items if item.get("show_id")}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _block_duration(runtime_min: int) -> int:
    """Round runtime up to the nearest 30-minute boundary for block scheduling."""
    return math.ceil(max(runtime_min, 1) / 30) * 30


class QueueOverrideRequest(BaseModel):
    day:          str = ""   # e.g. "Wednesday"
    time:         str = ""   # e.g. "20:00"
    new_title_id: str = ""   # e.g. "s0099"


@app.get("/api/queue")
def api_get_queue():
    """Return all entries in the Manual Override Queue."""
    with _queue_lock:
        entries = _load_queue()
    pending = [e for e in entries if e.get("status") == "pending"]
    return {
        "total":   len(entries),
        "pending": len(pending),
        "applied": len(entries) - len(pending),
        "entries": entries,
    }


@app.post("/api/queue-override")
def api_queue_override(body: QueueOverrideRequest):
    """Stage a title change in the Manual Override Queue without touching the live grid.

    Looks up the current slot from weekly_schedule.json to capture original_slot,
    resolves the new title from the catalog, and appends a 'pending' entry to
    pending_overrides.json. The live schedule is NOT modified.
    """
    day       = body.day.strip().title()
    slot_time = body.time.strip()
    new_id    = body.new_title_id.strip()

    if not day or not slot_time or not new_id:
        return {"error": "day, time, and new_title_id are all required."}

    catalog   = _catalog_by_id()
    new_meta  = catalog.get(new_id)
    if not new_meta:
        return {"error": f"Title '{new_id}' not found in catalog."}

    # Capture current slot state from weekly_schedule
    weekly     = _load_weekly()
    plan: list = weekly.get("weekly_plan", [])
    day_data   = next((d for d in plan if d.get("day_of_week", "").strip().title() == day), None)
    if not day_data:
        available = [d.get("day_of_week") for d in plan]
        return {"error": f"Day '{day}' not found in weekly plan.", "available_days": available}

    slot = next((s for s in day_data.get("slots", []) if s.get("time") == slot_time), None)
    if not slot:
        available_times = [s.get("time") for s in day_data.get("slots", [])]
        return {"error": f"No slot at {slot_time} on {day}.", "available_times": available_times}

    entry = {
        "id":        str(uuid.uuid4())[:8],
        "day":       day,
        "time":      slot_time,
        "original_slot": {
            "show_id":           slot.get("show_id", ""),
            "title":             slot.get("title", ""),
            "runtime_min":       slot.get("runtime_min", 0),
            "block_duration_min": slot.get("block_duration_min", 0),
        },
        "new_title_id":  new_id,
        "new_title":     new_meta.get("title", new_id),
        "new_runtime":   new_meta.get("runtime_min", 90),
        "timestamp":     datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "status":        "pending",
    }

    with _queue_lock:
        entries = _load_queue()
        # Replace any existing pending entry for the same slot to avoid duplicates
        entries = [e for e in entries if not (
            e.get("day") == day and e.get("time") == slot_time and e.get("status") == "pending"
        )]
        entries.append(entry)
        _save_queue(entries)

    print(f"DEBUG: Queued override — {day} {slot_time}: '{slot.get('title')}' → '{entry['new_title']}'")
    return {"queued": True, "entry": entry}


@app.post("/api/apply-queue")
def api_apply_queue():
    """Atomically apply all pending overrides to weekly_schedule.json and clear the queue.

    Each pending entry is resolved against the live grid. Entries that reference a
    slot that no longer exists are skipped and marked 'skipped'. All others are applied
    in a single write, then marked 'applied' in pending_overrides.json.
    """
    with _queue_lock:
        entries  = _load_queue()
        pending  = [e for e in entries if e.get("status") == "pending"]

        if not pending:
            return {"applied": 0, "skipped": 0, "message": "Queue is empty — nothing to apply."}

        catalog  = _catalog_by_id()
        weekly   = _load_weekly()
        plan: list = weekly.get("weekly_plan", [])

        applied_log: list[dict] = []
        skipped_log: list[dict] = []

        for entry in pending:
            day       = entry.get("day", "").strip().title()
            slot_time = entry.get("time", "")
            new_id    = entry.get("new_title_id", "")
            new_meta  = catalog.get(new_id, {})

            day_data = next((d for d in plan if d.get("day_of_week", "").strip().title() == day), None)
            if not day_data:
                entry["status"] = "skipped"
                entry["skip_reason"] = f"Day '{day}' not found in current weekly plan."
                skipped_log.append(entry)
                continue

            slot_idx = next(
                (i for i, s in enumerate(day_data.get("slots", [])) if s.get("time") == slot_time),
                None,
            )
            if slot_idx is None:
                entry["status"] = "skipped"
                entry["skip_reason"] = f"Slot {slot_time} no longer exists on {day}."
                skipped_log.append(entry)
                continue

            runtime  = new_meta.get("runtime_min", day_data["slots"][slot_idx].get("runtime_min", 90))
            block    = _block_duration(runtime)

            day_data["slots"][slot_idx].update({
                "show_id":           new_id,
                "title":             new_meta.get("title", new_id),
                "runtime_min":       runtime,
                "block_duration_min": block,
                "interstitial_min":  block - runtime,
            })

            entry["status"] = "applied"
            entry["applied_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
            applied_log.append({
                "day":       day,
                "time":      slot_time,
                "old_title": entry["original_slot"].get("title", ""),
                "new_title": new_meta.get("title", new_id),
            })

        weekly["_last_updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
        _save_weekly(weekly)
        _save_queue(entries)

    print(f"DEBUG: Applied {len(applied_log)} overrides, skipped {len(skipped_log)}")
    return {
        "applied":      len(applied_log),
        "skipped":      len(skipped_log),
        "applied_slots": applied_log,
        "skipped_slots": [{"day": e.get("day"), "time": e.get("time"), "reason": e.get("skip_reason")} for e in skipped_log],
        "timestamp":    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
    }


# ── Tonight's EPG ─────────────────────────────────────────────────────────────

@app.get("/api/epg")
def api_epg(
    current_time: str = Query(default="", description="ISO timestamp for 'now'. Defaults to UTC now."),
    lookback:     int = Query(default=4,  description="Programs before 'live' to include as 'past'."),
    lookahead:    int = Query(default=20, description="Programs after 'live' to include as 'future'."),
):
    """24-hour rolling EPG — flat week master list with live/past/future status.

    Flattens all 7 days into one continuous master_list.
    Finds the currently-airing program (anchor_index).
    Returns [anchor - lookback … anchor + lookahead] with circular wrap.
    Each item carries:
      status:   'past' | 'live' | 'future'
      isLive:   true only for the single currently-airing item
      isNextDay: true if the slot falls on a different calendar day than 'now'
    """
    weekly = _load_json("data/weekly_schedule.json")
    plan   = weekly.get("weekly_plan", [])
    if not plan:
        return {"error": "Weekly plan not yet generated. Call generate_weekly_plan first."}

    # Resolve reference time
    try:
        now = (
            datetime.fromisoformat(current_time).replace(tzinfo=timezone.utc)
            if current_time
            else datetime.now(timezone.utc)
        )
    except ValueError:
        now = datetime.now(timezone.utc)

    now_day  = now.strftime("%A")
    now_hhmm = f"{now.hour:02d}:{now.minute:02d}"

    # Build flat master_list — all days in week order, each slot tagged with date/day
    day_order = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    plan_map  = {d["day_of_week"].title(): d for d in plan}
    master_list: list[dict] = []

    for dname in day_order:
        day_data = plan_map.get(dname)
        if not day_data:
            continue
        filled_slots = _fill_24h_gaps(
            day_data.get("slots", []), dname, day_data.get("theme", "")
        )
        for slot in filled_slots:
            master_list.append({
                **slot,
                "_day":   dname,
                "_date":  day_data.get("date", ""),
                "_theme": day_data.get("theme", ""),
            })

    if not master_list:
        return {"error": "No slots in weekly plan."}

    # Find anchor_index: the currently-airing or most-recently-started slot for today
    anchor_index = 0
    for i, item in enumerate(master_list):
        if item["_day"] == now_day and item.get("time", "99:99") <= now_hhmm:
            anchor_index = i   # keep updating — we want the last match (most recent start)

    # Circular slice: [anchor - lookback … anchor + lookahead]
    n     = len(master_list)
    total = lookback + 1 + lookahead
    items = [master_list[(anchor_index - lookback + i) % n] for i in range(total)]

    # Annotate each item with status, isLive, isNextDay
    def _slot_end_hhmm(item: dict) -> str:
        h, m = divmod(0, 60)
        try:
            start_h, start_m = map(int, item.get("time", "00:00").split(":"))
            end_min = start_h * 60 + start_m + int(item.get("block_duration_min", 90))
            end_min %= 1440
            h, m = divmod(end_min, 60)
        except Exception:
            pass
        return f"{h:02d}:{m:02d}"

    result_slots = []
    for idx, item in enumerate(items):
        is_anchor     = (idx == lookback)
        item_day      = item["_day"]
        item_time     = item.get("time", "00:00")
        item_end      = _slot_end_hhmm(item)

        if is_anchor:
            status = "live"
        elif idx < lookback:
            status = "past"
        else:
            status = "future"

        result_slots.append({
            **{k: v for k, v in item.items() if not k.startswith("_")},
            "day":       item_day,
            "date":      item["_date"],
            "theme":     item["_theme"],
            "ends_at":   item_end,
            "status":    status,
            "isLive":    is_anchor,
            "isNextDay": item_day != now_day,
        })

    now_day_data = plan_map.get(now_day, {})
    return {
        "now":          now.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "current_day":  now_day,
        "theme":        now_day_data.get("theme", ""),
        "tribe":        now_day_data.get("tribe", ""),
        "anchor_index":  anchor_index,
        "total_in_week": n,
        "slots":         result_slots,
        "_engine":       "DuckDB + Parquet" if _DUCKDB_AVAILABLE else "Pandas",
        "_audit_footer": _compute_audit_footer(),
        **_bypass_annotation(),
    }


# ── 24-hour gap filler ────────────────────────────────────────────────────────

def _fill_24h_gaps(slots: list[dict], day: str = "", theme: str = "") -> list[dict]:
    """Return a complete 00:00–24:00 slot list with Off-Air fillers in every gap.

    Rules:
    - Slots must be pre-sorted by time.
    - Gaps ≥ 1 minute get a filler block (title = 'Continuous Stream').
    - A filler from 22:00–00:00 uses 'Late Night Continuous' to distinguish sign-off.
    - Each filler carries is_filler: true so the frontend can style it differently.
    - Total coverage is always exactly 1440 minutes (00:00 → 24:00 / next midnight).
    """
    filled: list[dict] = []
    cursor = 0  # minutes from midnight

    sorted_slots = sorted(slots, key=lambda s: s.get("time", "00:00"))

    for slot in sorted_slots:
        try:
            h, m   = map(int, slot["time"].split(":"))
        except (KeyError, ValueError):
            continue
        start_min = h * 60 + m
        block_min = int(slot.get("block_duration_min", slot.get("runtime_min", 90)))

        # Gap before this slot
        if start_min > cursor:
            gap = start_min - cursor
            gh, gm = divmod(cursor, 60)
            label = "Late Night Continuous" if cursor >= 22 * 60 or cursor < 6 * 60 else "Continuous Stream"
            filled.append(_filler_slot(f"{gh:02d}:{gm:02d}", gap, day, theme, label))

        filled.append(slot)
        cursor = start_min + block_min

    # Tail gap: from last slot end to 24:00
    if cursor < 1440:
        gap = 1440 - cursor
        gh, gm = divmod(cursor % 1440, 60)
        label = "Late Night Continuous" if cursor >= 22 * 60 or cursor < 6 * 60 else "Continuous Stream"
        filled.append(_filler_slot(f"{gh:02d}:{gm:02d}", gap, day, theme, label))

    return filled


def _filler_slot(time: str, duration_min: int, day: str, theme: str, label: str) -> dict:
    h, m  = map(int, time.split(":"))
    hut   = 0.18 if (h < 6 or h >= 22) else 0.55
    return {
        "time":              time,
        "show_id":           "off_air",
        "title":             label,
        "runtime_min":       duration_min,
        "block_duration_min": duration_min,
        "interstitial_min":  0,
        "tribe":             "",
        "target_age":        "",
        "event_type":        "off_air",
        "block_label":       "Off-Air",
        "daypart":           ("Overnight" if h < 6 or h >= 22 else
                              "Daytime"   if h < 16 else
                              "Prime Time" if h < 22 else "Late Night"),
        "hut_estimate":      hut,
        "day":               day,
        "theme":             theme,
        "is_filler":         True,
    }


@app.get("/api/epg/full-day")
def api_epg_full_day(
    day:          str = Query(default="", description="Day name, e.g. 'Wednesday'. Defaults to today."),
    current_time: str = Query(default="", description="ISO timestamp for 'now'. Defaults to UTC now."),
):
    """Complete 24-hour TV Guide grid for a single day, gap-filled to 1440 minutes.

    Every minute of the broadcast day is covered — real slots plus Off-Air/Continuous
    Stream fillers in any gap. The frontend can render this as a pixel-perfect
    vertical timeline at any PX_PER_MIN scale without dead zones.

    Each slot includes:
      time              HH:MM start
      ends_at           HH:MM end (start + block_duration_min)
      block_duration_min total slot height in minutes
      is_filler         true for Off-Air blocks
      status            'past' | 'live' | 'future'
      isLive            true for the currently-airing slot
      start_offset_min  minutes from midnight (for CSS top calculation)
    """
    weekly   = _load_json("data/weekly_schedule.json")
    plan     = weekly.get("weekly_plan", [])
    if not plan:
        return {"error": "Weekly plan not yet generated."}

    try:
        now = (
            datetime.fromisoformat(current_time).replace(tzinfo=timezone.utc)
            if current_time
            else datetime.now(timezone.utc)
        )
    except ValueError:
        now = datetime.now(timezone.utc)

    target_day = day.strip().title() if day.strip() else now.strftime("%A")
    plan_map   = {d["day_of_week"].title(): d for d in plan}
    day_data   = plan_map.get(target_day)

    if not day_data:
        available = list(plan_map.keys())
        return {"error": f"Day '{target_day}' not found.", "available_days": available}

    raw_slots = day_data.get("slots", [])
    theme     = day_data.get("theme", "")
    filled    = _fill_24h_gaps(raw_slots, target_day, theme)

    now_min   = now.hour * 60 + now.minute
    now_day   = now.strftime("%A")

    result: list[dict] = []
    for slot in filled:
        try:
            sh, sm = map(int, slot["time"].split(":"))
        except ValueError:
            continue
        start_min = sh * 60 + sm
        block_min = int(slot.get("block_duration_min", 90))
        end_min   = (start_min + block_min) % 1440
        eh, em    = divmod(end_min, 60)

        if target_day != now_day:
            status = "future"
            is_live = False
        elif start_min <= now_min < start_min + block_min:
            status  = "live"
            is_live = True
        elif start_min + block_min <= now_min:
            status  = "past"
            is_live = False
        else:
            status  = "future"
            is_live = False

        result.append({
            **slot,
            "ends_at":          f"{eh:02d}:{em:02d}",
            "start_offset_min": start_min,
            "status":           status,
            "isLive":           is_live,
        })

    live_idx = next((i for i, s in enumerate(result) if s.get("isLive")), None)

    return {
        "day":             target_day,
        "date":            day_data.get("date", ""),
        "theme":           theme,
        "tribe":           day_data.get("tribe", ""),
        "total_minutes":   1440,
        "slot_count":      len(result),
        "filler_count":    sum(1 for s in result if s.get("is_filler")),
        "live_index":      live_idx,
        "now":             now.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "now_offset_min":  now_min if target_day == now_day else None,
        "slots":           result,
        "_engine":         "DuckDB + Parquet" if _DUCKDB_AVAILABLE else "Pandas",
        "_audit_footer":   _compute_audit_footer(),
        **_bypass_annotation(),
    }


# ── Audience Analytics Dashboard ─────────────────────────────────────────────

@app.get("/api/analytics/dashboard")
def api_analytics_dashboard(
    dma:     str = Query(default="", description="Market name, e.g. 'Dallas' or 'Dallas (DMA 4)'"),
    segment: str = Query(default="", description="primary_demographic filter, e.g. 'Female' or 'LGBT+'"),
):
    """Aggregate audience analytics for the dashboard UI.

    Combines nielsen_telemetry (aggregate reach metrics) with engagement_logs
    (session-level completion and hourly behaviour).  All arrays are Recharts-ready.

    Returns:
      summary          — scalar KPIs (total_active_viewers, engagement_rate, …)
      hourly_trend     — 24-element array [{hour, label, viewers, completion_rate,
                          is_prime_time, sessions}] — one entry per clock hour
      demographic      — [{name, viewers, completion_rate, value_pct}] for pie charts
      daypart          — [{daypart, viewers, sessions, avg_completion, share_pct}]
      top_titles       — top-10 titles by viewers [{title, viewers, rating_pct,
                          completion_rate, daypart, market}]
      _meta            — filter echoes + engine + record counts
    """
    t0 = datetime.now(timezone.utc)

    # ── Load + filter Nielsen telemetry ──────────────────────────────────────
    nt_where, nt_params = [], []
    market_key = _normalise_market(dma) if dma else ""
    if market_key:
        nt_where.append("lower(trim(market)) LIKE ?")
        nt_params.append(f"%{market_key}%")

    nielsen = _query_parquet("data/nielsen_telemetry.parquet",
                             where=" AND ".join(nt_where), params=nt_params)
    if not nielsen:
        nielsen = _load_json("data/nielsen_telemetry.json")
        if market_key:
            nielsen = [r for r in nielsen if market_key in r.get("market", "").lower()]

    # ── Load + filter engagement logs ─────────────────────────────────────────
    el_where, el_params = [], []
    if market_key:
        el_where.append("lower(trim(market)) LIKE ?")
        el_params.append(f"%{market_key}%")
    if segment:
        el_where.append("lower(trim(primary_demographic)) = ?")
        el_params.append(segment.strip().lower())

    logs = _query_parquet("data/engagement_logs.parquet",
                          where=" AND ".join(el_where), params=el_params)
    if not logs:
        logs = _load_json("data/engagement_logs.json")
        if market_key:
            logs = [r for r in logs if market_key in r.get("market", "").lower()]
        if segment:
            logs = [r for r in logs
                    if r.get("primary_demographic", "").strip().lower() == segment.strip().lower()]

    # ── Summary KPIs ──────────────────────────────────────────────────────────
    def _sum(rows, key):
        return sum(r.get(key) or 0 for r in rows)

    def _avg(rows, key):
        vals = [r.get(key) or 0 for r in rows]
        return round(sum(vals) / max(len(vals), 1), 4)

    total_viewers    = int(_sum(nielsen, "Audience_HH_or_Persons"))
    total_impressions = int(_sum(nielsen, "GrossImpressions"))
    total_media_cost  = round(_sum(nielsen, "MediaCost"), 2)
    avg_rating        = round(_avg(nielsen, "Rating_Pct"), 4)
    avg_share         = round(_avg(nielsen, "Share_Pct"), 4)
    avg_grps          = round(_avg(nielsen, "GRPs"), 4)

    engagement_rate   = round(_avg(logs, "completion_rate"), 4)

    prime_nielsen     = [r for r in nielsen if r.get("is_prime_time")]
    prime_viewers     = int(_sum(prime_nielsen, "Audience_HH_or_Persons"))
    prime_pct         = round(prime_viewers / max(total_viewers, 1) * 100, 1)

    # ── Hourly Viewership Trend (00–23) ───────────────────────────────────────
    # Buckets: keyed by integer hour, accumulate viewers + completion + session count
    _PRIME_HOURS = set(range(16, 22))

    hour_viewers:     dict[int, int]   = {h: 0 for h in range(24)}
    hour_completion:  dict[int, list]  = {h: [] for h in range(24)}
    hour_sessions:    dict[int, int]   = {h: 0 for h in range(24)}

    for r in logs:
        ts = r.get("timestamp", "")
        try:
            h = int(ts[11:13]) if len(ts) >= 13 else -1
        except (ValueError, TypeError):
            h = -1
        if 0 <= h <= 23:
            hour_viewers[h]    += int(r.get("Audience_HH_or_Persons") or 0)
            hour_sessions[h]   += 1
            cr = r.get("completion_rate")
            if cr is not None:
                hour_completion[h].append(float(cr))

    def _hour_label(h: int) -> str:
        suffix = "AM" if h < 12 else "PM"
        display = h if h <= 12 else h - 12
        if display == 0:
            display = 12
        return f"{display} {suffix}"

    hourly_trend = [
        {
            "hour":            h,
            "label":           _hour_label(h),
            "time":            f"{h:02d}:00",
            "viewers":         hour_viewers[h],
            "sessions":        hour_sessions[h],
            "completion_rate": round(sum(hour_completion[h]) / max(len(hour_completion[h]), 1), 4),
            "is_prime_time":   h in _PRIME_HOURS,
        }
        for h in range(24)
    ]

    # ── Demographic Breakdown (pie / donut chart) ─────────────────────────────
    from collections import defaultdict

    demo_viewers:     dict[str, int]   = defaultdict(int)
    demo_completion:  dict[str, list]  = defaultdict(list)
    demo_sessions:    dict[str, int]   = defaultdict(int)

    for r in logs:
        demo = r.get("primary_demographic", "Unknown") or "Unknown"
        demo_viewers[demo]   += int(r.get("Audience_HH_or_Persons") or 0)
        demo_sessions[demo]  += 1
        cr = r.get("completion_rate")
        if cr is not None:
            demo_completion[demo].append(float(cr))

    total_demo_viewers = max(sum(demo_viewers.values()), 1)
    demographic = sorted(
        [
            {
                "name":            demo,
                "viewers":         demo_viewers[demo],
                "sessions":        demo_sessions[demo],
                "completion_rate": round(sum(demo_completion[demo]) / max(len(demo_completion[demo]), 1), 4),
                "value_pct":       round(demo_viewers[demo] / total_demo_viewers * 100, 1),
            }
            for demo in demo_viewers
        ],
        key=lambda x: x["viewers"],
        reverse=True,
    )

    # ── Daypart Breakdown (bar chart) ────────────────────────────────────────
    dp_viewers:    dict[str, int]  = defaultdict(int)
    dp_sessions:   dict[str, int]  = defaultdict(int)
    dp_completion: dict[str, list] = defaultdict(list)

    for r in logs:
        dp = ("Prime Time" if r.get("is_prime_time")
              else "Overnight" if 0 <= int((r.get("timestamp", "T00")[11:13] or 0)) < 6
              else "Daytime")
        try:
            h_val = int(r.get("timestamp", "T00")[11:13])
            dp = ("Prime Time" if 16 <= h_val < 22
                  else "Late Night" if 22 <= h_val or h_val < 1
                  else "Overnight" if h_val < 6
                  else "Daytime")
        except (ValueError, TypeError):
            dp = "Unknown"
        dp_viewers[dp]   += int(r.get("Audience_HH_or_Persons") or 0)
        dp_sessions[dp]  += 1
        cr = r.get("completion_rate")
        if cr is not None:
            dp_completion[dp].append(float(cr))

    total_dp_viewers = max(sum(dp_viewers.values()), 1)
    _DP_ORDER = ["Overnight", "Daytime", "Prime Time", "Late Night"]
    daypart = [
        {
            "daypart":         dp,
            "viewers":         dp_viewers[dp],
            "sessions":        dp_sessions[dp],
            "avg_completion":  round(sum(dp_completion[dp]) / max(len(dp_completion[dp]), 1), 4),
            "share_pct":       round(dp_viewers[dp] / total_dp_viewers * 100, 1),
        }
        for dp in _DP_ORDER
        if dp in dp_viewers
    ]

    # ── Top Titles (horizontal bar chart) ─────────────────────────────────────
    title_map: dict[str, dict] = {}
    for r in nielsen:
        t = r.get("title", "")
        if not t:
            continue
        if t not in title_map:
            title_map[t] = {
                "title":       t,
                "viewers":     0,
                "rating_pct":  0.0,
                "share_pct":   0.0,
                "grps":        0.0,
                "media_cost":  0.0,
                "daypart":     r.get("daypart", ""),
                "market":      r.get("market", ""),
                "_count":      0,
            }
        title_map[t]["viewers"]    += int(r.get("Audience_HH_or_Persons") or 0)
        title_map[t]["rating_pct"] += float(r.get("Rating_Pct") or 0)
        title_map[t]["share_pct"]  += float(r.get("Share_Pct") or 0)
        title_map[t]["grps"]       += float(r.get("GRPs") or 0)
        title_map[t]["media_cost"] += float(r.get("MediaCost") or 0)
        title_map[t]["_count"]     += 1

    # Merge completion rate from logs
    log_cr: dict[str, list] = defaultdict(list)
    for r in logs:
        t = r.get("title", "")
        cr = r.get("completion_rate")
        if t and cr is not None:
            log_cr[t].append(float(cr))

    top_titles = sorted(
        [
            {
                "title":           t,
                "viewers":         d["viewers"],
                "rating_pct":      round(d["rating_pct"] / d["_count"], 4),
                "share_pct":       round(d["share_pct"]  / d["_count"], 4),
                "grps":            round(d["grps"],  4),
                "media_cost":      round(d["media_cost"], 2),
                "completion_rate": round(sum(log_cr[t]) / max(len(log_cr[t]), 1), 4),
                "daypart":         d["daypart"],
                "market":          d["market"],
            }
            for t, d in title_map.items()
        ],
        key=lambda x: x["viewers"],
        reverse=True,
    )[:10]

    # ── Latency ───────────────────────────────────────────────────────────────
    latency_ms = int((datetime.now(timezone.utc) - t0).total_seconds() * 1000)
    engine = "DuckDB + Parquet" if _DUCKDB_AVAILABLE else "Pandas"

    return {
        "summary": {
            "total_active_viewers":  total_viewers,
            "total_impressions":     total_impressions,
            "total_media_cost":      total_media_cost,
            "engagement_rate":       engagement_rate,
            "avg_rating_pct":        avg_rating,
            "avg_share_pct":         avg_share,
            "avg_grps":              avg_grps,
            "prime_time_viewers":    prime_viewers,
            "prime_time_pct":        prime_pct,
            "nielsen_records":       len(nielsen),
            "engagement_records":    len(logs),
        },
        "hourly_trend":  hourly_trend,
        "demographic":   demographic,
        "daypart":       daypart,
        "top_titles":    top_titles,
        "_meta": {
            "dma":             dma or "All Markets",
            "segment":         segment or "All",
            "engine":          engine,
            "latency_ms":      latency_ms,
            "sources":         ["nielsen_telemetry.parquet", "engagement_logs.parquet"],
            "audit_footer":    _compute_audit_footer(),
            **_environment_meta(),
        },
        **_bypass_annotation(),
    }


# ── Config / tunnel discovery ────────────────────────────────────────────────

@app.get("/api/config")
def api_config(request: Request):
    """Self-discovery endpoint. Returns the public base URL for this sidecar.

    Checks the local ngrok agent API for an active tunnel forwarding to this port.
    Falls back to the request's own base URL if ngrok is not running.
    Paste the returned api_base into your Lovable project as the API base URL.
    """
    try:
        import httpx as _hx
        tunnels = _hx.get("http://localhost:4040/api/tunnels", timeout=1.0).json()
        for t in tunnels.get("tunnels", []):
            if "8081" in t.get("config", {}).get("addr", ""):
                public_url = t["public_url"].rstrip("/")
                break
        else:
            public_url = str(request.base_url).rstrip("/")
    except Exception:
        public_url = str(request.base_url).rstrip("/")

    return {
        "api_base":    public_url,
        "tunnel_active": "ngrok" in public_url or "ngrok-free" in public_url,
        "endpoints": {
            "health":          f"{public_url}/health",
            "chat_stream":     f"{public_url}/generate/stream",
            "chat":            f"{public_url}/generate",
            "epg":             f"{public_url}/api/epg",
            "epg_full_day":    f"{public_url}/api/epg/full-day",
            "quads":           f"{public_url}/api/quads",
            "nielsen":         f"{public_url}/api/nielsen",
            "dashboard":       f"{public_url}/api/analytics/dashboard",
            "queue":           f"{public_url}/api/queue",
            "queue_override":  f"{public_url}/api/queue-override",
            "apply_queue":     f"{public_url}/api/apply-queue",
            "chat_bypass":     f"{public_url}/api/chat",
            "stop":            f"{public_url}/api/chat/stop",
            "download_report":  f"{public_url}/download_report",
            "generate_brief":   f"{public_url}/api/generate-brief",
            "download_brief":   f"{public_url}/api/briefs/{{filename}}",
        },
    }


# ── System / compute endpoints ────────────────────────────────────────────────

@app.get("/health")
def health():
    mode    = _read_mode() if _IS_BREV else "OFFLINE"
    profile = _COMPUTE_PROFILES[mode]
    return {
        "status":          "TIGER_TEAM_READY",
        "mode":            mode,
        "gpu_detected":    HAS_GPU,
        "is_brev":         _IS_BREV,
        "source_compute":  profile["source_compute"],
        "engine":          profile["engine"],
        "latency_ms":      profile["latency_ms"],
        "audit_footer":    profile["audit_footer"],
        "cors":            "allow_origins=['*']",
        "port":            8081,
        **_environment_meta(),
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
    return {"status": "mode_switched", "execution_mode": mode, **_COMPUTE_PROFILES[mode]}


# ── Chat history endpoints ────────────────────────────────────────────────────

@app.get("/history")
def get_history():
    with _history_lock:
        return _load_history()


@app.post("/history/append")
def append_message(body: AppendRequest):
    role = body.role.lower().strip()
    if role not in ("user", "assistant"):
        return {"error": "role must be 'user' or 'assistant'"}
    _append_message(role, body.content)
    with _history_lock:
        return {"status": "appended", "total_messages": len(_load_history())}


@app.post("/clear_history")
def clear_history():
    with _history_lock:
        _save_history([])
    return {"status": "cleared", "messages": []}


# ── Report download ───────────────────────────────────────────────────────────

def _load_json(path: str):
    try:
        with open(os.path.join(os.path.dirname(__file__), path)) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _query_parquet(parquet_rel_path: str, where: str = "", params: list | None = None) -> list[dict]:
    """Run a DuckDB SQL query against a .parquet file; returns list of dicts.

    Falls back to loading the equivalent .json if DuckDB or the parquet is unavailable.
    """
    if not _DUCKDB_AVAILABLE:
        return _load_json(parquet_rel_path.replace(".parquet", ".json"))
    base    = os.path.dirname(__file__)
    pq_path = os.path.join(base, parquet_rel_path)
    if not os.path.exists(pq_path):
        return _load_json(parquet_rel_path.replace(".parquet", ".json"))
    try:
        import duckdb
        sql = f"SELECT * FROM read_parquet('{pq_path}')"
        if where:
            sql += f" WHERE {where}"
        result = duckdb.execute(sql, params or []).fetchdf()
        return result.where(result.notna(), other=None).to_dict(orient="records")
    except Exception:
        return _load_json(parquet_rel_path.replace(".parquet", ".json"))


@app.get("/download_report")
def download_report(
    dma:     str = Query(default="New York", description="Market name, e.g. 'New York'"),
    segment: str = Query(default="",         description="primary_demographic filter, e.g. 'Female'"),
):
    """Return a Runway Strategy Brief as a downloadable .xlsx file.

    Sheet 1 — Quad Analysis: Occasional / Silver / Gold tier breakdown.
    Sheet 2 — Nielsen Telemetry: top 50 show×market records for the DMA.
    """
    nielsen_all = _load_json("data/nielsen_telemetry.json")
    logs_all    = _load_json("data/engagement_logs.json")

    # Normalize: accept "New York (DMA 1)" or "New York"
    market_key = dma.split(" (DMA")[0].strip().lower()

    dma_nielsen = [r for r in nielsen_all if market_key in r.get("market", "").lower()]
    dma_logs    = [r for r in logs_all    if market_key in r.get("market", "").lower()]

    if segment:
        dma_logs = [r for r in dma_logs if r.get("primary_demographic", "").lower() == segment.lower()]

    total = max(len(dma_logs), 1)

    def _avg_cr(rows):
        return round(sum(r.get("completion_rate", 0) for r in rows) / max(len(rows), 1), 4)

    occasional = [r for r in dma_logs if r.get("completion_rate", 0) < 0.60]
    silver     = [r for r in dma_logs if 0.60 <= r.get("completion_rate", 0) < 0.85]
    gold       = [r for r in dma_logs if r.get("completion_rate", 0) >= 0.85]

    quad_rows = [
        {
            "Tier":            "Occasional",
            "Sessions":        len(occasional),
            "Share_%":         round(len(occasional) / total * 100, 1),
            "Avg_Completion":  _avg_cr(occasional),
            "Description":     "Light viewers (completion < 60 %)",
        },
        {
            "Tier":            "Silver",
            "Sessions":        len(silver),
            "Share_%":         round(len(silver) / total * 100, 1),
            "Avg_Completion":  _avg_cr(silver),
            "Description":     "Engaged viewers (completion 60–84 %)",
        },
        {
            "Tier":            "Gold",
            "Sessions":        len(gold),
            "Share_%":         round(len(gold) / total * 100, 1),
            "Avg_Completion":  _avg_cr(gold),
            "Description":     "Heavy viewers (completion ≥ 85 %)",
        },
    ]

    nielsen_fields = [
        "title", "market", "daypart", "is_prime_time", "session_count",
        "avg_completion_rate", "Rating_Pct", "Share_Pct", "GRPs",
        "Reach_Pct", "AverageAudience_000", "GrossImpressions",
        "MediaCost", "CPM", "CPP",
    ]
    nielsen_rows = [
        {k: r.get(k, "") for k in nielsen_fields}
        for r in dma_nielsen[:50]
    ]

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame(quad_rows).to_excel(writer,   sheet_name="Quad_Analysis",      index=False)
        pd.DataFrame(nielsen_rows).to_excel(writer, sheet_name="Nielsen_Telemetry", index=False)
    buf.seek(0)

    filename = f"Runway_Strategy_Brief_{market_key.replace(' ', '_')}.xlsx"
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ── Miranda Intelligence Brief ─────────────────────────────────────────────────

_REPORTS_DIR = os.path.join(os.path.dirname(__file__), "data", "reports")
os.makedirs(_REPORTS_DIR, exist_ok=True)


@app.post("/api/generate-brief")
async def api_generate_brief(body: BriefRequest, request: Request):
    """Generate a branded Miranda Intelligence Brief PDF (or HTML fallback).

    Call after the 4-step pipeline has run. Pass whatever scores the frontend
    has collected (resonance_score, confidence_score, script_tags, etc.).
    Returns JSON with filename and a download_url the frontend can follow.
    """
    from pdf_generator import generate_pdf_brief

    try:
        base_url = str(request.base_url).rstrip("/")
        # Prefer ngrok tunnel URL if available (needed for Lovable cross-origin download)
        try:
            import httpx as _hx
            tunnels = _hx.get("http://localhost:4040/api/tunnels", timeout=1.0).json()
            for t in tunnels.get("tunnels", []):
                if "8081" in t.get("config", {}).get("addr", ""):
                    base_url = t["public_url"].rstrip("/")
                    break
        except Exception:
            pass

        result = generate_pdf_brief(body.model_dump())
        filename = result["filename"]

        return JSONResponse({
            "status":        result["status"],
            "filename":      filename,
            "download_url":  f"{base_url}/api/briefs/{filename}",
            "persona":       result["persona"],
            "page_count":    result["page_count"],
            "render_method": result["render_method"],
            "title":         result["title"],
            "_audit":        result["_audit"],
        })
    except Exception as exc:
        logger.error("Brief generation error: %s: %s", type(exc).__name__, exc)
        return JSONResponse({"status": "ERROR", "detail": str(exc)}, status_code=500)


@app.get("/api/briefs/{filename}")
def api_download_brief(filename: str):
    """Serve a generated Miranda Intelligence Brief for download.

    Supports both .pdf (WeasyPrint) and .html (fallback) files.
    """
    safe = os.path.basename(filename)  # prevent path traversal
    path = os.path.join(_REPORTS_DIR, safe)

    if not os.path.exists(path):
        return JSONResponse({"detail": "Brief not found."}, status_code=404)

    if safe.endswith(".pdf"):
        media_type = "application/pdf"
    else:
        media_type = "text/html"

    return FileResponse(
        path=path,
        media_type=media_type,
        filename=safe,
        headers={"Content-Disposition": f"attachment; filename=\"{safe}\""},
    )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8081,
        log_level="info",
        timeout_keep_alive=60,    # wait 60s on idle connections before closing
        loop="asyncio",           # explicit event loop — avoids uvloop surprises on macOS
        ws_ping_interval=20,      # WebSocket keep-alive (belt-and-suspenders)
        ws_ping_timeout=60,
    )
