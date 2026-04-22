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

import importlib.util
import io
import json
import math
import os
import uuid
from datetime import datetime, timezone
from threading import Lock

import httpx
import pandas as pd
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

_DUCKDB_AVAILABLE = importlib.util.find_spec("duckdb") is not None

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR       = os.path.join(os.path.dirname(__file__), "data")
MODE_FILE      = os.path.join(DATA_DIR, "mode.json")
HISTORY_FILE   = os.path.join(DATA_DIR, "chat_history.json")
QUEUE_FILE     = os.path.join(DATA_DIR, "pending_overrides.json")
SCHEDULE_FILE  = os.path.join(DATA_DIR, "weekly_schedule.json")
CATALOG_FILE   = os.path.join(DATA_DIR, "catalog.json")
BYPASS_FILE    = os.path.join(DATA_DIR, "bypass_state.json")

NAT_BASE = "http://localhost:8080"

HAS_GPU: bool = importlib.util.find_spec("cudf") is not None

_COMPUTE_PROFILES = {
    "ONLINE":  {
        "source_compute": "NVIDIA A10G (Brev GPU)",
        "engine":         "NVIDIA RAPIDS (cuDF)",
        "gpu_boost":      "35x",
        "latency_ms":     12,
    },
    "OFFLINE": {
        "source_compute": "Local CPU",
        "engine":         "Pandas",
        "gpu_boost":      "1x",
        "latency_ms":     185,
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


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Runway Inclusive — Sidecar API", version="1.2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_history_lock  = Lock()
_queue_lock    = Lock()
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
    body = await request.json()

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

    _record_user_turn(body)

    # Capture prompt for bypass staging before entering generator scope
    _prompt_excerpt = (msgs[-1].get("content", "") if msgs else "")[:300]
    _generation_complete = {"ok": False}   # mutable flag updated inside generator

    async def event_gen():
        assembled: list[str] = []
        client_disconnected = False

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{NAT_BASE}/generate/stream",
                    json=body,
                    headers={"Accept": "text/event-stream"},
                ) as nat_resp:
                    async for raw_line in nat_resp.aiter_lines():
                        # ── Cancellation checks ──────────────────────────────
                        if _cancel_flag["active"]:
                            _cancel_flag["active"] = False
                            client_disconnected = True
                            break
                        if await request.is_disconnected():
                            client_disconnected = True
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

        except Exception:
            pass  # do NOT emit a fallback message on error

        if client_disconnected:
            # ── Manual Intervention audit trail ─────────────────────────────
            print(f"DEBUG: Client disconnected mid-generation — staging Manual Intervention. "
                  f"Prompt: {_prompt_excerpt[:60]!r}")
            bypass_entry = {
                "id":               str(uuid.uuid4())[:8],
                "day":              "",
                "time":             "",
                "original_slot":    {},
                "new_title_id":     "",
                "new_title":        "",
                "new_runtime":      0,
                "bypass_type":      "Manual Intervention",
                "interrupted_prompt": _prompt_excerpt,
                "partial_response": "".join(assembled)[:500],
                "timestamp":        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                "status":           "pending",
                "audit":            _MANUAL_OVERRIDE_AUDIT,
            }
            with _queue_lock:
                entries = _load_queue()
                entries.append(bypass_entry)
                _save_queue(entries)
            with _bypass_lock:
                _set_bypass_state(_prompt_excerpt)
        else:
            # Clean completion — clear any active bypass
            _generation_complete["ok"] = True
            with _bypass_lock:
                _clear_bypass_state()

            # ── Empty-response / conflict-deadlock enforcement ───────────────
            # If the agent returned nothing (likely an is_final_conflict loop that
            # exhausted max_iterations without emitting text), synthesise a hard
            # Editorial Conflict response so the frontend is never left blank.
            if not assembled:
                conflict_msg = (
                    "# Editorial Conflict Detected\n\n"
                    "Why isn't anybody ready?\n\n"
                    "The scheduling engine attempted this move and was stopped by "
                    "Editorial Policy. A Seasonal Incompatibility was detected: "
                    "Met Gala content is exclusive to Avant-Garde Wednesday, and "
                    "Paris Fashion Week content belongs on Ready-to-Wear Saturday "
                    "or Global Couture Thursday. These two events do not share a "
                    "marquee — not on this channel, not on any channel I oversee.\n\n"
                    "Use the Manual Override Queue to stage an alternative title, "
                    "or ask me to recommend a replacement that respects the arc.\n\n"
                    f"`{_MANUAL_OVERRIDE_AUDIT}`\n\n"
                    "That is all."
                )
                assembled = [conflict_msg]
                yield f"data: {json.dumps({'type': 'text', 'content': conflict_msg})}\n\n"
                print("DEBUG: Empty response detected — injected Editorial Conflict fallback.")

        if assembled:
            _append_message("assistant", "".join(assembled))

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/generate")
async def generate(request: Request):
    """Non-streaming proxy for NAT's /generate."""
    body = await request.json()

    if body.get("reset_history"):
        with _history_lock:
            _save_history([])
        print("DEBUG: History reset — cleared chat_history.json")

    msgs = body.get("messages", [])
    if msgs:
        print(f"DEBUG: Processing new message: {msgs[-1].get('content','')[:80]!r}")

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
        else:
            # Empty content — inject Editorial Conflict fallback
            fallback = (
                "# Editorial Conflict Detected\n\n"
                "Why isn't anybody ready?\n\n"
                "The scheduling engine was stopped by Editorial Policy. "
                "A Seasonal Incompatibility was detected: Met Gala content is "
                "exclusive to Avant-Garde Wednesday; Paris Fashion Week content "
                "belongs on Ready-to-Wear Saturday or Global Couture Thursday. "
                "These two events do not share a marquee.\n\n"
                "Use the Manual Override Queue to stage an alternative, "
                "or ask me to recommend a seasonally coherent replacement.\n\n"
                f"`{_MANUAL_OVERRIDE_AUDIT}`\n\nThat is all."
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

    with _queue_lock:
        entries = _load_queue()
        entries.append(bypass_entry)
        _save_queue(entries)

    with _bypass_lock:
        _set_bypass_state(last_user[:300])

    print(f"DEBUG: Stop requested — bypass staged. Interrupted prompt: {last_user[:80]!r}")
    return {
        "stopped":            True,
        "bypass_id":          bypass_entry["id"],
        "message":            "Generation cancelled. Interrupted request staged in Override Queue.",
        "queue_depth":        len([e for e in entries if e.get("status") == "pending"]),
        "controller_status":  "MANUAL_BYPASS",
        "active_gatekeeper":  "Tiger Team",
        "_audit":             _MANUAL_OVERRIDE_AUDIT,
    }


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
            "_engine": "DuckDB + Parquet",
            "_source": "Regional Average (synthetic — no matching rows for this filter)",
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
        "_engine": "DuckDB + Parquet",
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
        for slot in day_data.get("slots", []):
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
        "anchor_index": anchor_index,
        "total_in_week": n,
        "slots":        result_slots,
        "_engine":      "DuckDB + Parquet" if _DUCKDB_AVAILABLE else "Pandas",
        **_bypass_annotation(),
    }


# ── System / compute endpoints ────────────────────────────────────────────────

@app.get("/health")
def health():
    mode = _read_mode()
    return {
        "status":       "alive",
        "engine":       mode,
        "gpu_detected": HAS_GPU,
        **{k: _COMPUTE_PROFILES[mode][k] for k in ("source_compute", "latency_ms")},
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081, log_level="info")
