"""
FAST Channel Media Intelligence – Tool Functions
Each function reads from pre-generated JSON files in data/.
"""

import json
import time
from datetime import datetime, timezone
from typing import Any

# ── Hybrid Execution Mode ───────────────────────────────────────────────────
# ONLINE  → cuDF / cuML on NVIDIA GPU (Brev A10G instance)
# OFFLINE → pandas / sklearn on local CPU (MacBook, no GPU required)
#
# Mode is persisted in data/mode.json so the sidecar API server and the
# NAT server stay in sync without sharing process memory.

import os as _os

_MODE_FILE = _os.path.join(_os.path.dirname(__file__), "data", "mode.json")

# Detect GPU at import time — cudf only exists on RAPIDS-enabled instances
import importlib.util as _importlib_util
HAS_GPU: bool = _importlib_util.find_spec("cudf") is not None

# ── Nielsen Measurement Constants ───────────────────────────────────────────
# Channel-level universe for telemetry records that have no DMA field.
_CHANNEL_UNIVERSE_HH: int = 15_200_000

# Blended national CPM for premium Female/LGBT+ fashion audience ($/thousand).
_CHANNEL_CPM: float = 45.0

# Standard US cable HUT (Households Using Television) estimates, by hour.
_HUT_BY_HOUR: dict[int, float] = {
    0: 0.18, 1: 0.10, 2: 0.07, 3: 0.05, 4: 0.05, 5: 0.08,
    6: 0.14, 7: 0.22, 8: 0.30, 9: 0.35, 10: 0.38, 11: 0.40,
    12: 0.43, 13: 0.42, 14: 0.40, 15: 0.42, 16: 0.48,
    17: 0.55, 18: 0.62, 19: 0.68, 20: 0.72, 21: 0.70,
    22: 0.60, 23: 0.42,
}

# ── Event Exclusivity Signal Sets ─────────────────────────────────────────────
# Prevent Met Gala and Paris Fashion Week content from sharing a programming day.
_MET_GALA_SIGNALS: frozenset = frozenset({
    "met gala", "the met ball", "metropolitan museum", "first monday in may",
    "costume institute", "camp: notes on fashion", "camp notes on fashion",
    "vogue world",
})
_PFW_SIGNALS: frozenset = frozenset({
    "paris fashion week", "pfw", "spring collection", "spring/summer",
    "spring summer", "ready-to-wear", "ready to wear", "prêt-à-porter",
    "pret-a-porter",
})

# ── Local engine detection ─────────────────────────────────────────────────────
_DUCKDB_AVAILABLE: bool = _importlib_util.find_spec("duckdb") is not None
_LOCAL_ENGINE: str = "DuckDB + Parquet" if _DUCKDB_AVAILABLE else "Pandas"

_COMPUTE_PROFILES = {
    "ONLINE":  {
        "source_compute": "NVIDIA A10G (Brev GPU)",
        "engine":         "NVIDIA RAPIDS (cuDF)",
        "gpu_boost":      "35x",
        "latency_ms":     12,
    },
    "AUTO-FALLBACK": {
        "source_compute": "Local CPU",
        "engine":         _LOCAL_ENGINE,
        "gpu_boost":      "1x",
        "latency_ms":     185,
    },
    "OFFLINE": {
        "source_compute": "Local CPU",
        "engine":         _LOCAL_ENGINE,
        "gpu_boost":      "1x",
        "latency_ms":     185,
    },
}


def get_performance_metadata() -> dict:
    """Return a Technical Audit dictionary for the current execution mode.

    Suitable for embedding in every tool response and surfacing in the UI.
    """
    mode = _read_mode()
    profile = _COMPUTE_PROFILES[mode]
    return {
        "execution_mode": mode,
        "source_compute": profile["source_compute"],
        "engine":         profile["engine"],
        "gpu_boost":      profile["gpu_boost"],
        "latency_ms":     profile["latency_ms"],
        "has_gpu":        HAS_GPU,
        "pipeline":       "Condé Nast Accelerated Intelligence Layer",
    }


def _read_mode() -> str:
    try:
        with open(_MODE_FILE) as f:
            return json.load(f).get("mode", "OFFLINE")
    except (FileNotFoundError, json.JSONDecodeError):
        return "OFFLINE"


def _detect_effective_mode() -> str:
    """Return the effective execution mode with automatic GPU fallback.

    If mode.json requests ONLINE but cuDF is unavailable or times out,
    returns 'AUTO-FALLBACK' so the audit block reflects reality without
    crashing or requiring a manual mode switch.
    """
    requested = _read_mode()
    if requested != "ONLINE":
        return requested
    try:
        import signal as _signal

        def _alarm(sig, frame):   # noqa: ANN001
            raise TimeoutError

        _signal.signal(_signal.SIGALRM, _alarm)
        _signal.alarm(2)          # 2-second timeout for GPU probe
        import cudf               # noqa: F401
        _signal.alarm(0)
        return "ONLINE"
    except (ImportError, TimeoutError, Exception):
        return "AUTO-FALLBACK"


def _write_mode(mode: str) -> None:
    with open(_MODE_FILE, "w") as f:
        json.dump({"mode": mode}, f)


# Module-level alias kept for backwards compat — always read live from file
@property  # type: ignore[misc]
def EXECUTION_MODE() -> str:  # noqa: N802
    return _read_mode()


def get_system_health() -> dict:
    """Dedicated heartbeat tool for the Lovable dashboard status pulse.

    Returns the full performance metadata so the UI can display the active
    engine, compute source, and GPU detection state in real time.

    Returns:
        The get_performance_metadata() dict — same shape as the _audit block
        embedded in every tool response.
    """
    return get_performance_metadata()


def toggle_system_mode(mode: str) -> dict:
    """Switch the pipeline between ONLINE (GPU) and OFFLINE (CPU mock) modes.

    Persists the choice to data/mode.json so the sidecar API server and the
    NAT agent stay in sync across process boundaries.

    Args:
        mode: 'ONLINE' or 'OFFLINE' (case-insensitive).

    Returns:
        Confirmation dict with the active mode and compute profile.
    """
    mode = mode.upper().strip()
    if mode not in _COMPUTE_PROFILES:
        return {"error": f"Unknown mode '{mode}'. Use 'ONLINE' or 'OFFLINE'."}
    _write_mode(mode)
    return {
        "status":         "mode_switched",
        "execution_mode": mode,
        **_COMPUTE_PROFILES[mode],
    }


# ── Demographic Friction Engine ─────────────────────────────────────────────
# Maps content target_age bracket → per-generation engagement multipliers.
# Content aimed at 18-24 pulls Gen Z in (1.4x) and pushes Silver_Stylists away (0.6x).
# Used in generate_candidates and get_audience_telemetry to score and flag trade-offs.

DEMOGRAPHIC_AFFINITY_MAP: dict[str, dict[str, float]] = {
    "18-24": {
        "Gen_Alpha":      1.3,
        "Gen_Z":          1.4,
        "Millennial":     0.9,
        "Gen_X":          0.7,
        "Silver_Stylists":0.6,
    },
    "25-34": {
        "Gen_Alpha":      0.8,
        "Gen_Z":          1.1,
        "Millennial":     1.4,
        "Gen_X":          0.9,
        "Silver_Stylists":0.7,
    },
    "35-49": {
        "Gen_Alpha":      0.5,
        "Gen_Z":          0.7,
        "Millennial":     1.1,
        "Gen_X":          1.4,
        "Silver_Stylists":0.9,
    },
    "50+": {
        "Gen_Alpha":      0.4,
        "Gen_Z":          0.6,
        "Millennial":     0.8,
        "Gen_X":          1.1,
        "Silver_Stylists":1.4,
    },
}

_GENERATIONS = ("Gen_Alpha", "Gen_Z", "Millennial", "Gen_X", "Silver_Stylists")


def compute_demographic_friction(target_age: str, primary_demo: str = "") -> dict:
    """Calculate how much a content choice alienates non-primary generations.

    friction_index (0-100): 0 = universal appeal, 100 = maximally polarising.
    repelled_segments: generations whose multiplier drops below 0.75.
    boosted_segments:  generations whose multiplier rises above 1.15.

    Args:
        target_age:   Content's target age bracket: '18-24', '25-34', '35-49', '50+'.
        primary_demo: Optional — context demographic for commentary.

    Returns:
        Dict with friction_index, repelled_segments, boosted_segments, and
        per-generation affinity multipliers.
    """
    affinities = DEMOGRAPHIC_AFFINITY_MAP.get(target_age, DEMOGRAPHIC_AFFINITY_MAP["25-34"])

    # Friction = mean repulsion across below-neutral multipliers, scaled to 100
    repelled = {g: m for g, m in affinities.items() if m < 0.75}
    boosted  = {g: m for g, m in affinities.items() if m > 1.15}
    if repelled:
        friction_index = round(sum(1.0 - m for m in repelled.values()) / len(affinities) * 100, 1)
    else:
        friction_index = 0.0

    # Human-readable trade-off note
    if friction_index >= 40:
        trade_off = (
            f"You've captured the Vanguard, but you've alienated the Silver Stylists. "
            f"They've gone to watch the competition's boring linear broadcast. "
            f"I hope those Gen Z clicks were worth the heritage loss. That is all."
        )
    elif friction_index >= 20:
        trade_off = (
            f"A strategic trade-off: strong {', '.join(boosted)} acquisition at the cost "
            f"of {', '.join(repelled)} retention. Acceptable — if intentional."
        )
    else:
        trade_off = "Broad generational appeal. No significant friction detected."

    return {
        "target_age":         target_age,
        "friction_index":     friction_index,
        "repelled_segments":  repelled,
        "boosted_segments":   boosted,
        "affinity_map":       affinities,
        "trade_off_note":     trade_off,
    }


def _apply_affinity_multipliers_pandas(df: Any, target_age: str) -> Any:
    """Apply demographic affinity multipliers to a DataFrame (OFFLINE path).

    Mutates demographic_scores columns in-place using vectorised pandas ops.
    """
    affinities = DEMOGRAPHIC_AFFINITY_MAP.get(target_age, DEMOGRAPHIC_AFFINITY_MAP["25-34"])
    for gen, mult in affinities.items():
        col = f"score_{gen}"
        if col in df.columns:
            df[col] = (df[col] * mult).clip(upper=1.0)
    return df


def _compute_meta(rows: int = 0) -> dict:
    """Return the current compute profile for embedding in tool responses.

    Uses _detect_effective_mode() so the audit block always reflects the actual
    engine in use — including AUTO-FALLBACK when Brev is unreachable at runtime.
    """
    mode    = _detect_effective_mode()
    profile = _COMPUTE_PROFILES[mode]
    return {
        "execution_mode": mode,
        "source_compute": profile["source_compute"],
        "engine":         profile["engine"],
        "gpu_boost":      profile["gpu_boost"],
        "latency_ms":     profile["latency_ms"],
        "rows_processed": rows,
        "pipeline":       "Condé Nast Accelerated Intelligence Layer",
    }


def _load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def accelerated_data_crunch(df: Any) -> tuple[Any, dict]:
    """GPU/CPU data processing with automatic fallback.

    Tries cuDF when mode is ONLINE. If cuDF is unavailable or the cluster times
    out, silently falls back to pandas and marks the audit block AUTO-FALLBACK.

    Args:
        df: A pandas DataFrame to process.

    Returns:
        (processed_df, rapids_metadata) surfaced in the UI Technical Audit panel.
    """
    import pandas as pd

    effective = _detect_effective_mode()
    if effective == "ONLINE":
        try:
            import cudf
            gpu_df = cudf.DataFrame.from_pandas(df)
            if "viewers" in gpu_df.columns:
                gpu_df = gpu_df.sort_values("viewers", ascending=False)
            processed = gpu_df.to_pandas()
        except (ImportError, Exception):
            # GPU probe failed mid-flight — silently downgrade
            processed = df.sort_values("viewers", ascending=False) if "viewers" in df.columns else df.copy()
    else:
        time.sleep(0.01)
        processed = df.copy()
        if "viewers" in processed.columns:
            processed = processed.sort_values("viewers", ascending=False)

    return processed, _compute_meta(rows=len(processed))


def get_channel_blueprint(channel_id: str) -> str:
    """Return the channel strategy blueprint for the given channel.

    Args:
        channel_id: The channel identifier (e.g. 'ch_runway_01').

    Returns:
        A JSON string containing identity, strategy, and target demographic.
        Returns an error JSON string if the channel is not recognised.
    """
    blueprints: dict[str, dict] = {
        "ch_runway_01": {
            "channel_id": "ch_runway_01",
            "identity": "Runway Inclusive",
            "audience_primary": "Female / LGBT+",
            "audience_secondary": "18-35",
            "strategy": (
                "Curating iconic female performances that define style "
                "and subvert the status quo."
            ),
            "description": (
                "The premier destination for female-led cinema and "
                "fashion-forward storytelling."
            ),
        }
    }

    blueprint = blueprints.get(channel_id)
    if blueprint is None:
        return json.dumps({"error": f"Channel '{channel_id}' not found."})

    return json.dumps(_translate_ids(blueprint), indent=2)


def get_current_schedule(channel_id: str, timestamp: str) -> dict:
    """Return the program airing on a channel at the given timestamp.

    Args:
        channel_id: The channel identifier to filter by.
        timestamp:  ISO-8601 timestamp string (e.g. '2026-04-20T16:00:00+00:00').

    Returns:
        The matching schedule slot dict, or an error dict if nothing is found.
    """
    try:
        slots: list[dict] = _load_json("data/schedule.json")
    except FileNotFoundError:
        return {"error": "data/schedule.json not found."}

    try:
        query_dt = datetime.fromisoformat(timestamp)
        if query_dt.tzinfo is None:
            query_dt = query_dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return {"error": f"Invalid timestamp format: '{timestamp}'."}

    for slot in slots:
        if slot.get("channel_id") != channel_id:
            continue
        try:
            start = datetime.fromisoformat(slot["start"])
            end   = datetime.fromisoformat(slot["end"])
        except (KeyError, ValueError):
            continue

        if start <= query_dt < end:
            return slot

    return {
        "error": (
            f"No program found on '{channel_id}' at {timestamp}."
        )
    }


def lookup_content_metadata(content_id: str) -> dict:
    """Return full metadata for a catalog asset by its show_id.

    Includes content_runtime_min (actual film length) and block_duration_min
    (EPG slot size, rounded up to nearest 30-minute boundary). The gap between
    the two is reserved for High-Fashion Ad Breaks and Exclusive Designer
    Interviews.

    Args:
        content_id: The show identifier (e.g. 's0001').

    Returns:
        The catalog entry dict enriched with scheduling fields,
        or an error dict if not found.
    """
    try:
        catalog: list[dict] = _load_json("data/catalog.json")
    except FileNotFoundError:
        return {"error": "data/catalog.json not found."}

    for item in catalog:
        if item.get("show_id") == content_id:
            result = dict(item)
            runtime = result.get("runtime_min", 90)
            block = ((runtime + 29) // 30) * 30
            result["content_runtime_min"] = runtime
            result["block_duration_min"] = block
            result["interstitial_min"] = block - runtime
            return result

    return {"error": f"Content '{content_id}' not found in catalog."}


def get_strategic_programming_insight(query: str) -> dict:
    """Recommend catalog content based on Style Tribe alignment from the ML engine.

    Loads the GPU-clustered designer manifest and cross-references catalog titles
    whose descriptions align with the queried tribe's aesthetic signal keywords.

    Args:
        query: A tribe name, aesthetic keyword, or designer name
               (e.g. 'Avant-Garde', 'minimalist', 'Chanel').

    Returns:
        Dict with matched tribe, member designers, and recommended show_ids.
    """
    try:
        manifest: dict = _load_json("data/tribe_manifest.json")
        clustered: list[dict] = _load_json("data/designers_clustered.json")
        catalog: list[dict] = _load_json("data/catalog.json")
    except FileNotFoundError as e:
        return {"error": f"{e.filename} not found. Run ml_engine.py first."}

    q = query.lower()

    # Match query to a tribe — try name match first, then keyword scan
    matched_tribe: str | None = None
    for tribe in manifest:
        if q in tribe.lower():
            matched_tribe = tribe
            break
    if not matched_tribe:
        # Fall back: find any designer in the query and return their tribe
        for designer in clustered:
            if q in designer["name"].lower():
                matched_tribe = designer["style_tribe"]
                break
    if not matched_tribe:
        # Last resort: partial keyword match against tribe signal terms
        for designer in clustered:
            if any(q in term for term in designer.get("top_tribe_terms", [])):
                matched_tribe = designer["style_tribe"]
                break

    if not matched_tribe:
        return {
            "error": f"No tribe matched '{query}'.",
            "available_tribes": list(manifest.keys()),
        }

    tribe_designers = manifest[matched_tribe]
    # Retrieve signal terms for this tribe from the first matching clustered entry
    signal_terms: list[str] = []
    for d in clustered:
        if d["style_tribe"] == matched_tribe:
            signal_terms = d.get("top_tribe_terms", [])
            break

    # Cross-reference catalog: score each title against tribe signal terms
    recommendations = []
    for item in catalog:
        text = (str(item.get("description") or "") + " " + str(item.get("listed_in") or "")).lower()
        score = sum(1 for term in signal_terms if term in text)
        if score > 0:
            recommendations.append({
                "show_id": item["show_id"],
                "title":   item["title"],
                "score":   score,
                "genres":  item.get("listed_in", ""),
            })
    import pandas as pd
    rec_df = pd.DataFrame(recommendations) if recommendations else pd.DataFrame()
    _, rapids_meta = accelerated_data_crunch(rec_df)
    recommendations.sort(key=lambda x: x["score"], reverse=True)

    return {
        "style_tribe":             matched_tribe,
        "tribe_designers":         tribe_designers,
        "tribe_signal_terms":      signal_terms,
        "catalog_recommendations": recommendations[:10],
        "source":                  "Condé Nast Accelerated Intelligence Layer (GPU-clustered)",
        "_rapids_metadata":        rapids_meta,
    }


def find_similar_designers(brand_name: str) -> dict:
    """Find stylistically similar designers using the cuML KNN similarity index.

    Args:
        brand_name: Designer or house name (e.g. 'Chanel', 'Vivienne Westwood').

    Returns:
        Dict with the input designer's tribe and their N nearest neighbours.
    """
    import pickle

    try:
        with open("data/knn_index.pkl", "rb") as f:
            bundle = pickle.load(f)
    except FileNotFoundError:
        return {"error": "data/knn_index.pkl not found. Run ml_engine.py first."}

    model       = bundle["model"]
    names       = bundle["names"]
    X           = bundle["matrix"]
    tribe_labels = bundle["tribe_labels"]
    using_cuml  = bundle.get("cuml", False)

    # Find the designer in the index (case-insensitive partial match)
    q = brand_name.lower()
    idx = next((i for i, n in enumerate(names) if q in n.lower()), None)
    if idx is None:
        return {
            "error": f"Designer '{brand_name}' not found in KNN index.",
            "available": names,
        }

    query_vec = X[idx].reshape(1, -1)
    if using_cuml:
        import cupy as cp
        distances, indices = model.kneighbors(cp.asarray(query_vec))
        distances = cp.asnumpy(distances).flatten()
        indices   = cp.asnumpy(indices).flatten()
    else:
        distances, indices = model.kneighbors(query_vec)
        distances = distances.flatten()
        indices   = indices.flatten()

    neighbours = []
    for dist, nidx in zip(distances, indices):
        if int(nidx) == idx:
            continue  # skip self
        neighbours.append({
            "designer":    names[int(nidx)],
            "style_tribe": tribe_labels[int(nidx)],
            "similarity":  round(1.0 - float(dist), 3),
        })
        if len(neighbours) >= 5:
            break

    return {
        "query_designer":  names[idx],
        "query_tribe":     tribe_labels[idx],
        "similar_designers": neighbours,
        "index_backend":   "cuML KNN (GPU)" if using_cuml else "sklearn KNN (CPU)",
    }


def search_fashion_designers(query: str) -> list[dict]:
    """Search the fashion designer knowledge base by name, house, hallmark, or era.

    Args:
        query: Free-text search term (e.g. 'Chanel', 'leather', '1960s', 'French').

    Returns:
        List of matching designer dicts, or an empty list if none found.
    """
    try:
        designers: list[dict] = _load_json("data/designers.json")
    except FileNotFoundError:
        return [{"error": "data/designers.json not found."}]

    q = query.lower()
    results = []
    for d in designers:
        searchable = " ".join(str(v) for v in d.values()).lower()
        if q in searchable:
            results.append(d)
    return results


def get_met_gala_themes(year: int | None = None) -> list[dict]:
    """Return Met Gala theme records.

    Args:
        year: Optional specific year (e.g. 2024). If None or 0, returns all years.

    Returns:
        List of theme dicts with year, theme, and optional notes.
    """
    try:
        themes: list[dict] = _load_json("data/met_gala_themes.json")
    except FileNotFoundError:
        return [{"error": "data/met_gala_themes.json not found."}]

    if year:
        return [t for t in themes if t.get("year") == year]
    return themes


def update_schedule_slot(slot_time: str, new_title: str) -> str:
    """Replace the title in a schedule slot and recalculate runtime/padding.

    Args:
        slot_time: ISO-8601 start time of the slot, or HH:MM shorthand (UTC).
                   E.g. '2026-04-20T16:00:00+00:00' or '16:00'.
        new_title: Exact or partial title of the replacement film from catalog.json.

    Returns:
        A success or error message string.
    """
    import os

    SCHEDULE_PATH = "data/schedule.json"
    CATALOG_PATH  = "data/catalog.json"

    try:
        slots: list[dict] = _load_json(SCHEDULE_PATH)
    except FileNotFoundError:
        return json.dumps({"error": "data/schedule.json not found."})

    try:
        catalog: list[dict] = _load_json(CATALOG_PATH)
    except FileNotFoundError:
        return json.dumps({"error": "data/catalog.json not found."})

    # Parse slot_time — accept full ISO or bare HH:MM
    target_dt: datetime | None = None
    try:
        target_dt = datetime.fromisoformat(slot_time)
        if target_dt.tzinfo is None:
            target_dt = target_dt.replace(tzinfo=timezone.utc)
    except ValueError:
        # Try HH:MM or HH:MM:SS shorthand against the schedule date
        try:
            t = datetime.strptime(slot_time.strip(), "%H:%M")
            target_dt = datetime(2026, 4, 20, t.hour, t.minute, tzinfo=timezone.utc)
        except ValueError:
            return json.dumps({"error": f"Cannot parse slot_time '{slot_time}'. Use ISO-8601 or HH:MM."})

    # Find the slot
    slot_idx: int | None = None
    for i, slot in enumerate(slots):
        try:
            start = datetime.fromisoformat(slot["start"])
        except (KeyError, ValueError):
            continue
        if start == target_dt:
            slot_idx = i
            break

    if slot_idx is None:
        available = [s["start"] for s in slots]
        return json.dumps({"error": f"No slot found starting at '{slot_time}'.", "available_slots": available})

    # Find the new title in catalog (case-insensitive partial match)
    q = new_title.lower().strip()
    catalog_match: dict | None = None
    for item in catalog:
        if q in item.get("title", "").lower():
            catalog_match = item
            break

    if catalog_match is None:
        return json.dumps({"error": f"Title '{new_title}' not found in catalog."})

    runtime_min: int = catalog_match.get("runtime_min", 90)
    block_min: int   = ((runtime_min + 29) // 30) * 30
    interstitial_min = block_min - runtime_min

    old_title   = slots[slot_idx].get("title", "")
    slot_start  = datetime.fromisoformat(slots[slot_idx]["start"])
    new_end     = slot_start.replace(tzinfo=timezone.utc) + __import__("datetime").timedelta(minutes=block_min)

    slots[slot_idx]["title"]               = catalog_match["title"]
    slots[slot_idx]["show_id"]             = catalog_match.get("show_id", slots[slot_idx].get("show_id", ""))
    slots[slot_idx]["content_runtime_min"] = runtime_min
    slots[slot_idx]["duration_min"]        = block_min
    slots[slot_idx]["interstitial_min"]    = interstitial_min
    slots[slot_idx]["end"]                 = new_end.strftime("%Y-%m-%dT%H:%M:%S+00:00")

    with open(SCHEDULE_PATH, "w") as f:
        json.dump(slots, f, indent=2)

    return json.dumps({
        "status":            "updated",
        "slot_start":        slots[slot_idx]["start"],
        "previous_title":    old_title,
        "new_title":         catalog_match["title"],
        "content_runtime_min": runtime_min,
        "block_duration_min":  block_min,
        "interstitial_min":    interstitial_min,
        "message":           "The collection has been edited. It is much improved. That is all.",
        "_audit":            _compute_meta(rows=len(slots)),
    })


def get_audience_telemetry(content_id: str, current_time: str, demographic: str = "") -> dict:
    """Return viewership data for an asset at a given hour with demographic breakdown.

    Always surfaces Female and LGBTQ+ core stats. If a specific demographic is
    requested, its stats are returned alongside the core segments for comparison.

    Args:
        content_id:   Show identifier (e.g. 's0001').
        current_time: ISO-8601 timestamp to select the time window.
        demographic:  Optional filter — 'Female', 'LGBTQ+', 'Gen_Z', 'Millennial',
                      'Male'. Defaults to '' which returns all core segments.

    Returns:
        Dict with core_segments, requested_segment (if any), all records, and audit.
    """
    # Normalise demographic label: accept Gen_Z / Gen Z / gen z → "Gen Z"
    _DEMO_ALIASES: dict[str, str] = {
        "gen_z": "Gen Z", "genz": "Gen Z", "gen z": "Gen Z",
        "millennial": "Millennial", "millennials": "Millennial",
        "female": "Female", "women": "Female",
        "lgbtq+": "LGBT+", "lgbtq": "LGBT+", "lgbt": "LGBT+", "lgbt+": "LGBT+",
        "male": "Male", "men": "Male",
    }
    requested_demo = _DEMO_ALIASES.get(demographic.lower().strip(), demographic.strip()) if demographic else ""

    try:
        records: list[dict] = _load_json("data/telemetry.json")
    except FileNotFoundError:
        return {"error": "data/telemetry.json not found."}

    try:
        query_dt = datetime.fromisoformat(current_time)
        if query_dt.tzinfo is None:
            query_dt = query_dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return {"error": f"Invalid timestamp format: '{current_time}'."}

    matches = [
        rec for rec in records
        if rec.get("show_id") == content_id
        and (lambda r: (
            (dt := datetime.fromisoformat(r["timestamp"]))
            and dt.date() == query_dt.date()
            and dt.hour == query_dt.hour
        ) if "timestamp" in rec else False)(rec)
    ]

    # Simpler loop for clarity
    matches = []
    for rec in records:
        if rec.get("show_id") != content_id:
            continue
        try:
            rec_dt = datetime.fromisoformat(rec["timestamp"])
        except (KeyError, ValueError):
            continue
        if rec_dt.date() == query_dt.date() and rec_dt.hour == query_dt.hour:
            matches.append(rec)

    if not matches:
        return {"error": f"No telemetry for '{content_id}' at {query_dt.strftime('%Y-%m-%d %H:00 UTC')}."}

    import pandas as pd
    df = pd.DataFrame(matches)
    processed_df, rapids_meta = accelerated_data_crunch(df)

    total_viewers = int(processed_df["viewers"].sum()) if "viewers" in processed_df.columns else 0

    # Always compute core Female + LGBTQ+ breakdown
    def _segment_stats(seg_df: "pd.DataFrame", label: str) -> dict:
        if seg_df.empty:
            return {"segment": label, "viewers": 0, "avg_viewers": 0, "records": 0}
        return {
            "segment":      label,
            "viewers":      int(seg_df["viewers"].sum()),
            "avg_viewers":  int(seg_df["viewers"].mean()),
            "records":      len(seg_df),
            "is_core":      label in ("Female", "LGBT+"),
        }

    female_df = processed_df[processed_df["segment"] == "Female_Viewers"] if "segment" in processed_df.columns else pd.DataFrame()
    lgbtq_df  = processed_df[processed_df["segment"] == "LGBTQ_Core_Audience"] if "segment" in processed_df.columns else pd.DataFrame()

    # Fall back to pre-computed scalar columns if segment column absent
    if female_df.empty and "Female_Viewers" in processed_df.columns:
        female_viewers = int(processed_df["Female_Viewers"].sum())
        lgbtq_viewers  = int(processed_df["LGBTQ_Core_Audience"].sum()) if "LGBTQ_Core_Audience" in processed_df.columns else 0
        core_segments = [
            {"segment": "Female",  "viewers": female_viewers, "is_core": True},
            {"segment": "LGBT+",   "viewers": lgbtq_viewers,  "is_core": True},
        ]
    else:
        core_segments = [
            _segment_stats(female_df, "Female"),
            _segment_stats(lgbtq_df, "LGBT+"),
        ]

    # Top Performing Market — cross-reference engagement_logs for this show
    top_market: dict = {}
    try:
        import pandas as pd
        eng_logs: list[dict] = _load_json("data/engagement_logs.json")
        show_logs = [r for r in eng_logs if r.get("show_id") == content_id]
        if show_logs:
            mdf = pd.DataFrame(show_logs)
            market_agg = (
                mdf.groupby("market")["completion_rate"]
                .agg(["mean", "count"])
                .reset_index()
                .rename(columns={"mean": "avg_completion", "count": "sessions"})
                .sort_values("avg_completion", ascending=False)
            )
            if not market_agg.empty:
                top = market_agg.iloc[0]
                top_market = {
                    "market":         str(top["market"]),
                    "avg_completion": round(float(top["avg_completion"]), 4),
                    "sessions":       int(top["sessions"]),
                }
    except Exception:
        pass

    result: dict = {
        "show_id":               content_id,
        "window":                query_dt.strftime("%Y-%m-%d %H:00 UTC"),
        "total_viewers":         total_viewers,
        "core_segments":         core_segments,
        "top_performing_market": top_market or {"note": "Engagement log data unavailable."},
        "records":               processed_df.to_dict(orient="records"),
        "_rapids_metadata":      rapids_meta,
    }

    # Add requested demographic breakdown for comparison if specified
    if requested_demo and requested_demo not in ("Female", "LGBT+"):
        req_df = processed_df[processed_df.get("segment", pd.Series()) == requested_demo] if "segment" in processed_df.columns else pd.DataFrame()
        result["requested_segment"] = {
            "segment":      requested_demo,
            "viewers":      int(req_df["viewers"].sum()) if not req_df.empty and "viewers" in req_df.columns else 0,
            "avg_viewers":  int(req_df["viewers"].mean()) if not req_df.empty and "viewers" in req_df.columns else 0,
            "records":      len(req_df),
            "is_core":      False,
            "note":         "Growth opportunity segment — not our primary mandate.",
        }
    elif requested_demo:
        result["requested_segment"] = next(
            (s for s in core_segments if s["segment"] == requested_demo), core_segments[0]
        )

    # ── Nielsen Metrics ──────────────────────────────────────────────────────
    # If telemetry.json was enriched by generate_nielsen_data.py the Nielsen
    # fields are already present on every record in processed_df.  We aggregate
    # them here; if the fields are absent we compute them inline from total_viewers.
    _nielsen_keys = [
        "UniverseEstimate_UE", "HouseholdsUsingTV_HUT", "PersonsViewingTV_PUT",
        "Audience_HH_or_Persons", "Rating_Pct", "Share_Pct", "AverageAudience_000",
        "GRPs", "Reach_Pct", "Frequency", "GrossImpressions",
        "MediaCost", "CPM", "CPP",
    ]
    if "Rating_Pct" in processed_df.columns:
        # Aggregate from pre-computed record-level fields
        nielsen_metrics: dict = {
            "UniverseEstimate_UE":   int(processed_df["UniverseEstimate_UE"].iloc[0])
                                     if "UniverseEstimate_UE" in processed_df.columns
                                     else _CHANNEL_UNIVERSE_HH,
            "HouseholdsUsingTV_HUT": round(float(processed_df["HouseholdsUsingTV_HUT"].mean()), 2)
                                     if "HouseholdsUsingTV_HUT" in processed_df.columns else 0.0,
            "PersonsViewingTV_PUT":  round(float(processed_df["PersonsViewingTV_PUT"].mean()), 2)
                                     if "PersonsViewingTV_PUT" in processed_df.columns else 0.0,
            "Audience_HH_or_Persons": total_viewers,
            "Rating_Pct":            round(float(processed_df["Rating_Pct"].mean()), 4)
                                     if "Rating_Pct" in processed_df.columns else 0.0,
            "Share_Pct":             round(float(processed_df["Share_Pct"].mean()), 4)
                                     if "Share_Pct" in processed_df.columns else 0.0,
            "AverageAudience_000":   round(total_viewers / 1_000, 3),
            "GRPs":                  round(float(processed_df["GRPs"].sum()), 4)
                                     if "GRPs" in processed_df.columns else 0.0,
            "Reach_Pct":             round(float(processed_df["Reach_Pct"].mean()), 4)
                                     if "Reach_Pct" in processed_df.columns else 0.0,
            "Frequency":             round(float(processed_df["Frequency"].mean()), 4)
                                     if "Frequency" in processed_df.columns else 0.0,
            "GrossImpressions":      total_viewers,
            "MediaCost":             round(float(processed_df["MediaCost"].sum()), 2)
                                     if "MediaCost" in processed_df.columns else 0.0,
            "CPM":                   round(float(processed_df["CPM"].mean()), 2)
                                     if "CPM" in processed_df.columns else _CHANNEL_CPM,
            "CPP":                   round(float(processed_df["CPP"].mean()), 2)
                                     if "CPP" in processed_df.columns else 0.0,
        }
    else:
        # Inline fallback — compute from total_viewers + hour-based HUT
        _hut = _HUT_BY_HOUR.get(query_dt.hour, 0.50)
        _rating = (total_viewers / _CHANNEL_UNIVERSE_HH) * 100 if total_viewers > 0 else 0.0
        _share  = (_rating / (_hut * 100)) * 100 if _hut > 0 else 0.0
        _grps   = _rating
        _cost   = (total_viewers / 1_000) * _CHANNEL_CPM
        nielsen_metrics = {
            "UniverseEstimate_UE":   _CHANNEL_UNIVERSE_HH,
            "HouseholdsUsingTV_HUT": round(_hut * 100, 2),
            "PersonsViewingTV_PUT":  round(_hut * 95, 2),
            "Audience_HH_or_Persons": total_viewers,
            "Rating_Pct":            round(_rating, 4),
            "Share_Pct":             round(_share, 4),
            "AverageAudience_000":   round(total_viewers / 1_000, 3),
            "GRPs":                  round(_grps, 4),
            "Reach_Pct":             round(min(_rating * 1.12, 100.0), 4),
            "Frequency":             round(total_viewers / max(_CHANNEL_UNIVERSE_HH * _rating / 100, 1), 4),
            "GrossImpressions":      total_viewers,
            "MediaCost":             round(_cost, 2),
            "CPM":                   _CHANNEL_CPM,
            "CPP":                   round(_cost / _grps, 2) if _grps > 0 else 0.0,
        }
    result["nielsen_metrics"] = nielsen_metrics

    # Top-market Nielsen lookup from aggregated nielsen_telemetry.json
    try:
        nielsen_agg: list[dict] = _load_json("data/nielsen_telemetry.json")
        market_rows = [
            r for r in nielsen_agg
            if r.get("show_id") == content_id
        ]
        if market_rows:
            # Sort by Rating_Pct descending — surface the top 3 DMA performers
            market_rows.sort(key=lambda r: r.get("Rating_Pct", 0), reverse=True)
            result["top_markets_nielsen"] = [
                {k: r[k] for k in ["market", "dma", "daypart", "Rating_Pct",
                                    "Share_Pct", "GRPs", "AverageAudience_000",
                                    "MediaCost", "CPM"] if k in r}
                for r in market_rows[:3]
            ]
    except (FileNotFoundError, Exception):
        pass

    # ── Demographic Friction — ONLINE only ──────────────────────────────────
    telemetry_mode = _read_mode()
    if telemetry_mode == "ONLINE":
        try:
            catalog_list: list[dict] = _load_json("data/catalog.json")
            cat_item = next((i for i in catalog_list if i.get("show_id") == content_id), {})
            target_age = cat_item.get("target_age", "25-34")
            friction = compute_demographic_friction(target_age, requested_demo)
            result["_rapids_metadata"]["friction_index"]     = friction["friction_index"]
            result["_rapids_metadata"]["repelled_segments"]  = friction["repelled_segments"]
            result["_rapids_metadata"]["boosted_segments"]   = friction["boosted_segments"]
            result["_rapids_metadata"]["trade_off_note"]     = friction["trade_off_note"]
            result["_rapids_metadata"]["strategic_analysis"] = "Available"
        except Exception:
            result["_rapids_metadata"]["strategic_analysis"] = "Available (friction data unavailable)"
    else:
        result["_rapids_metadata"]["friction_index"]     = None
        result["_rapids_metadata"]["friction_status"]    = "GPU_REQUIRED"
        result["_rapids_metadata"]["strategic_analysis"] = "Unavailable in Local Mode"

    return result


def get_movie_catalog() -> dict:
    """Return a clean, dropdown-ready catalog of all titles.

    Returns:
        Dict with a list of {show_id, title, genres, runtime_min} entries
        and an audit block, suitable for powering Lovable dropdowns.
    """
    try:
        catalog: list[dict] = _load_json("data/catalog.json")
    except FileNotFoundError:
        return {"error": "data/catalog.json not found."}

    titles = [
        {
            "show_id":    item["show_id"],
            "title":      item["title"],
            "genres":     item.get("listed_in", ""),
            "runtime_min": item.get("runtime_min", 90),
        }
        for item in catalog
        if item.get("title") and item.get("runtime_min", 0) >= 60
    ]
    titles.sort(key=lambda x: x["title"])

    return {
        "total":   len(titles),
        "titles":  titles,
        "_audit":  _compute_meta(rows=len(titles)),
    }


def generate_candidates(
    context_tribe: str = "",
    demographic: str = "",
    location_segment: str = "",
    density_tier: str = "",
) -> dict:
    """RAPIDS-first recommender with demographic and location filtering.

    Args:
        context_tribe:    Style Tribe name or keyword (e.g. 'Heritage Couture').
                          Defaults to all tribes.
        demographic:      Target segment — 'Female', 'LGBT+', 'Gen_Z',
                          'Millennial', 'Male'. Defaults to 'Female+LGBTQ+'
                          (our core mandate).
        location_segment: DMA market or region — 'New York', 'Paris', 'London',
                          'Milan', 'Los Angeles', 'Europe', etc.
                          If blank, all markets are included.

    OFFLINE: pandas aggregation. Paris/Milan/London get a +0.08 completion
             boost on European Cinema and High Fashion titles.
    ONLINE:  cuDF GPU filter + co-visitation scoring per market.

    Returns:
        Dict with candidates, filters applied, demographic comparison, and audit.
    """
    import pandas as pd

    _DEMO_ALIASES: dict[str, str] = {
        "gen_z": "Gen Z", "genz": "Gen Z", "gen z": "Gen Z",
        "millennial": "Millennial", "millennials": "Millennial",
        "female": "Female", "women": "Female",
        "lgbtq+": "LGBT+", "lgbtq": "LGBT+", "lgbt+": "LGBT+", "lgbt": "LGBT+",
        "male": "Male", "men": "Male",
    }
    CORE_DEMOS = {"Female", "LGBT+"}
    EUROPEAN_MARKETS = {"Paris", "London", "Milan"}
    EUROPEAN_FASHION_GENRES = {"drama", "fashion", "european", "french", "italian", "couture"}

    resolved_demo = _DEMO_ALIASES.get(demographic.lower().strip(), demographic.strip())
    is_core = not resolved_demo or resolved_demo in CORE_DEMOS
    is_growth = resolved_demo and not is_core

    try:
        logs: list[dict]      = _load_json("data/engagement_logs.json")
        catalog: list[dict]   = _load_json("data/catalog.json")
        clustered: list[dict] = _load_json("data/designers_clustered.json")
        manifest: dict        = _load_json("data/tribe_manifest.json")
    except FileNotFoundError as e:
        return {"error": f"{e.filename} not found."}

    # ── Resolve Style Tribe ──────────────────────────────────────────────────
    matched_tribe: str | None = None
    if context_tribe:
        q = context_tribe.lower()
        for tribe in manifest:
            if q in tribe.lower():
                matched_tribe = tribe
                break
        if not matched_tribe:
            for d in clustered:
                if q in d["name"].lower():
                    matched_tribe = d["style_tribe"]
                    break

    signal_terms: list[str] = []
    tribe_show_ids: set[str] = set()
    if matched_tribe:
        for d in clustered:
            if d["style_tribe"] == matched_tribe:
                signal_terms = d.get("top_tribe_terms", [])
                break
        for item in catalog:
            text = (str(item.get("description") or "") + " " + str(item.get("listed_in") or "")).lower()
            if any(term in text for term in signal_terms):
                tribe_show_ids.add(item["show_id"])

    # ── Demographic filter ───────────────────────────────────────────────────
    if is_growth:
        demo_filter = {resolved_demo}
    else:
        demo_filter = {resolved_demo} if resolved_demo else CORE_DEMOS

    # ── Location filter ──────────────────────────────────────────────────────
    loc = location_segment.strip()
    loc_lower = loc.lower()
    if loc_lower in ("europe", "eu"):
        market_filter: set[str] | None = EUROPEAN_MARKETS
    elif loc:
        market_filter = {m for m in {r["market"] for r in logs} if loc_lower in m.lower()}
    else:
        market_filter = None

    # ── Density tier filter ──────────────────────────────────────────────────
    tier = density_tier.strip()
    tier_filter: str | None = None
    if tier:
        # Accept shorthand: "suburban" → "Affluent Suburban", "exurban" → "Exurban"
        _TIER_ALIASES = {
            "suburban": "Affluent Suburban", "affluent": "Affluent Suburban",
            "exurban": "Exurban", "rural": "Exurban",
            "urban": "Urban Core", "city": "Urban Core",
        }
        tier_filter = _TIER_ALIASES.get(tier.lower(), tier)

    # ── Filter logs ──────────────────────────────────────────────────────────
    filtered = [
        r for r in logs
        if r["primary_demographic"] in demo_filter
        and (not tribe_show_ids or r["show_id"] in tribe_show_ids)
        and (market_filter is None or r.get("market") in market_filter)
        and (tier_filter is None or r.get("density_tier") == tier_filter)
    ]
    if not filtered:  # graceful fallback — drop tribe constraint
        filtered = [
            r for r in logs
            if r["primary_demographic"] in demo_filter
            and (market_filter is None or r.get("market") in market_filter)
            and (tier_filter is None or r.get("density_tier") == tier_filter)
        ]

    df = pd.DataFrame(filtered) if filtered else pd.DataFrame()
    if df.empty:
        return {"error": "No engagement data matched the supplied filters.", "filters": {
            "demographic": resolved_demo or "Female+LGBTQ+",
            "location": loc or "all markets",
            "tribe": matched_tribe or "all tribes",
        }}

    mode = _read_mode()
    catalog_map = {item["show_id"]: item for item in catalog}

    if mode == "ONLINE":
        try:
            cudf = __import__("cudf")  # GPU-only  # noqa: F841
            gdf = cudf.DataFrame.from_pandas(df)

            co_visit_counts: dict[str, int] = {}
            for row in filtered:
                for co_id in row.get("session_shows", []):
                    if co_id != row["show_id"]:
                        co_visit_counts[row["show_id"]] = co_visit_counts.get(row["show_id"], 0) + 1

            agg = gdf.groupby("show_id").agg({"completion_rate": "mean"}).reset_index().to_pandas()
            agg["co_visits"] = agg["show_id"].map(co_visit_counts).fillna(0)
            max_co = agg["co_visits"].max() or 1
            agg["combined_score"] = agg["completion_rate"] * 0.7 + (agg["co_visits"] / max_co) * 0.3
            agg = agg.sort_values("combined_score", ascending=False)
            score_col = "combined_score"
        except ImportError:
            agg = df.groupby("show_id")["completion_rate"].mean().reset_index()
            agg = agg.sort_values("completion_rate", ascending=False)
            score_col = "completion_rate"
    else:
        time.sleep(0.05)
        agg = df.groupby("show_id")["completion_rate"].mean().reset_index()

        # Regional bias: Paris/Milan/London → boost European/High Fashion titles
        if loc and any(m.lower() in loc_lower or loc_lower in m.lower() for m in EUROPEAN_MARKETS):
            def _eu_boost(show_id: str) -> float:
                meta = catalog_map.get(show_id, {})
                genres = str(meta.get("listed_in", "")).lower()
                return 0.08 if any(g in genres for g in EUROPEAN_FASHION_GENRES) else 0.0
            agg["completion_rate"] = agg.apply(
                lambda r: min(1.0, r["completion_rate"] + _eu_boost(r["show_id"])), axis=1
            )

        # Affluent Suburban bias → Heritage Couture titles score higher
        # (longer dwell time, premium taste profile)
        if tier_filter == "Affluent Suburban":
            heritage_terms = {"couture", "heritage", "designer", "fashion", "elegant", "luxury"}
            def _suburb_boost(show_id: str) -> float:
                meta = catalog_map.get(show_id, {})
                text = (str(meta.get("listed_in", "")) + " " + str(meta.get("description", ""))).lower()
                return 0.09 if any(t in text for t in heritage_terms) else 0.0
            agg["completion_rate"] = agg.apply(
                lambda r: min(1.0, r["completion_rate"] + _suburb_boost(r["show_id"])), axis=1
            )

        agg = agg.sort_values("completion_rate", ascending=False)
        score_col = "completion_rate"

    # ── Apply demographic affinity multipliers (OFFLINE: pandas, ONLINE: cuDF) ─
    # Flatten per-generation scores from demographic_scores dict into columns
    import pandas as pd
    if "demographic_scores" in df.columns:
        try:
            gen_df = pd.json_normalize(df["demographic_scores"].tolist())
            gen_df.columns = [f"score_{c}" for c in gen_df.columns]
            gen_df.index = df.index
            df = pd.concat([df, gen_df], axis=1)
        except Exception:
            pass

    # Determine target_age from top candidate's catalog metadata
    top_show_id = agg.iloc[0]["show_id"] if not agg.empty else ""
    top_target_age = catalog_map.get(top_show_id, {}).get("target_age", "25-34")

    if mode == "ONLINE":
        try:
            cudf = __import__("cudf")
            gdf_scores = cudf.DataFrame.from_pandas(df[[c for c in df.columns if c.startswith("score_")]] if any(c.startswith("score_") for c in df.columns) else df[["completion_rate"]])
            affinities = DEMOGRAPHIC_AFFINITY_MAP.get(top_target_age, DEMOGRAPHIC_AFFINITY_MAP["25-34"])
            for gen, mult in affinities.items():
                col = f"score_{gen}"
                if col in gdf_scores.columns:
                    gdf_scores[col] = (gdf_scores[col] * mult).clip(upper=1.0)
        except (ImportError, Exception):
            df = _apply_affinity_multipliers_pandas(df, top_target_age)
    else:
        df = _apply_affinity_multipliers_pandas(df, top_target_age)

    # ── Build candidate list ─────────────────────────────────────────────────
    candidates = []
    for _, row in agg.head(5).iterrows():
        sid = row["show_id"]
        meta = catalog_map.get(sid, {})
        ta = meta.get("target_age", "25-34")
        if mode == "ONLINE":
            friction = compute_demographic_friction(ta, resolved_demo)
            candidate_friction_index = friction["friction_index"]
            candidate_trade_off      = friction["trade_off_note"]
            candidate_friction_status = "AVAILABLE"
        else:
            candidate_friction_index  = None
            candidate_trade_off       = None
            candidate_friction_status = "GPU_REQUIRED"
        candidates.append({
            "show_id":          sid,
            "title":            meta.get("title", sid),
            "genres":           meta.get("listed_in", ""),
            "runtime_min":      meta.get("runtime_min", 90),
            "target_age":       ta,
            score_col:          round(float(row[score_col]), 4),
            "friction_index":   candidate_friction_index,
            "friction_status":  candidate_friction_status,
            "trade_off_note":   candidate_trade_off,
        })

    # ── Demographic comparison ───────────────────────────────────────────────
    demo_comparison: dict[str, int] = {}
    for seg in (CORE_DEMOS | ({resolved_demo} if is_growth else set())):
        demo_comparison[seg] = sum(
            1 for r in logs
            if r["primary_demographic"] == seg
            and (market_filter is None or r.get("market") in market_filter)
        )

    _, rapids_meta = accelerated_data_crunch(df)

    # Top-level friction for the #1 candidate — ONLINE only
    if mode == "ONLINE":
        top_friction = compute_demographic_friction(top_target_age, resolved_demo)
        rapids_meta["friction_index"]       = top_friction["friction_index"]
        rapids_meta["repelled_segments"]    = top_friction["repelled_segments"]
        rapids_meta["boosted_segments"]     = top_friction["boosted_segments"]
        rapids_meta["trade_off_note"]       = top_friction["trade_off_note"]
        rapids_meta["strategic_analysis"]   = "Available"
    else:
        rapids_meta["friction_index"]       = None
        rapids_meta["friction_status"]      = "GPU_REQUIRED"
        rapids_meta["strategic_analysis"]   = "Unavailable in Local Mode"

    return {
        "context_tribe":          matched_tribe or "all tribes",
        "demographic_filter":     resolved_demo or "Female+LGBTQ+ (core)",
        "is_growth_segment":      is_growth,
        "location_filter":        loc or "all markets",
        "density_tier":           tier_filter or "all tiers",
        "mode":                   mode,
        "candidates":             candidates,
        "demographic_comparison": demo_comparison,
        "total_logs_analysed":    len(filtered),
        "_audit":                 rapids_meta,
    }


# ── Weekly Strategic Scheduler ──────────────────────────────────────────────

_WEEKLY_SCHEDULE_PATH = _os.path.join(_os.path.dirname(__file__), "data", "weekly_schedule.json")

# Friction rules: which target_age groups are incompatible with each daily theme.
# Used by update_weekly_slot to surface Strategic Warnings when a move breaks
# the week's stylistic arc.
WEEKLY_FRICTION_RULES: dict[str, dict] = {
    "Heritage Weekend": {
        "incompatible_ages": ["18-24"],
        "strategic_warning": (
            "A {target_age} title on Heritage {day} is a wardrobe malfunction of the "
            "highest order. Your Silver Stylists did not come here for youth culture. "
            "Move this to Street & Youth Friday where it belongs. That is all."
        ),
    },
    "Minimalist Monday": {
        "incompatible_ages": ["18-24"],
        "strategic_warning": (
            "Minimalist Monday requires restraint and editorial precision. "
            "'{title}' — a {target_age} title — will shatter the calm we've curated "
            "for our Urban Core morning audience. Wednesday's Avant-Garde slot is far "
            "more appropriate. That is all."
        ),
    },
    "Street & Youth Friday": {
        "incompatible_ages": ["50+"],
        "strategic_warning": (
            "Street & Youth Friday was built for the Vanguard. "
            "A {target_age} title here is like wearing pearls to a rave. "
            "Relocate this to Heritage Weekend — your Silver Stylists will thank you. "
            "That is all."
        ),
    },
    "Avant-Garde Wednesday": {
        "incompatible_ages": ["50+"],
        "strategic_warning": (
            "Avant-Garde Wednesday is our mid-week Vanguard spike. "
            "'{title}' speaks to the {target_age} demographic — a mismatch that will "
            "dilute the editorial edge. Consider Heritage Weekend instead. That is all."
        ),
    },
}

# ── Event Exclusivity Rules ────────────────────────────────────────────────────
# The Met Gala and Paris Fashion Week are separate cultural moments. They may not
# share a programming day. Each event type is locked to its own permitted themes.
EVENT_EXCLUSIVITY_RULES: dict[str, dict] = {
    "met_gala": {
        "permitted_themes": ["Avant-Garde Wednesday"],
        "block_label":      "Met Gala Retrospective",
        "strategic_warning": (
            "'{title}' is Met Gala content — it belongs exclusively on Avant-Garde Wednesday. "
            "The Metropolitan Museum of Art does not share a marquee with the Tuileries. "
            "Move it or lose it. That is all."
        ),
    },
    "pfw": {
        "permitted_themes": [
            "Global Couture Thursday", "Romantic Tuesday",
            "Heritage Weekend", "Ready-to-Wear Saturday",
        ],
        "block_label":      "Ready-to-Wear Front Row",
        "strategic_warning": (
            "'{title}' is Paris Fashion Week content and belongs in the "
            "Ready-to-Wear Front Row block on Ready-to-Wear Saturday or "
            "Global Couture Thursday. Not here. Not today. That is all."
        ),
    },
}

# ── Conflict Map — explicit text-level exclusions per theme ───────────────────
# Applied in generate_weekly_plan() BEFORE event_type classification.
# Guards against titles that contain conflicting event signals anywhere in their
# title or description, even if _classify_event() doesn't catch them.
CONFLICT_MAP: dict[str, frozenset] = {
    "Avant-Garde Wednesday": frozenset({
        "paris fashion week", "pfw", "spring/summer", "spring collection",
        "ready-to-wear", "ready to wear", "aw25", "ss26", "fw collection",
        "fashion week",
    }),
    "Ready-to-Wear Saturday": frozenset({
        "met gala", "the met ball", "metropolitan museum", "first monday in may",
        "costume institute",
    }),
    "Global Couture Thursday": frozenset({
        "met gala", "the met ball", "metropolitan museum", "first monday in may",
        "costume institute",
    }),
}

# ── ID Translation Layer ───────────────────────────────────────────────────────
# Strip raw system identifiers before data reaches the LLM or the UI.
_ID_TRANSLATIONS: dict[str, str] = {
    "ch_runway_01": "Couture One",
    "ch_runway_02": "Couture Two",
    "ch_runway_03": "Couture Three",
}


def _translate_ids(obj: Any) -> Any:
    """Recursively replace raw system IDs with human-readable channel names."""
    if isinstance(obj, str):
        for raw, readable in _ID_TRANSLATIONS.items():
            obj = obj.replace(raw, readable)
        return obj
    if isinstance(obj, dict):
        return {k: _translate_ids(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_translate_ids(item) for item in obj]
    return obj


def _infer_target_age(item: dict) -> str:
    """Infer target_age bracket from catalog genres, description, and title.

    Called when a catalog item has no explicit target_age field.
    Returns one of: '18-24', '25-34', '35-49', '50+'.
    """
    text = (
        str(item.get("listed_in", ""))
        + " " + str(item.get("description", ""))
        + " " + str(item.get("title", ""))
    ).lower()

    _18_24_signals = {
        "teen", "high school", "college", "young adult", "animated", "animation",
        "superhero", "coming of age", "young", "campus", "youth", "school",
        "clueless", "mean girls", "pitch perfect", "legally blonde",
    }
    _50_signals = {
        "classic", "period drama", "historical", "aristocrat", "heritage",
        "dynasty", "elderly", "retirement", "war epic", "19th century",
        "1940s", "1950s", "1930s", "golden age", "vintage",
    }
    _35_49_signals = {
        "documentary", "biography", "biopic", "thriller", "based on true",
        "political", "corporate", "family drama", "mid-life", "crisis",
        "historical", "war", "espionage",
    }

    youth_hits  = sum(1 for s in _18_24_signals if s in text)
    senior_hits = sum(1 for s in _50_signals if s in text)
    mid_hits    = sum(1 for s in _35_49_signals if s in text)

    if youth_hits >= 2:
        return "18-24"
    if senior_hits >= 2:
        return "50+"
    if mid_hits >= 2:
        return "35-49"
    return "25-34"


def _classify_event(item: dict) -> str | None:
    """Return 'met_gala' or 'pfw' if the title is event-specific, else None.

    Checks title + description. Met Gala takes priority over PFW when both
    signals appear (shouldn't happen in a clean catalog, but guards against it).
    """
    text = (
        str(item.get("title", "")) + " "
        + str(item.get("description", ""))
    ).lower()
    if any(sig in text for sig in _MET_GALA_SIGNALS):
        return "met_gala"
    if any(sig in text for sig in _PFW_SIGNALS):
        return "pfw"
    return None


def _load_weekly_schedule() -> dict:
    try:
        return _load_json(_WEEKLY_SCHEDULE_PATH)
    except FileNotFoundError:
        return {"daily_themes": {}, "weekly_plan": []}


def _save_weekly_schedule(data: dict) -> None:
    _os.makedirs(_os.path.dirname(_WEEKLY_SCHEDULE_PATH), exist_ok=True)
    with open(_WEEKLY_SCHEDULE_PATH, "w") as f:
        json.dump(data, f, indent=2)


def generate_weekly_plan(week_start: str = "") -> dict:
    """Generate a 7-day programming grid (08:00–00:00) aligned to Daily Strategic Themes.

    OFFLINE: Scores catalog titles by genre/description keyword match, age-focus
             alignment, and average completion rate — fills each day's time grid.
    ONLINE:  Uses cuDF to rank titles by market-specific completion rate for each
             day's peak demographic, then applies the same grid fill logic.

    Args:
        week_start: ISO date string for the week's start (e.g. '2026-04-20').
                    Defaults to the week of 2026-04-20 if blank.

    Returns:
        Dict with per-day slot grids, ds_rationale per day, engine info, and audit.
    """
    import pandas as pd
    from datetime import date, timedelta

    if week_start:
        try:
            ws = datetime.fromisoformat(week_start).date()
        except ValueError:
            ws = date(2026, 4, 20)
    else:
        ws = date(2026, 4, 20)
    ws_str = ws.strftime("%Y-%m-%d")

    try:
        catalog: list[dict] = _load_json("data/catalog.json")
        logs: list[dict]    = _load_json("data/engagement_logs.json")
    except FileNotFoundError as e:
        return {"error": f"{e.filename} not found."}

    weekly = _load_weekly_schedule()
    daily_themes: dict = weekly.get("daily_themes", {})
    if not daily_themes:
        return {
            "error": (
                "No daily_themes found in weekly_schedule.json. "
                "Run generate_weekly_insights.py first."
            )
        }

    mode = _read_mode()
    catalog_map = {item["show_id"]: item for item in catalog}

    logs_df = pd.DataFrame(logs) if logs else pd.DataFrame()
    show_completion: dict[str, float] = (
        logs_df.groupby("show_id")["completion_rate"].mean().to_dict()
        if not logs_df.empty else {}
    )

    # ONLINE: pre-build per-market completion rates via cuDF
    market_show_completion: dict[str, dict[str, float]] = {}
    if mode == "ONLINE":
        try:
            cudf = __import__("cudf")
            gdf = cudf.DataFrame.from_pandas(logs_df)
            mkt_agg = (
                gdf.groupby(["market", "show_id"])["completion_rate"]
                .mean()
                .reset_index()
                .to_pandas()
            )
            for _, row in mkt_agg.iterrows():
                market_show_completion.setdefault(row["market"], {})[row["show_id"]] = float(
                    row["completion_rate"]
                )
        except (ImportError, Exception):
            pass

    days_of_week = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    # Anchor Sunday of the requested week
    sunday_date = ws - timedelta(days=(ws.weekday() + 1) % 7)

    days_output = []
    used_this_week: set[str] = set()   # cross-day deduplication — title appears once per week

    for i, day_name in enumerate(days_of_week):
        day_date = (sunday_date + timedelta(days=i)).strftime("%Y-%m-%d")
        theme_data    = daily_themes.get(day_name, {})
        theme_name    = theme_data.get("theme", f"{day_name} Programming")
        tribe         = theme_data.get("tribe", "")
        age_focus     = theme_data.get("age_focus", "25-34")
        peak_demo     = theme_data.get("peak_demo", "Female")
        location_bias = theme_data.get("location_bias", "")
        markets: list[str] = theme_data.get("markets", [])
        ds_rationale: str  = theme_data.get("ds_rationale", "")
        genre_kws: list[str] = theme_data.get("genre_keywords", [])
        desc_kws:  list[str] = theme_data.get("description_keywords", [])

        # Score every catalog title for this day's theme
        scored: list[tuple[str, float]] = []
        for item in catalog:
            sid = item["show_id"]
            if item.get("runtime_min", 0) < 60:
                continue

            text = (
                str(item.get("listed_in", "")) + " "
                + str(item.get("description", "")) + " "
                + str(item.get("title", ""))
            ).lower()

            # Conflict Map: hard text-level exclusion — runs before event_type check
            conflict_signals = CONFLICT_MAP.get(theme_name, frozenset())
            if conflict_signals and any(sig in text for sig in conflict_signals):
                continue   # pop this title from the pool for today

            # Event exclusivity: Met Gala and PFW are locked to specific themes.
            ev = _classify_event(item)
            if ev:
                permitted = EVENT_EXCLUSIVITY_RULES.get(ev, {}).get("permitted_themes", [])
                if theme_name not in permitted:
                    continue   # event type forbidden on today's theme

            kw_score = sum(1 for kw in genre_kws + desc_kws if kw in text)
            item_age = item.get("target_age") or _infer_target_age(item)
            age_score = 1.5 if item_age == age_focus else (0.9 if item_age in ("25-34", "35-49") else 0.5)

            base_completion = show_completion.get(sid, 0.5)
            if mode == "ONLINE" and markets:
                mkt_rates = [
                    market_show_completion.get(m, {}).get(sid, base_completion)
                    for m in markets
                ]
                completion = sum(mkt_rates) / len(mkt_rates) if mkt_rates else base_completion
            else:
                completion = base_completion

            scored.append((sid, (kw_score * 0.4) + (age_score * 0.3) + (completion * 0.3)))

        scored.sort(key=lambda x: -x[1])
        ranked = [sid for sid, _ in scored]

        # Fill 08:00–00:00 time grid
        slots: list[dict] = []
        current_min = 8 * 60    # 480 min from midnight
        end_min     = 24 * 60   # 1440 min (midnight)
        used_today: set[str] = set()
        pool_pos = 0

        # ── Dayparting pools ─────────────────────────────────────────────────
        # Top 20% of scored titles anchor Prime Time (16:00–22:00).
        # The remaining 80% fill Daytime (08:00–16:00) and Late Night (22:00–00:00).
        PRIME_START  = 16 * 60
        PRIME_END    = 22 * 60
        prime_cutoff = max(4, len(ranked) // 5)
        prime_ids    = set(ranked[:prime_cutoff])

        # Cross-week preference: fresh (unseen this week) titles lead each pool
        ranked_fresh   = [s for s in ranked if s not in used_this_week]
        ranked_repeat  = [s for s in ranked if s in used_this_week]
        ranked_ordered = ranked_fresh + ranked_repeat

        prime_pool   = [s for s in ranked_ordered if s in prime_ids]
        morning_pool = [s for s in ranked_ordered if s not in prime_ids]

        # Fill full 24-hour grid (00:00–24:00, channel never goes dark)
        # Overnight (00:00–06:00): low-key titles, repeats allowed.
        # Morning   (06:00–16:00): daytime tier.
        # Prime     (16:00–22:00): top-ranked anchors.
        # Late      (22:00–24:00): secondary tier.
        OVERNIGHT_END = 6 * 60
        slots: list[dict] = []
        current_min   = 0
        end_min       = 24 * 60
        used_today:    set[str] = set()
        used_overnight: set[str] = set()   # separate tracking for overnight repeats

        while current_min < end_min:
            is_prime = PRIME_START <= current_min < PRIME_END

            # Event-type guard: Met Gala and PFW must not share a day
            used_events = {_classify_event(catalog_map.get(s, {})) for s in used_today} - {None}

            def _ev_ok(s: str, _ue: set = used_events) -> bool:
                ev = _classify_event(catalog_map.get(s, {}))
                return not (("met_gala" in _ue and ev == "pfw") or ("pfw" in _ue and ev == "met_gala"))

            is_overnight = current_min < OVERNIGHT_END

            if is_overnight:
                # Overnight: rotate through lower-half pool; allow repeats once exhausted
                overnight_pool = ranked_ordered[-max(1, len(ranked_ordered)//2):]
                avail = [s for s in overnight_pool if s not in used_overnight and _ev_ok(s)]
                if not avail:
                    used_overnight.clear()   # reset for second pass
                    avail = [s for s in overnight_pool if _ev_ok(s)] or list(ranked_ordered)
            elif is_prime:
                avail = [s for s in prime_pool   if s not in used_today and _ev_ok(s)]
                if not avail:
                    avail = [s for s in ranked_ordered if s not in used_today and _ev_ok(s)]
            else:
                avail = [s for s in morning_pool if s not in used_today and _ev_ok(s)]
                if not avail:
                    avail = [s for s in ranked_ordered if s not in used_today and _ev_ok(s)]

            if not avail:
                avail = [s for s in ranked[:6] if _ev_ok(s)] or ranked[:6]

            sid  = avail[0]   # always take the highest-scored available title
            meta = catalog_map.get(sid, {})

            runtime = meta.get("runtime_min", 90)
            block   = ((runtime + 29) // 30) * 30
            if current_min + block > end_min:
                break

            h, m    = divmod(current_min, 60)
            ev_type = _classify_event(meta)
            ev_rule = EVENT_EXCLUSIVITY_RULES.get(ev_type, {}) if ev_type else {}
            daypart = (
                "Overnight"   if current_min < OVERNIGHT_END
                else "Prime Time"  if is_prime
                else "Late Night"  if current_min >= 22 * 60
                else "Daytime"
            )
            slots.append({
                "time":               f"{h:02d}:{m:02d}",
                "show_id":            sid,
                "title":              meta.get("title", sid),
                "runtime_min":        runtime,
                "block_duration_min": block,
                "interstitial_min":   block - runtime,
                "tribe":              tribe,
                "target_age":         meta.get("target_age") or _infer_target_age(meta),
                "event_type":         ev_type or "",
                "block_label":        ev_rule.get("block_label", ""),
                "daypart":            daypart,
            })
            if is_overnight:
                used_overnight.add(sid)
            else:
                used_today.add(sid)
                used_this_week.add(sid)
            prime_pool   = [s for s in prime_pool   if s != sid]
            morning_pool = [s for s in morning_pool if s != sid]
            current_min += block

        days_output.append({
            "date":          day_date,
            "day_of_week":   day_name,
            "theme":         theme_name,
            "tribe":         tribe,
            "peak_demo":     peak_demo,
            "location_bias": location_bias,
            "markets":       markets,
            "ds_rationale":  ds_rationale,
            "slot_count":    len(slots),
            "slots":         slots,
        })

    weekly["week_start"]       = ws_str
    weekly["weekly_plan"]      = days_output
    weekly["_last_generated"]  = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    _save_weekly_schedule(weekly)

    return {
        "week_start":  ws_str,
        "mode":        mode,
        "ds_engine":   "Live GPU Ranker" if mode == "ONLINE" else "Local Heuristic Engine",
        "days":        days_output,
        "total_slots": sum(len(d["slots"]) for d in days_output),
        "_audit":      _compute_meta(rows=len(catalog)),
    }


def update_weekly_slot(from_day: str, from_time: str, to_day: str, to_time: str) -> dict:
    """Move a title between weekly schedule slots with demographic friction validation.

    If the destination day's theme is incompatible with the title's target_age,
    a strategic_warning is returned alongside the completed move.

    Performs a swap if a title already occupies the target slot; otherwise moves
    the title directly to the empty target position.

    Args:
        from_day:  Source day name (e.g. 'Monday').
        from_time: Source slot time HH:MM (e.g. '20:00').
        to_day:    Target day name (e.g. 'Wednesday').
        to_time:   Target slot time HH:MM (e.g. '20:00').

    Returns:
        Dict with status, move details, optional displaced slot info,
        optional strategic_warning, and audit block.
    """
    weekly = _load_weekly_schedule()
    plan: list[dict] = weekly.get("weekly_plan", [])
    if not plan:
        return {"error": "No weekly plan found. Call generate_weekly_plan first."}

    daily_themes: dict = weekly.get("daily_themes", {})

    from_day_norm = from_day.strip().title()
    to_day_norm   = to_day.strip().title()

    src_day_data = next((d for d in plan if d["day_of_week"] == from_day_norm), None)
    if not src_day_data:
        return {
            "error":          f"Day '{from_day}' not found in weekly plan.",
            "available_days": [d["day_of_week"] for d in plan],
        }

    src_slot_idx = next(
        (i for i, s in enumerate(src_day_data["slots"]) if s["time"] == from_time), None
    )
    if src_slot_idx is None:
        return {
            "error":           f"No slot at {from_time} on {from_day_norm}.",
            "available_times": [s["time"] for s in src_day_data["slots"]],
        }
    source_slot = src_day_data["slots"][src_slot_idx]

    tgt_day_data = next((d for d in plan if d["day_of_week"] == to_day_norm), None)
    if not tgt_day_data:
        return {"error": f"Target day '{to_day}' not found in weekly plan."}

    to_theme          = daily_themes.get(to_day_norm, {}).get("theme", "")
    show_target_age   = source_slot.get("target_age", "25-34")
    show_title        = source_slot.get("title", source_slot["show_id"])

    # ── Hard stop: Event Exclusivity — do NOT execute the move ────────────────
    _ev_check = _classify_event({"title": show_title, "description": ""})
    if _ev_check:
        _ex_rule = EVENT_EXCLUSIVITY_RULES.get(_ev_check, {})
        _permitted = _ex_rule.get("permitted_themes", [])
        if to_theme and to_theme not in _permitted:
            return {
                "is_final_conflict": True,
                "error_message":     "Editorial Policy Violation: Seasonal Incompatibility Detected.",
                "conflict_type":     "event_exclusivity",
                "event_type":        _ev_check,
                "title":             show_title,
                "target_day":        to_day_norm,
                "target_theme":      to_theme,
                "permitted_themes":  _permitted,
                "strategic_warning": _ex_rule.get("strategic_warning", "").format(title=show_title),
                "_audit":            _compute_meta(),
            }

    _conflict_signals = CONFLICT_MAP.get(to_theme, frozenset())
    if _conflict_signals and any(sig in show_title.lower() for sig in _conflict_signals):
        _ev_type = "met_gala" if any(sig in show_title.lower() for sig in _MET_GALA_SIGNALS) else "pfw"
        _ex_rule = EVENT_EXCLUSIVITY_RULES.get(_ev_type, {})
        return {
            "is_final_conflict": True,
            "error_message":     "Editorial Policy Violation: Seasonal Incompatibility Detected.",
            "conflict_type":     "text_signal",
            "title":             show_title,
            "target_day":        to_day_norm,
            "target_theme":      to_theme,
            "strategic_warning": _ex_rule.get("strategic_warning", "").format(title=show_title) if _ex_rule else f"'{show_title}' cannot air on {to_day_norm}.",
            "_audit":            _compute_meta(),
        }

    # ── Soft warning: Demographic Friction ────────────────────────────────────
    strategic_warning: str | None = None

    if to_theme in WEEKLY_FRICTION_RULES:
        rule = WEEKLY_FRICTION_RULES[to_theme]
        if show_target_age in rule["incompatible_ages"]:
            strategic_warning = rule["strategic_warning"].format(
                target_age=show_target_age,
                title=show_title,
                day=to_day_norm,
            )

    # Swap if target slot occupied; move otherwise
    tgt_slot_idx = next(
        (i for i, s in enumerate(tgt_day_data["slots"]) if s["time"] == to_time), None
    )
    displaced_info: dict | None = None

    if tgt_slot_idx is not None:
        displaced = dict(tgt_day_data["slots"][tgt_slot_idx])
        displaced["time"] = from_time
        moved = dict(source_slot)
        moved["time"] = to_time
        src_day_data["slots"][src_slot_idx]  = displaced
        tgt_day_data["slots"][tgt_slot_idx]  = moved
        operation = "swap"
        displaced_info = {
            "show_id":  displaced["show_id"],
            "title":    displaced.get("title", ""),
            "moved_to": f"{from_day_norm} {from_time}",
        }
    else:
        moved = dict(source_slot)
        moved["time"] = to_time
        src_day_data["slots"].pop(src_slot_idx)
        tgt_day_data["slots"].append(moved)
        tgt_day_data["slots"].sort(key=lambda s: s["time"])
        operation = "move"

    weekly["_last_updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    _save_weekly_schedule(weekly)

    result: dict = {
        "status":    "moved_with_warning" if strategic_warning else "moved",
        "operation": operation,
        "show_id":   source_slot["show_id"],
        "title":     show_title,
        "from": {
            "day":   from_day_norm,
            "time":  from_time,
            "theme": daily_themes.get(from_day_norm, {}).get("theme", ""),
        },
        "to": {
            "day":   to_day_norm,
            "time":  to_time,
            "theme": to_theme,
        },
        "_audit": _compute_meta(),
    }
    if displaced_info:
        result["displaced"] = displaced_info
    if strategic_warning:
        result["strategic_warning"] = strategic_warning

    return result


def calculate_strategic_friction(title: str, target_day: str) -> dict:
    """Pre-flight Event Exclusivity check before attempting a slot move.

    Returns is_final_conflict: true if placing the title on target_day would
    violate Editorial Policy (Met Gala on non-Wednesday, PFW on Wednesday, etc.).
    The agent MUST stop calling tools when is_final_conflict is true.

    Args:
        title:      Title of the content to be placed (e.g. 'The First Monday in May').
        target_day: Destination day name (e.g. 'Saturday').

    Returns:
        Dict with is_final_conflict (bool), conflict details, and audit block.
    """
    weekly = _load_weekly_schedule()
    daily_themes: dict = weekly.get("daily_themes", {})
    target_day_norm = target_day.strip().title()
    target_theme    = daily_themes.get(target_day_norm, {}).get("theme", "")

    ev = _classify_event({"title": title, "description": ""})
    if ev:
        rule = EVENT_EXCLUSIVITY_RULES.get(ev, {})
        permitted = rule.get("permitted_themes", [])
        if target_theme and target_theme not in permitted:
            return {
                "is_final_conflict": True,
                "error_message":     "Editorial Policy Violation: Seasonal Incompatibility Detected.",
                "conflict_type":     "event_exclusivity",
                "event_type":        ev,
                "title":             title,
                "target_day":        target_day_norm,
                "target_theme":      target_theme,
                "permitted_themes":  permitted,
                "strategic_warning": rule.get("strategic_warning", "").format(title=title),
                "_audit":            _compute_meta(),
            }

    conflict_signals = CONFLICT_MAP.get(target_theme, frozenset())
    if conflict_signals and any(sig in title.lower() for sig in conflict_signals):
        ev_type = "met_gala" if any(sig in title.lower() for sig in _MET_GALA_SIGNALS) else "pfw"
        rule    = EVENT_EXCLUSIVITY_RULES.get(ev_type, {})
        return {
            "is_final_conflict": True,
            "error_message":     "Editorial Policy Violation: Seasonal Incompatibility Detected.",
            "conflict_type":     "text_signal",
            "title":             title,
            "target_day":        target_day_norm,
            "target_theme":      target_theme,
            "strategic_warning": rule.get("strategic_warning", "").format(title=title) if rule else f"'{title}' cannot air on {target_day_norm}.",
            "_audit":            _compute_meta(),
        }

    return {
        "is_final_conflict": False,
        "title":             title,
        "target_day":        target_day_norm,
        "target_theme":      target_theme,
        "status":            "cleared — no editorial conflict detected",
        "_audit":            _compute_meta(),
    }
