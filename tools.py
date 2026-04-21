"""
FAST Channel Media Intelligence – Tool Functions
Each function reads from pre-generated JSON files in data/.
"""

import json
from datetime import datetime, timezone
from typing import Any


def _load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


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

    return json.dumps(blueprint, indent=2)


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
    recommendations.sort(key=lambda x: x["score"], reverse=True)

    return {
        "style_tribe":           matched_tribe,
        "tribe_designers":       tribe_designers,
        "tribe_signal_terms":    signal_terms,
        "catalog_recommendations": recommendations[:10],
        "source":                "Condé Nast Accelerated Intelligence Layer (GPU-clustered)",
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


def get_audience_telemetry(content_id: str, current_time: str) -> list[dict]:
    """Return viewership demographic data for an asset at a given hour.

    Matches telemetry records whose timestamp shares the same UTC date and hour
    as *current_time*, across all demographics.

    Args:
        content_id:   The show identifier (e.g. 's0001').
        current_time: ISO-8601 timestamp string used to select the time window.

    Returns:
        A list of telemetry record dicts for that asset and hour, sorted by
        viewers descending. Returns a list with a single error dict on failure.
    """
    try:
        records: list[dict] = _load_json("data/telemetry.json")
    except FileNotFoundError:
        return [{"error": "data/telemetry.json not found."}]

    try:
        query_dt = datetime.fromisoformat(current_time)
        if query_dt.tzinfo is None:
            query_dt = query_dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return [{"error": f"Invalid timestamp format: '{current_time}'."}]

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
        return [
            {
                "error": (
                    f"No telemetry for '{content_id}' at "
                    f"{query_dt.strftime('%Y-%m-%d %H:00 UTC')}."
                )
            }
        ]

    return sorted(matches, key=lambda r: r.get("viewers", 0), reverse=True)
