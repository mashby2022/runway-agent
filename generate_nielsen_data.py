"""
generate_nielsen_data.py — Nielsen Measurement Layer

Enriches two existing data files with standard Nielsen fields, then builds a
clean aggregated view suited for UI display and tool responses.

Outputs
───────
  data/telemetry.json         (overwritten) + Nielsen fields per record
  data/engagement_logs.json   (overwritten) + DMA-level Nielsen constants
  data/nielsen_telemetry.json (new)          aggregated DMA × show × daypart view

Nielsen formulas used
─────────────────────
  Rating_Pct       = (Audience / UniverseEstimate_UE) × 100
  Share_Pct        = Rating_Pct / (HUT × 100) × 100
  AverageAudience  = Audience / 1 000
  GrossImpressions = Audience (per spot, same unit)
  Reach_Pct        ≈ Rating_Pct × 1.12  (unduplicated estimate, single airing)
  Frequency        = GrossImpressions / (UniverseEstimate_UE × Reach_Pct/100)
  MediaCost        = (GrossImpressions / 1 000) × CPM_rate
  CPM              = MediaCost / (GrossImpressions / 1 000)
  CPP              = MediaCost / GRPs

Run standalone:
    python generate_nielsen_data.py
"""

import json
import math
import os as _os
import os
import random
from datetime import datetime, timezone

random.seed(42)

# ── DMA Universe Estimates — TV Households (2026 est.) ───────────────────────
# Source: Nielsen 2024/2025 DMA reports, adjusted for 2026 growth projections.
DMA_UNIVERSE_HH: dict[str, int] = {
    "New York":      7_440_000,
    "Los Angeles":   5_610_000,
    "Chicago":       3_480_000,
    "Dallas":        2_720_000,
    "San Francisco": 2_510_000,
    "Philadelphia":  2_960_000,
    "Atlanta":       2_550_000,
    "Paris":         4_250_000,   # Paris Île-de-France metro TV HH
    "London":        3_820_000,   # Greater London TV HH
    "Milan":         1_480_000,   # Milan metro TV HH
}
DEFAULT_UNIVERSE_HH = 2_000_000

# ch_runway_01 national FAST channel reach (sum of all market penetrations)
CHANNEL_UNIVERSE_HH = 15_200_000

# ── CPM Rates by market — Female / luxury / fashion audience ─────────────────
# Premium segment commands 40-65 % above standard cable CPMs.
CPM_BY_MARKET: dict[str, float] = {
    "New York":      58.0,
    "Los Angeles":   45.0,
    "Chicago":       35.0,
    "Dallas":        28.0,
    "San Francisco": 42.0,
    "Philadelphia":  32.0,
    "Atlanta":       26.0,
    "Paris":         52.0,
    "London":        48.0,
    "Milan":         41.0,
    "_default":      30.0,
}
CHANNEL_CPM = 45.0   # national blended rate for telemetry records (no DMA)

# ── HUT — Gaussian prime-time model (replaces flat lookup table) ──────────────
# Bell curve centered at 20:30 (σ=1.8 h) layered over a daytime arch baseline.
# Range: ~0.06 overnight → ~0.74 peak prime time.
_HUT_PRIME_CENTER = 20.5
_HUT_PRIME_SIGMA  = 1.8
_HUT_DAY_CENTER   = 14.0
_HUT_DAY_SIGMA    = 6.0

def _gaussian_hut(hour: int) -> float:
    """HUT with Gaussian prime-time spike + organic noise. Range 0.05–0.82."""
    daytime = 0.10 + 0.30 * math.exp(-0.5 * ((hour - _HUT_DAY_CENTER) / _HUT_DAY_SIGMA) ** 2)
    prime   = 0.44 * math.exp(-0.5 * ((hour - _HUT_PRIME_CENTER) / _HUT_PRIME_SIGMA) ** 2)
    noise   = random.gauss(0.0, 0.012)
    return round(max(0.05, min(0.82, daytime + prime + noise)), 4)

# Legacy constant kept for nielsen_telemetry aggregation (deterministic lookup)
HUT_BY_HOUR: dict[int, float] = {
    h: round(
        0.10 + 0.30 * math.exp(-0.5 * ((h - 14.0) / 6.0) ** 2)
        + 0.44 * math.exp(-0.5 * ((h - 20.5) / 1.8) ** 2),
        4,
    )
    for h in range(24)
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def _load(path: str):
    with open(path) as f:
        return json.load(f)


def _save(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _save_parquet(json_path: str, records: list) -> str:
    """Write a parallel .parquet file for DuckDB queries."""
    try:
        import pandas as pd
        parquet_path = json_path.replace(".json", ".parquet")
        pd.DataFrame(records).to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception as exc:
        print(f"  ⚠  Parquet write skipped: {exc}")
        return ""


def _get_hour(timestamp: str) -> int:
    try:
        return datetime.fromisoformat(timestamp).hour
    except (ValueError, TypeError):
        return 20  # fallback to prime time


def _nielsen_block(audience: int, universe: int, hut: float, cpm: float) -> dict:
    """Return the full 14-field Nielsen measurement block.

    All values rounded to sensible precision. CPP = 0 when GRPs is zero
    (no audience, no cost-per-point).
    """
    if universe <= 0 or audience <= 0:
        return {
            "UniverseEstimate_UE":  universe,
            "HouseholdsUsingTV_HUT": round(hut * 100, 2),
            "PersonsViewingTV_PUT":  round(hut * 95, 2),
            "Audience_HH_or_Persons": 0,
            "Rating_Pct":            0.0,
            "Share_Pct":             0.0,
            "AverageAudience_000":   0.0,
            "GRPs":                  0.0,
            "Reach_Pct":             0.0,
            "Frequency":             0.0,
            "GrossImpressions":      0,
            "MediaCost":             0.0,
            "CPM":                   cpm,
            "CPP":                   0.0,
        }

    rating      = (audience / universe) * 100
    share       = (rating / (hut * 100)) * 100 if hut > 0 else 0.0
    avg_aud_000 = audience / 1_000
    grps        = rating                             # single spot
    reach       = min(rating * 1.12, 100.0)
    frequency   = audience / max(universe * reach / 100, 1)
    impressions = audience
    cost        = (impressions / 1_000) * cpm
    cpp         = cost / grps if grps > 0 else 0.0

    return {
        "UniverseEstimate_UE":   universe,
        "HouseholdsUsingTV_HUT": round(hut * 100, 2),     # expressed as %
        "PersonsViewingTV_PUT":  round(hut * 95.0, 2),    # PUT ≈ HUT × 0.95
        "Audience_HH_or_Persons": audience,
        "Rating_Pct":            round(rating, 4),
        "Share_Pct":             round(share, 4),
        "AverageAudience_000":   round(avg_aud_000, 3),
        "GRPs":                  round(grps, 4),
        "Reach_Pct":             round(reach, 4),
        "Frequency":             round(frequency, 4),
        "GrossImpressions":      impressions,
        "MediaCost":             round(cost, 2),
        "CPM":                   round(cpm, 2),
        "CPP":                   round(cpp, 2),
    }


# ── Nielsen noise helper ──────────────────────────────────────────────────────

def _apply_nielsen_noise(block: dict) -> dict:
    """Add multiplicative Gaussian noise to Rating_Pct and Share_Pct.

    Produces organic decimals (e.g. 1.42 %, 2.17 %) rather than clean values.
    GRPs, Reach_Pct, CPP are recalculated to remain internally consistent.
    """
    r_noise = random.gauss(1.0, 0.09)   # ±9 % std dev
    s_noise = random.gauss(1.0, 0.11)   # ±11 % std dev (share varies more)
    block["Rating_Pct"] = round(max(0.0001, block["Rating_Pct"] * r_noise), 4)
    block["Share_Pct"]  = round(max(0.0001, block["Share_Pct"]  * s_noise), 4)
    block["GRPs"]       = block["Rating_Pct"]
    block["Reach_Pct"]  = round(min(block["Rating_Pct"] * 1.12, 100.0), 4)
    if block["GRPs"] > 0:
        block["CPP"] = round(block["MediaCost"] / block["GRPs"], 2)
    return block


# ── 1. Enrich telemetry.json (channel-level, no DMA) ────────────────────────

def enrich_telemetry(telemetry_path: str) -> list[dict]:
    print(f"Loading {telemetry_path} …")
    records = _load(telemetry_path)
    print(f"  {len(records):,} records — enriching with Nielsen fields …")

    enriched = []
    for rec in records:
        hour    = _get_hour(rec.get("timestamp", ""))
        hut     = _gaussian_hut(hour)
        viewers = int(rec.get("viewers", 0))

        nielsen = _apply_nielsen_noise(_nielsen_block(viewers, CHANNEL_UNIVERSE_HH, hut, CHANNEL_CPM))
        enriched.append({**rec, **nielsen})

    print(f"  ✓ telemetry.json enriched ({len(enriched):,} records)")
    return enriched


# ── 2. Enrich engagement_logs.json (DMA-level) ───────────────────────────────
# Each log row is a single viewing session. We derive an estimated per-session
# "audience equivalent" by scaling DMA universe × channel share × completion.
# CHANNEL_SHARE_OF_HUT = 0.0020  →  Runway Inclusive holds ~0.2 % of HUT
# (realistic for a niche FAST channel with a premium audience).

CHANNEL_SHARE_OF_HUT = 0.0020

def enrich_engagement_logs(logs_path: str) -> list[dict]:
    print(f"Loading {logs_path} …")
    records = _load(logs_path)
    print(f"  {len(records):,} records — enriching with DMA Nielsen fields …")

    enriched = []
    for rec in records:
        market   = rec.get("market", "_default")
        universe = DMA_UNIVERSE_HH.get(market, DEFAULT_UNIVERSE_HH)
        hour     = _get_hour(rec.get("timestamp", ""))
        hut      = _gaussian_hut(hour)
        cpm      = CPM_BY_MARKET.get(market, CPM_BY_MARKET["_default"])
        cr       = float(rec.get("completion_rate", 0.5))

        # Estimated viewers = universe × HUT × channel_share × completion uplift
        est_viewers = max(1, int(universe * hut * CHANNEL_SHARE_OF_HUT * cr))

        nielsen = _apply_nielsen_noise(_nielsen_block(est_viewers, universe, hut, cpm))
        enriched.append({**rec, **nielsen})

    print(f"  ✓ engagement_logs.json enriched ({len(enriched):,} records)")
    return enriched


# ── 3. Build aggregated nielsen_telemetry.json ───────────────────────────────
# Aggregate engagement_logs by (show_id, market, is_prime_time) for
# dashboard-ready market intelligence.

def build_nielsen_telemetry(enriched_logs: list[dict], catalog_path: str) -> list[dict]:
    print("Building aggregated nielsen_telemetry.json …")

    catalog = _load(catalog_path)
    catalog_map = {item["show_id"]: item for item in catalog}

    from collections import defaultdict
    buckets: dict[tuple, list] = defaultdict(list)
    for rec in enriched_logs:
        key = (rec["show_id"], rec.get("market", "_default"), rec.get("is_prime_time", False))
        buckets[key].append(rec)

    aggregated = []
    for (show_id, market, is_prime), rows in sorted(buckets.keys().__iter__() and buckets.items()):
        universe  = DMA_UNIVERSE_HH.get(market, DEFAULT_UNIVERSE_HH)
        cpm       = CPM_BY_MARKET.get(market, CPM_BY_MARKET["_default"])
        avg_hut   = sum(HUT_BY_HOUR.get(_get_hour(r.get("timestamp", "")), 0.5) for r in rows) / len(rows)
        avg_cr    = sum(float(r.get("completion_rate", 0.5)) for r in rows) / len(rows)

        # Aggregate audience: sum of individual est_viewers over the window
        total_audience = sum(int(r.get("Audience_HH_or_Persons", 0)) for r in rows)
        # Clamp to plausible single-hour ceiling
        max_audience   = int(universe * avg_hut * 0.006)  # 3× channel share ceiling
        audience       = min(total_audience, max_audience)

        nielsen = _nielsen_block(audience, universe, avg_hut, cpm)

        meta    = catalog_map.get(show_id, {})
        daypart = "Prime Time (16:00–22:00)" if is_prime else "Daytime / Late Night"

        aggregated.append({
            "show_id":         show_id,
            "title":           meta.get("title", show_id),
            "market":          market,
            "dma":             next(
                (r.get("dma", market) for r in rows if r.get("dma")), market
            ),
            "density_tier":    rows[0].get("density_tier", ""),
            "is_prime_time":   is_prime,
            "daypart":         daypart,
            "session_count":   len(rows),
            "avg_completion_rate": round(avg_cr, 4),
            "genres":          meta.get("listed_in", ""),
            **nielsen,
        })

    # Sort by GRPs descending — highest-rated content first
    aggregated.sort(key=lambda x: x["GRPs"], reverse=True)
    print(f"  ✓ nielsen_telemetry.json built ({len(aggregated):,} market×show×daypart records)")
    return aggregated


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    base = os.path.dirname(__file__)

    # Paths
    tel_path     = os.path.join(base, "data", "telemetry.json")
    eng_path     = os.path.join(base, "data", "engagement_logs.json")
    cat_path     = os.path.join(base, "data", "catalog.json")
    nielsen_path = os.path.join(base, "data", "nielsen_telemetry.json")

    print("══════════════════════════════════════════════════════")
    print("  Runway Inclusive — Nielsen Measurement Layer")
    print("══════════════════════════════════════════════════════\n")

    enriched_tel  = enrich_telemetry(tel_path)
    enriched_logs = enrich_engagement_logs(eng_path)
    nielsen_agg   = build_nielsen_telemetry(enriched_logs, cat_path)

    print("\nSaving …")
    _save(tel_path,     enriched_tel)
    _save(eng_path,     enriched_logs)
    _save(nielsen_path, nielsen_agg)
    print(f"  ✓ data/telemetry.json         ({len(enriched_tel):,} records)")
    print(f"  ✓ data/engagement_logs.json   ({len(enriched_logs):,} records)")
    print(f"  ✓ data/nielsen_telemetry.json ({len(nielsen_agg):,} records)")

    # Parquet output for DuckDB queries
    for path, records in [
        (tel_path, enriched_tel),
        (eng_path, enriched_logs),
        (nielsen_path, nielsen_agg),
    ]:
        pq = _save_parquet(path, records)
        if pq:
            print(f"  ✓ {_os.path.basename(pq)}")

    # ── Sample output ──────────────────────────────────────────────────────
    print("\n── Sample Nielsen Record (telemetry.json) ──")
    sample_tel = next((r for r in enriched_tel if r.get("viewers", 0) > 500), enriched_tel[0])
    nielsen_fields = [
        "UniverseEstimate_UE", "HouseholdsUsingTV_HUT", "PersonsViewingTV_PUT",
        "Audience_HH_or_Persons", "Rating_Pct", "Share_Pct", "AverageAudience_000",
        "GRPs", "Reach_Pct", "Frequency", "GrossImpressions",
        "MediaCost", "CPM", "CPP",
    ]
    sample_out = {
        "show_id":    sample_tel["show_id"],
        "title":      sample_tel["title"],
        "timestamp":  sample_tel["timestamp"],
        "viewer_type": sample_tel["viewer_type"],
        "viewers":    sample_tel["viewers"],
        **{k: sample_tel[k] for k in nielsen_fields if k in sample_tel},
    }
    print(json.dumps(sample_out, indent=2))

    print("\n── Sample Nielsen Record (nielsen_telemetry.json — NYC prime time) ──")
    nyc_prime = next(
        (r for r in nielsen_agg if r.get("market") == "New York" and r.get("is_prime_time")),
        nielsen_agg[0]
    )
    print(json.dumps(nyc_prime, indent=2))

    print("\nThat is all.")


if __name__ == "__main__":
    main()
