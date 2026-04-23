"""
generate_logs.py — Vault Intelligence Edition

Produces data/engagement_logs.json featuring:
  - Nielsen 'The Gauge' Proportions: 48% Streaming, 21% Broadcast, 20% Cable.
  - 2028 Thematic Projections: +35% Personal Growth, -50% Family-Daughter narratives.
  - Platform Logic: Tubi/Issa Rae boost (+20%) and Netflix Vault boosters.
  - Cultural Signals: Integration of music charts and podcast trend data.
"""

import json
import math
import os
import random
from datetime import date, datetime, timezone, timedelta

random.seed(42)

# ── Constants ─────────────────────────────────────────────────────────────────

BASE_DATE   = date(2026, 4, 14)
NUM_RECORDS = 5_000

# Nielsen 'The Gauge' Platform Weights (Dec 2025 / Apr 2026 proportions)
PLATFORM_CATEGORIES = ["Streaming", "Broadcast", "Cable", "Other"]
PLATFORM_WEIGHTS    = [0.48, 0.21, 0.20, 0.11]

STREAMING_SERVICES = ["YouTube", "Netflix", "Disney+", "Prime Video", "Tubi", "Hulu"]
STREAMING_WEIGHTS  = [0.25, 0.18, 0.10, 0.09, 0.08, 0.30]

# Vault 2028 thematic signals
PERSONAL_GROWTH_KWS = {"personal growth", "self-discovery", "empowerment", "career", "independence"}
FAMILY_DAUGHTER_KWS = {"family", "daughter", "mother-daughter", "parenting", "domestic"}

CULTURAL_SIGNALS = [
    "Spotify Global Top 50 Synergy",
    "Viral TikTok Aesthetic",
    "Podcast Sentiment Spike",
    "Election Cycle Engagement",
    "Archival Fashion Trend",
]

MARKETS = [
    ("New York",      "New York (DMA 1)",    "Urban Core"),
    ("Los Angeles",   "Los Angeles (DMA 2)", "Urban Core"),
    ("Dallas",        "Dallas (DMA 4)",      "Affluent Suburban"),
    ("Atlanta",       "Atlanta (DMA 7)",     "Exurban"),
    ("Paris",         "Paris (DMA 8)",       "Urban Core"),
    ("Milan",         "Milan (DMA 10)",      "Affluent Suburban"),
]

DEMOGRAPHICS = [
    "Female", "Male", "LGBT+", "Gen_Z", "Millennial", "Gen_X",
]

# Pareto quad tiers: (name, cumulative_weight, cr_lo, cr_hi)
_TIERS = [
    ("Occasional", 0.20, 0.15, 0.59),
    ("Silver",     0.57, 0.60, 0.84),
    ("Gold",       1.00, 0.85, 1.00),
]

PRIME_CENTER = 20.5   # 20:30
PRIME_SIGMA  = 1.5

# Tubi Issa Rae boost tags
_TUBI_BOOST_TAGS = {"black", "queer"}


# ── Platform picker ───────────────────────────────────────────────────────────

def _pick_platform() -> tuple:
    """Returns (category, service) weighted by Nielsen Gauge proportions."""
    cat = random.choices(PLATFORM_CATEGORIES, weights=PLATFORM_WEIGHTS)[0]
    service = "Linear"
    if cat == "Streaming":
        service = random.choices(STREAMING_SERVICES, weights=STREAMING_WEIGHTS)[0]
    return cat, service


# ── Thematic adjustment (Vault 2028 projections) ──────────────────────────────

def _thematic_adjustment(text: str, cr: float) -> float:
    if any(kw in text for kw in PERSONAL_GROWTH_KWS):
        return round(min(cr * 1.35, 1.00), 4)
    if any(kw in text for kw in FAMILY_DAUGHTER_KWS):
        return round(cr * 0.50, 4)
    return cr


# ── Target-age inference (mirrors tools.py._infer_target_age) ─────────────────

def _infer_target_age(item: dict) -> str:
    text = (
        str(item.get("listed_in", "")) + " " +
        str(item.get("description", "")) + " " +
        str(item.get("title", ""))
    ).lower()
    _y = {"teen", "high school", "college", "young adult", "animated",
          "coming of age", "school", "gen z", "tiktok", "gaming", "k-pop"}
    _s = {"classic", "period drama", "historical", "aristocrat", "dynasty",
          "vintage", "golden age", "silver", "heritage", "antique", "estate"}
    _m = {"documentary", "biography", "biopic", "thriller", "crime",
          "political", "financial", "corporate", "investigative"}
    y  = sum(1 for s in _y if s in text)
    sr = sum(1 for s in _s if s in text)
    md = sum(1 for s in _m if s in text)
    if y  >= 2: return "18-24"
    if sr >= 2: return "50+"
    if md >= 2: return "35-49"
    return "25-34"


# ── Gaussian prime-time helpers ───────────────────────────────────────────────

def _prime_weight(hour: float) -> float:
    return math.exp(-0.5 * ((hour - PRIME_CENTER) / PRIME_SIGMA) ** 2)


def _weighted_hour() -> int:
    hours   = list(range(6, 24))
    weights = [_prime_weight(h + 0.5) for h in hours]
    return random.choices(hours, weights=weights)[0]


# ── Demographic score generation ──────────────────────────────────────────────

def _demo_scores(target_age: str) -> dict:
    base = {
        "Gen_Alpha":  random.uniform(0.35, 0.55),
        "Gen_Z":      random.uniform(0.50, 0.70),
        "Millennial": random.uniform(0.55, 0.75),
        "Gen_X":      random.uniform(0.45, 0.70),
    }
    boosts = {
        "18-24": ("Gen_Z",      0.82, 1.00),
        "25-34": ("Millennial", 0.85, 1.00),
        "35-49": ("Gen_X",      0.82, 1.00),
        "50+":   ("Gen_X",      0.78, 0.95),
    }
    if target_age in boosts:
        key, lo, hi = boosts[target_age]
        base[key] = random.uniform(lo, hi)
    return {k: round(v, 4) for k, v in base.items()}


# ── Tier / completion rate sampling ──────────────────────────────────────────

def _pick_tier() -> tuple:
    roll = random.random()
    for name, cum_w, lo, hi in _TIERS:
        if roll < cum_w:
            return name, lo, hi
    return _TIERS[-1][0], _TIERS[-1][2], _TIERS[-1][3]


def _completion_rate(cr_lo: float, cr_hi: float, hour: int) -> float:
    base     = random.uniform(cr_lo, cr_hi)
    pt_boost = _prime_weight(hour + 0.5) * 0.025
    cr       = base + pt_boost
    return round(min(max(cr, 0.05), 1.00), 4)


# ── Viewership Pulse ─────────────────────────────────────────────────────────

_NIELSEN_STREAMING_SHARE = 0.475   # Nielsen The Gauge streaming baseline

def generate_viewership_pulse(base_dir: str = ".") -> dict:
    """Generate a 24-hour viewership pulse with primary and secondary audience lines.

    Primary line   — Gaussian curve centred at 20:30 (prime-time spike).
    Secondary line — Broader, flatter general audience curve.
    Both are anchored to the Nielsen 47.5% streaming share as a baseline multiplier.
    """
    hours = list(range(24))

    # Primary: tight Gaussian prime-time spike (σ = 1.5h, peak at 20.5)
    primary = []
    for h in hours:
        raw = math.exp(-0.5 * ((h + 0.5 - PRIME_CENTER) / PRIME_SIGMA) ** 2)
        # Scale to a plausible viewer index (0–100) with Nielsen streaming baseline
        value = round(raw * 100 * _NIELSEN_STREAMING_SHARE + random.uniform(-1.5, 1.5), 2)
        primary.append(max(0.0, value))

    # Secondary: broader, flatter general audience curve (σ = 3.5h, peak at 20.0)
    secondary = []
    for h in hours:
        raw = math.exp(-0.5 * ((h + 0.5 - 20.0) / 3.5) ** 2)
        # Flatter ceiling at 60% of primary peak; all-day floor from 10% baseline
        value = round(raw * 60 * _NIELSEN_STREAMING_SHARE + 10.0 + random.uniform(-1.0, 1.0), 2)
        secondary.append(max(0.0, value))

    pulse = {
        "generated_at":    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "nielsen_baseline": _NIELSEN_STREAMING_SHARE,
        "note":            "Primary = prime-time Gaussian; Secondary = general audience envelope",
        "hours":           hours,
        "primary":         primary,
        "secondary":       secondary,
    }

    out_path = os.path.join(base_dir, "data", "pulse_history.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(pulse, f, indent=2)

    return pulse


# ── Dashboard Snapshot ────────────────────────────────────────────────────────

def generate_dashboard_snapshot(records: list[dict], base_dir: str = ".") -> dict:
    """Derive Nielsen-aligned dashboard KPIs from the generated engagement records.

    Base values are anchored to the provided Nielsen targets, then adjusted upward
    for records where the Vault 2028 thematic surge (+35% Personal Growth) applies.
    """
    total = len(records)

    # Broadcast + Cable share = 21% + 20% = 41% of records
    broadcast_cable = [
        r for r in records
        if r.get("platform_category") in ("Broadcast", "Cable")
    ]

    # Personal Growth thematic surge records
    personal_growth = [
        r for r in records
        if any(kw in f"{r.get('title', '')}".lower() for kw in
               {"growth", "empowerment", "career", "self-discovery", "independence"})
    ]
    pg_boost = 1.35 if personal_growth else 1.0

    # C3 Rating — derived from Broadcast/Cable records, boosted by thematic surge
    base_c3 = 4.2
    c3_rating = round(base_c3 * pg_boost, 2) if personal_growth else base_c3

    # Share % — anchored to top-tier distributor share (Disney ~12.8%)
    base_share = 12.8
    share_pct = round(base_share * (1 + 0.05 * (len(personal_growth) / max(total, 1))), 2)

    # HH Impressions — scaled from Broadcast/Cable record count
    bc_ratio = len(broadcast_cable) / max(total, 1)
    hh_impressions = int(round(5_100_000 * (bc_ratio / 0.41) * pg_boost))

    # Avg Frequency — base 3.4, slight lift for high-Gold completion
    gold_records = [r for r in records if r.get("completion_rate", 0) >= 0.85]
    gold_ratio = len(gold_records) / max(total, 1)
    avg_frequency = round(3.4 + gold_ratio * 0.5, 2)

    snapshot = {
        "generated_at":       datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "source":             "Nielsen The Gauge + Vault 2028 Thematic Projections",
        "c3_rating":          c3_rating,
        "share_pct":          share_pct,
        "hh_impressions":     hh_impressions,
        "avg_frequency":      avg_frequency,
        "thematic_surge_applied": bool(personal_growth),
        "personal_growth_records": len(personal_growth),
        "broadcast_cable_records": len(broadcast_cable),
        "gold_tier_records":  len(gold_records),
        "total_records":      total,
    }

    out_path = os.path.join(base_dir, "data", "dashboard_snapshot.json")
    with open(out_path, "w") as f:
        json.dump(snapshot, f, indent=2)

    return snapshot


# ── Main generator ────────────────────────────────────────────────────────────

def _load(path: str):
    with open(path) as f:
        return json.load(f)


def _save(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _save_parquet(json_path: str, records: list[dict]) -> str:
    try:
        import pandas as pd
        parquet_path = json_path.replace(".json", ".parquet")
        pd.DataFrame(records).to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception as exc:
        print(f"  ⚠  Parquet write skipped: {exc}")
        return ""


def generate_logs(catalog_path: str, n: int = NUM_RECORDS) -> list[dict]:
    catalog = _load(catalog_path)
    for item in catalog:
        if not item.get("target_age") or item["target_age"] == "N/A":
            item["target_age"] = _infer_target_age(item)

    show_map = {item["show_id"]: item for item in catalog}
    show_ids = list(show_map.keys())

    records: list[dict] = []
    for i in range(n):
        show_id    = random.choice(show_ids)
        item       = show_map[show_id]
        target_age = item["target_age"]

        platform_cat, platform_service = _pick_platform()
        market, dma, density_tier      = random.choice(MARKETS)
        demo                           = random.choice(DEMOGRAPHICS)
        hour                           = _weighted_hour()
        _, lo, hi                      = _pick_tier()
        cr                             = _completion_rate(lo, hi, hour)

        full_text = f"{item.get('title', '')} {item.get('description', '')}".lower()

        # Vault 2028 thematic projection
        cr = _thematic_adjustment(full_text, cr)

        # Tubi Issa Rae strategic boost
        if platform_service == "Tubi" and any(tag in full_text for tag in _TUBI_BOOST_TAGS):
            cr = round(min(cr * 1.20, 1.00), 4)

        day_offset = random.randint(0, 13)
        minute     = random.randint(0, 59)
        ts = (
            datetime(
                BASE_DATE.year, BASE_DATE.month, BASE_DATE.day,
                hour, minute, 0, tzinfo=timezone.utc,
            ) + timedelta(days=day_offset)
        )

        records.append({
            "session_id":              2000 + i,
            "show_id":                 show_id,
            "title":                   item.get("title", show_id),
            "platform_category":       platform_cat,
            "platform_service":        platform_service,
            "season_number":           random.randint(1, 5),
            "viewership_share_source": "Nielsen The Gauge (v2026)",
            "target_age":              target_age,
            "primary_demographic":     demo,
            "completion_rate":         cr,
            "demographic_scores":      _demo_scores(target_age),
            "cultural_signals":        random.sample(CULTURAL_SIGNALS, k=random.randint(1, 2)),
            "metadata_boosters":       ["Director Influence", "IP Origin"] if platform_service == "Netflix" else [],
            "timestamp":               ts.isoformat(),
            "dma":                     dma,
            "market":                  market,
            "density_tier":            density_tier,
            "is_prime_time":           16 <= hour < 22,
        })

    return records


def main() -> None:
    base     = os.path.dirname(__file__)
    cat_path = os.path.join(base, "data", "catalog.json")
    out_path = os.path.join(base, "data", "engagement_logs.json")

    print("══════════════════════════════════════════════════════")
    print("  Vault Intelligence — Engagement Logs Generator")
    print("══════════════════════════════════════════════════════\n")

    records = generate_logs(cat_path)
    _save(out_path, records)
    pq = _save_parquet(out_path, records)
    if pq:
        print(f"  ✓ engagement_logs.parquet → {pq}")

    pulse = generate_viewership_pulse(base_dir=base)
    print(f"  ✓ pulse_history.json  (primary peak = {max(pulse['primary']):.1f}, "
          f"secondary peak = {max(pulse['secondary']):.1f})")

    snap = generate_dashboard_snapshot(records, base_dir=base)
    print(f"  ✓ dashboard_snapshot.json  "
          f"C3={snap['c3_rating']}  Share={snap['share_pct']}%  "
          f"HH={snap['hh_impressions']:,}  AvgFreq={snap['avg_frequency']}")

    total      = len(records)
    occasional = sum(1 for r in records if r["completion_rate"] < 0.60)
    silver     = sum(1 for r in records if 0.60 <= r["completion_rate"] < 0.85)
    gold       = sum(1 for r in records if r["completion_rate"] >= 0.85)
    prime      = sum(1 for r in records if r["is_prime_time"])
    streaming  = sum(1 for r in records if r["platform_category"] == "Streaming")
    tubi_boost = sum(
        1 for r in records
        if r["platform_service"] == "Tubi" and r["completion_rate"] > 0.60
    )
    z_ss_drop  = [
        r for r in records
        if r["target_age"] == "18-24" and r["primary_demographic"] == "Silver_Stylists"
    ]

    print(f"  ✓ {total:,} records → {out_path}")
    print(f"\n── Quad Distribution ──────────────────────────────────")
    print(f"  Occasional (<0.60):    {occasional:>5,}  ({occasional/total*100:5.1f}%)")
    print(f"  Silver    (0.60–0.84): {silver:>5,}  ({silver/total*100:5.1f}%)")
    print(f"  Gold      (0.85–1.00): {gold:>5,}  ({gold/total*100:5.1f}%)")
    print(f"\n── Nielsen Platform Split ─────────────────────────────")
    print(f"  Streaming sessions:    {streaming:>5,}  ({streaming/total*100:5.1f}%)")
    print(f"\n── Prime Time (16:00–22:00) ───────────────────────────")
    print(f"  Prime sessions:        {prime:>5,}  ({prime/total*100:5.1f}%)")
    print(f"\n── Tubi Issa Rae Boost ────────────────────────────────")
    print(f"  Tubi high-completion:  {tubi_boost:>5,}")
    print(f"\n── Gen Z→Silver Friction ──────────────────────────────")
    if z_ss_drop:
        avg_cr = sum(r["completion_rate"] for r in z_ss_drop) / len(z_ss_drop)
        print(f"  18-24 content / Silver_Stylists: {len(z_ss_drop)} sessions, avg_cr={avg_cr:.3f}")
    print(f"\n── Sample Record ──────────────────────────────────────")
    import json as _j
    print(_j.dumps(records[0], indent=2))
    print("\nThat is all.")


if __name__ == "__main__":
    main()
