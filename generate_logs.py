"""
generate_logs.py — Engagement Logs Generator with Organic Variance

Produces data/engagement_logs.json — 5,000 viewing sessions featuring:
  - Quad distribution (Pareto): 65% Occasional / 25% Silver / 10% Gold
  - Gaussian prime-time curve centered at 20:30 (σ = 1.5 h)
  - Gen Z → Silver Stylist friction: 15–30% completion drop for 18-24 content
  - Organic demographic score variance per title target_age

Run standalone:
    python generate_logs.py
"""

import json
import math
import os
import random
from datetime import date, datetime, timezone, timedelta

random.seed(42)

# ── Constants ─────────────────────────────────────────────────────────────────

BASE_DATE   = date(2026, 4, 14)   # Monday — two weeks of data ending current week
NUM_RECORDS = 5_000

MARKETS = [
    ("New York",      "New York (DMA 1)",      "Urban Core"),
    ("Los Angeles",   "Los Angeles (DMA 2)",   "Urban Core"),
    ("Chicago",       "Chicago (DMA 3)",       "Urban Core"),
    ("Dallas",        "Dallas (DMA 4)",        "Affluent Suburban"),
    ("San Francisco", "San Francisco (DMA 5)", "Urban Core"),
    ("Philadelphia",  "Philadelphia (DMA 6)",  "Affluent Suburban"),
    ("Atlanta",       "Atlanta (DMA 7)",       "Exurban"),
    ("Paris",         "Paris (DMA 8)",         "Urban Core"),
    ("London",        "London (DMA 9)",        "Urban Core"),
    ("Milan",         "Milan (DMA 10)",        "Affluent Suburban"),
]

DEMOGRAPHICS = [
    "Female", "Male", "LGBT+", "Gen_Z", "Millennial", "Gen_X", "Silver_Stylists",
]

# Pareto quad tiers: (name, cumulative_weight, cr_lo, cr_hi)
_TIERS = [
    ("Occasional", 0.65, 0.15, 0.59),
    ("Silver",     0.90, 0.60, 0.84),
    ("Gold",       1.00, 0.85, 1.00),
]

PRIME_CENTER = 20.5   # 20:30
PRIME_SIGMA  = 1.5


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
    """Sample an hour weighted by the Gaussian prime-time curve (06–23)."""
    hours   = list(range(6, 24))
    weights = [_prime_weight(h + 0.5) for h in hours]
    return random.choices(hours, weights=weights)[0]


# ── Demographic score generation ──────────────────────────────────────────────

def _demo_scores(target_age: str) -> dict:
    base = {
        "Gen_Alpha":       random.uniform(0.35, 0.55),
        "Gen_Z":           random.uniform(0.50, 0.70),
        "Millennial":      random.uniform(0.55, 0.75),
        "Gen_X":           random.uniform(0.45, 0.70),
        "Silver_Stylists": random.uniform(0.40, 0.62),
    }
    boosts = {
        "18-24": ("Gen_Z",           0.82, 1.00),
        "25-34": ("Millennial",      0.85, 1.00),
        "35-49": ("Gen_X",           0.82, 1.00),
        "50+":   ("Silver_Stylists", 0.84, 1.00),
    }
    if target_age in boosts:
        key, lo, hi = boosts[target_age]
        base[key] = random.uniform(lo, hi)
    # Gen Z → Silver Stylist friction
    if target_age == "18-24":
        drop = random.uniform(0.15, 0.30)
        base["Silver_Stylists"] = round(base["Silver_Stylists"] * (1.0 - drop), 4)
    return {k: round(v, 4) for k, v in base.items()}


# ── Tier / completion rate sampling ──────────────────────────────────────────

def _pick_tier() -> tuple:
    roll = random.random()
    for name, cum_w, lo, hi in _TIERS:
        if roll < cum_w:
            return name, lo, hi
    return _TIERS[-1][0], _TIERS[-1][2], _TIERS[-1][3]


def _completion_rate(
    cr_lo: float, cr_hi: float,
    hour: int,
    target_age: str,
    demo: str,
) -> float:
    base     = random.uniform(cr_lo, cr_hi)
    pt_boost = _prime_weight(hour + 0.5) * 0.025  # up to +0.025 at peak — preserves tier shape
    cr       = base + pt_boost
    # Gen Z → Silver Stylist friction
    if target_age == "18-24" and demo == "Silver_Stylists":
        drop = random.uniform(0.15, 0.30)
        cr   = cr * (1.0 - drop)
    return round(min(max(cr, 0.05), 1.00), 4)


# ── Main generator ────────────────────────────────────────────────────────────

def _load(path: str):
    with open(path) as f:
        return json.load(f)


def _save(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _save_parquet(json_path: str, records: list[dict]) -> str:
    """Write records to .parquet alongside the .json for DuckDB queries."""
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

    show_ids = [item["show_id"] for item in catalog]
    show_map  = {item["show_id"]: item for item in catalog}

    records: list[dict] = []
    for i in range(n):
        show_id    = random.choice(show_ids)
        item       = show_map[show_id]
        target_age = item["target_age"]

        market, dma, density_tier = random.choice(MARKETS)
        demo        = random.choice(DEMOGRAPHICS)
        hour        = _weighted_hour()
        is_prime    = 16 <= hour < 22
        _, lo, hi   = _pick_tier()
        cr          = _completion_rate(lo, hi, hour, target_age, demo)
        scores      = _demo_scores(target_age)

        day_offset = random.randint(0, 13)
        minute     = random.randint(0, 59)
        ts = (
            datetime(
                BASE_DATE.year, BASE_DATE.month, BASE_DATE.day,
                hour, minute, 0, tzinfo=timezone.utc,
            ) + timedelta(days=day_offset)
        )

        session_shows = random.sample(show_ids, min(2, len(show_ids)))

        records.append({
            "session_id":          1000 + i,
            "show_id":             show_id,
            "title":               item.get("title", show_id),
            "target_age":          target_age,
            "primary_demographic": demo,
            "completion_rate":     cr,
            "demographic_scores":  scores,
            "timestamp":           ts.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "dma":                 dma,
            "market":              market,
            "density_tier":        density_tier,
            "is_prime_time":       is_prime,
            "session_shows":       session_shows,
        })

    return records


def main() -> None:
    base     = os.path.dirname(__file__)
    cat_path = os.path.join(base, "data", "catalog.json")
    out_path = os.path.join(base, "data", "engagement_logs.json")

    print("══════════════════════════════════════════════════════")
    print("  Runway Inclusive — Engagement Logs Generator")
    print("══════════════════════════════════════════════════════\n")

    records = generate_logs(cat_path)
    _save(out_path, records)
    pq = _save_parquet(out_path, records)
    if pq:
        print(f"  ✓ engagement_logs.parquet → {pq}")

    total      = len(records)
    occasional = sum(1 for r in records if r["completion_rate"] < 0.60)
    silver     = sum(1 for r in records if 0.60 <= r["completion_rate"] < 0.85)
    gold       = sum(1 for r in records if r["completion_rate"] >= 0.85)
    prime      = sum(1 for r in records if r["is_prime_time"])
    z_ss_drop  = [
        r for r in records
        if r["target_age"] == "18-24" and r["primary_demographic"] == "Silver_Stylists"
    ]

    print(f"  ✓ {total:,} records → {out_path}")
    print(f"\n── Quad Distribution ──────────────────────────────────")
    print(f"  Occasional (<0.60):    {occasional:>5,}  ({occasional/total*100:5.1f}%)")
    print(f"  Silver    (0.60–0.84): {silver:>5,}  ({silver/total*100:5.1f}%)")
    print(f"  Gold      (0.85–1.00): {gold:>5,}  ({gold/total*100:5.1f}%)")
    print(f"\n── Prime Time (16:00–22:00) ───────────────────────────")
    print(f"  Prime sessions:        {prime:>5,}  ({prime/total*100:5.1f}%)")
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
