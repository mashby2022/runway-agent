"""
FAST Channel Media Intelligence – Data Prep
Builds catalog.json, telemetry.json, schedule.json
"""

import json
import math
import random
from datetime import datetime, timedelta, timezone

import pandas as pd

DATA = "data"
TMDB_FILE = f"{DATA}/tmdb/TMDB_movie_dataset_v11.csv"
NETFLIX_FILE = f"{DATA}/netflix/netflix_titles.csv"
RATINGS_FILE = f"{DATA}/movielens/ml-25m/ratings.csv"

CATALOG_OUT = f"{DATA}/catalog.json"
TELEMETRY_OUT = f"{DATA}/telemetry.json"
SCHEDULE_OUT = f"{DATA}/schedule.json"

CHANNEL_ID = "ch_runway_01"

TARGET_CAST = {"meryl streep", "anne hathaway", "emily blunt", "stanley tucci"}
TARGET_KEYWORDS = {"fashion", "magazine", "couture"}

# Titles known to feature the target cast (cast data absent from v11 dataset)
TITLE_ALLOW = {
    "the devil wears prada",
    "the intern",
    "prada",
    "valentino",
    "september issue",
    "coco before chanel",
    "dior and i",
    "unzipped",
    "devil wears prada",
}


# ── Task 1: CMS Catalog ────────────────────────────────────────────────────────

def _kw_match(kw_str: str) -> bool:
    if not isinstance(kw_str, str):
        return False
    lower = kw_str.lower()
    return any(k in lower for k in TARGET_KEYWORDS)


def _title_match(title: str) -> bool:
    if not isinstance(title, str):
        return False
    return title.lower() in TITLE_ALLOW


def _cast_match(name_str: str) -> bool:
    """Best-effort: check if a comma-separated name string contains target actors."""
    if not isinstance(name_str, str):
        return False
    lower = name_str.lower()
    return any(a in lower for a in TARGET_CAST)


def build_catalog() -> list[dict]:
    print("Loading TMDB v11 …")
    tmdb = pd.read_csv(TMDB_FILE, low_memory=False)

    print(f"  Raw rows: {len(tmdb):,}")

    # Normalise keyword / title strings, handle NaN
    tmdb["keywords"] = tmdb["keywords"].fillna("")
    tmdb["title"] = tmdb["title"].fillna("")

    kw_mask = tmdb["keywords"].apply(_kw_match)
    title_mask = tmdb["title"].apply(_title_match)
    mask = kw_mask | title_mask

    filtered = tmdb[mask].copy()
    print(f"  After fashion/magazine/couture filter: {len(filtered):,} rows")

    # Derive release year safely
    filtered["release_year"] = pd.to_datetime(
        filtered["release_date"], errors="coerce"
    ).dt.year.fillna(0).astype(int)

    # Format duration
    filtered["duration_str"] = (
        filtered["runtime"]
        .fillna(0)
        .astype(int)
        .astype(str)
        .add(" min")
    )

    # Production country → first value
    filtered["country_str"] = (
        filtered["production_countries"]
        .fillna("")
        .apply(lambda x: x.split(",")[0].strip() if x else "")
    )

    catalog = []
    for i, (_, row) in enumerate(filtered.iterrows(), start=1):
        show_id = f"s{i:04d}"
        catalog.append({
            "show_id":      show_id,
            "type":         "Movie",
            "title":        row["title"] or "",
            "director":     "",          # not present in v11
            "cast":         "",          # not present in v11
            "country":      row["country_str"],
            "date_added":   "",
            "release_year": int(row["release_year"]),
            "rating":       "",          # not present in v11
            "duration":     row["duration_str"],
            "listed_in":    row.get("genres", "") or "",
            "description":  row.get("overview", "") or "",
        })

    print(f"  Catalog entries: {len(catalog)}")
    return catalog


# ── Task 2: Telemetry ──────────────────────────────────────────────────────────

def build_telemetry(catalog: list[dict]) -> list[dict]:
    print("Loading MovieLens ratings (50k rows) …")
    ratings = pd.read_csv(RATINGS_FILE, nrows=50_000)

    show_ids = [item["show_id"] for item in catalog]
    titles   = {item["show_id"]: item["title"] for item in catalog}

    # Find Devil Wears Prada entry for spike injection
    dwp_id = next(
        (s for s, t in titles.items() if "devil wears prada" in t.lower()),
        show_ids[0]
    )
    print(f"  Spike target: {dwp_id} ({titles[dwp_id]})")

    random.seed(42)
    demographics = ["13-17", "18-35", "25-34", "35-44", "45-54", "55+"]
    spike_demo   = "18-35"

    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    telemetry = []
    unique_users = ratings["userId"].unique()

    for idx, show_id in enumerate(show_ids):
        base_viewers = int(ratings["rating"].iloc[idx % len(ratings)] * 1_000)
        for hour in range(0, 24, 2):
            ts = today + timedelta(hours=hour)
            for demo in demographics:
                viewers = max(50, base_viewers + random.randint(-200, 200))

                # Inject massive spike for Devil Wears Prada at 16:00 in 18-35 demo
                if show_id == dwp_id and hour == 16 and demo == "18-35":
                    viewers = random.randint(48_000, 52_000)

                telemetry.append({
                    "show_id":     show_id,
                    "title":       titles[show_id],
                    "timestamp":   ts.isoformat(),
                    "channel_id":  CHANNEL_ID,
                    "demographic": demo,
                    "viewers":     viewers,
                    "avg_rating":  round(
                        float(ratings["rating"].iloc[(idx + hour) % len(ratings)]), 2
                    ),
                })

    print(f"  Telemetry records: {len(telemetry):,}")
    return telemetry


# ── Task 3: EPG Schedule ───────────────────────────────────────────────────────

def build_schedule(catalog: list[dict]) -> list[dict]:
    print("Generating 24-hour EPG …")

    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day = today + timedelta(days=1)

    # Alternate 60-min and 30-min slots
    slot_pattern = [60, 30, 60, 30, 60, 60, 30]
    schedule = []
    current_time = today
    cat_idx = 0

    while current_time < end_of_day:
        slot_mins = slot_pattern[len(schedule) % len(slot_pattern)]
        end_time  = min(current_time + timedelta(minutes=slot_mins), end_of_day)
        item      = catalog[cat_idx % len(catalog)]

        schedule.append({
            "channel_id": CHANNEL_ID,
            "show_id":    item["show_id"],
            "title":      item["title"],
            "start":      current_time.isoformat(),
            "end":        end_time.isoformat(),
            "duration_min": int((end_time - current_time).total_seconds() // 60),
        })

        current_time = end_time
        cat_idx += 1

    print(f"  EPG slots: {len(schedule)}")
    return schedule


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    catalog   = build_catalog()
    telemetry = build_telemetry(catalog)
    schedule  = build_schedule(catalog)

    with open(CATALOG_OUT, "w") as f:
        json.dump(catalog, f, indent=2)
    print(f"Saved {CATALOG_OUT}  ({len(catalog)} entries)")

    with open(TELEMETRY_OUT, "w") as f:
        json.dump(telemetry, f, indent=2)
    print(f"Saved {TELEMETRY_OUT}  ({len(telemetry):,} records)")

    with open(SCHEDULE_OUT, "w") as f:
        json.dump(schedule, f, indent=2)
    print(f"Saved {SCHEDULE_OUT}  ({len(schedule)} slots)")

    print("\nAll done.")


if __name__ == "__main__":
    main()
