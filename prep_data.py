"""
FAST Channel Media Intelligence – Data Prep
Builds catalog.json, telemetry.json, schedule.json

Catalog:    Female-led fashion/drama titles only.
Telemetry:  Female_Viewers and LGBTQ_Core_Audience segments with
            engagement spikes during high-fashion blocks.
"""

import json
import random
from datetime import datetime, timedelta, timezone

import pandas as pd

DATA = "data"
TMDB_FILE = f"{DATA}/tmdb/TMDB_movie_dataset_v11.csv"
RATINGS_FILE = f"{DATA}/movielens/ml-25m/ratings.csv"

CATALOG_OUT = f"{DATA}/catalog.json"
TELEMETRY_OUT = f"{DATA}/telemetry.json"
SCHEDULE_OUT = f"{DATA}/schedule.json"

CHANNEL_ID = "ch_runway_01"

# Female leads whose presence confirms a film belongs in this catalog.
# TMDB v11 lacks a cast column so this supplements keyword/title matching.
TARGET_CAST = {
    "meryl streep", "anne hathaway", "emily blunt",
    "reese witherspoon", "sandra bullock", "julia roberts",
    "audrey hepburn", "emma stone", "cate blanchett",
    "natalie portman", "charlize theron", "kate blanchett",
    "sarah jessica parker", "isla fisher", "renée zellweger",
    "jennifer aniston", "nicole kidman", "helen mirren",
    "viola davis", "jessica lange", "amy adams",
}

TARGET_KEYWORDS = {
    "fashion", "magazine", "couture", "haute couture",
    "runway", "supermodel", "designer", "vogue",
    "female protagonist", "independent woman",
}

# Known female-led fashion/drama titles not reliably caught by keywords.
TITLE_ALLOW = {
    "the devil wears prada",
    "the intern",
    "coco before chanel",
    "the september issue",
    "dior and i",
    "valentino: the last emperor",
    "unzipped",
    "pret-a-porter",
    "prêt-à-porter",
    "funny face",
    "breakfast at tiffany's",
    "breakfast at tiffanys",
    "legally blonde",
    "legally blonde 2",
    "confessions of a shopaholic",
    "pretty woman",
    "bridget jones's diary",
    "bridget jones: the edge of reason",
    "working girl",
    "cruella",
    "mamma mia!",
    "mamma mia",
    "mamma mia: here we go again",
    "mamma mia! here we go again",
    "julie & julia",
    "the iron lady",
    "florence foster jenkins",
    "wild",
    "miss congeniality",
    "miss congeniality 2",
    "erin brockovich",
    "nine",
    "diana vreeland: the eye has to travel",
    "the first monday in may",
    "sigrid olsen: a runway story",
}

# Titles to exclude even if they match keywords (male-led or off-brand).
TITLE_BLOCK = {
    "zoolander",
    "zoolander 2",
    "zoolander no. 2",
    "male model",
    "the house that jack built",
}

# High-fashion titles that drive LGBTQ+ engagement spikes.
HIGH_FASHION_TITLES = {
    "the devil wears prada",
    "coco before chanel",
    "dior and i",
    "valentino: the last emperor",
    "the september issue",
    "pret-a-porter",
    "prêt-à-porter",
    "diana vreeland: the eye has to travel",
    "the first monday in may",
    "funny face",
    "cruella",
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


def _title_blocked(title: str) -> bool:
    if not isinstance(title, str):
        return False
    return title.lower() in TITLE_BLOCK


def _cast_match(name_str: str) -> bool:
    if not isinstance(name_str, str):
        return False
    lower = name_str.lower()
    return any(a in lower for a in TARGET_CAST)


def build_catalog() -> list[dict]:
    print("Loading TMDB v11 …")
    tmdb = pd.read_csv(TMDB_FILE, low_memory=False)
    print(f"  Raw rows: {len(tmdb):,}")

    tmdb["keywords"] = tmdb["keywords"].fillna("")
    tmdb["title"] = tmdb["title"].fillna("")

    # Cast column may or may not be present in this version of the dataset
    has_cast = "cast" in tmdb.columns
    if has_cast:
        tmdb["cast"] = tmdb["cast"].fillna("")
        cast_mask = tmdb["cast"].apply(_cast_match)
    else:
        cast_mask = pd.Series(False, index=tmdb.index)

    kw_mask = tmdb["keywords"].apply(_kw_match)
    title_mask = tmdb["title"].apply(_title_match)
    block_mask = tmdb["title"].apply(_title_blocked)

    filtered = tmdb[((kw_mask | title_mask | cast_mask) & ~block_mask)].copy()
    print(f"  After female-led filter (blocked {block_mask.sum()} titles): {len(filtered):,} rows")

    filtered["release_year"] = (
        pd.to_datetime(filtered["release_date"], errors="coerce")
        .dt.year.fillna(0).astype(int)
    )
    filtered["duration_str"] = (
        filtered["runtime"].fillna(0).astype(int).astype(str).add(" min")
    )
    filtered["country_str"] = (
        filtered["production_countries"]
        .fillna("")
        .apply(lambda x: x.split(",")[0].strip() if x else "")
    )

    catalog = []
    for i, (_, row) in enumerate(filtered.iterrows(), start=1):
        show_id = f"s{i:04d}"
        runtime_min = int(row["runtime"]) if pd.notna(row["runtime"]) and int(row["runtime"]) > 0 else 90
        catalog.append({
            "show_id":      show_id,
            "type":         "Movie",
            "title":        row["title"] or "",
            "director":     "",
            "cast":         "",
            "country":      row["country_str"],
            "date_added":   "",
            "release_year": int(row["release_year"]),
            "rating":       "",
            "duration":     row["duration_str"],
            "runtime_min":  runtime_min,
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

    # Identify high-fashion titles by show_id for spike injection
    fashion_ids = {
        sid for sid, title in titles.items()
        if title.lower() in HIGH_FASHION_TITLES
    }
    dwp_id = next(
        (s for s, t in titles.items() if "devil wears prada" in t.lower()),
        show_ids[0]
    )
    print(f"  DWP spike target : {dwp_id} ({titles[dwp_id]})")
    print(f"  High-fashion IDs : {fashion_ids}")

    random.seed(42)
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    # Two audience segments replace the old generic age cohorts
    segments = ["Female_Viewers", "LGBTQ_Core_Audience"]

    telemetry = []
    for idx, show_id in enumerate(show_ids):
        base_viewers = int(ratings["rating"].iloc[idx % len(ratings)] * 1_000)
        is_fashion = show_id in fashion_ids
        is_dwp     = show_id == dwp_id

        for hour in range(0, 24, 2):
            ts = today + timedelta(hours=hour)
            prime_hour = 16 <= hour < 22  # 4 PM – 10 PM

            for seg in segments:
                viewers = max(50, base_viewers + random.randint(-200, 200))

                if seg == "Female_Viewers":
                    # Female viewership runs ~40 % higher baseline for all titles
                    viewers = int(viewers * 1.4)
                    # Extra 20 % spike during prime-time fashion blocks
                    if is_fashion and prime_hour:
                        viewers = int(viewers * 1.20)

                elif seg == "LGBTQ_Core_Audience":
                    # LGBTQ+ audience is a smaller but highly engaged segment
                    viewers = max(50, int(viewers * 0.55))
                    # 15 % spike on any high-fashion title during prime time
                    if is_fashion and prime_hour:
                        viewers = int(viewers * 1.15)
                    # Extra uplift specifically for The Devil Wears Prada at 16:00
                    if is_dwp and hour == 16:
                        viewers = int(viewers * 1.15)

                telemetry.append({
                    "show_id":     show_id,
                    "title":       titles[show_id],
                    "timestamp":   ts.isoformat(),
                    "channel_id":  CHANNEL_ID,
                    "viewer_type": seg,
                    "viewers":     viewers,
                    "avg_rating":  round(
                        float(ratings["rating"].iloc[(idx + hour) % len(ratings)]), 2
                    ),
                })

    print(f"  Telemetry records: {len(telemetry):,}")
    return telemetry


# ── Task 3: EPG Schedule ───────────────────────────────────────────────────────

def _block_duration(runtime_min: int) -> int:
    """Round runtime UP to the nearest 30-minute block."""
    return ((runtime_min + 29) // 30) * 30


def build_schedule(catalog: list[dict]) -> list[dict]:
    print("Generating 24-hour block-scheduled EPG …")

    EPG_START = datetime(2026, 4, 20, 0, 0, 0, tzinfo=timezone.utc)
    end_of_day = EPG_START + timedelta(days=1)

    schedule = []
    current_time = EPG_START
    cat_idx = 0

    while current_time < end_of_day:
        item = catalog[cat_idx % len(catalog)]
        actual_runtime = item["runtime_min"]
        block_mins = _block_duration(actual_runtime)

        end_time = current_time + timedelta(minutes=block_mins)
        if end_time > end_of_day:
            break

        schedule.append({
            "channel_id":        CHANNEL_ID,
            "show_id":           item["show_id"],
            "title":             item["title"],
            "start":             current_time.isoformat(),
            "end":               end_time.isoformat(),
            "duration_min":      block_mins,
            "content_runtime_min": actual_runtime,
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

    # Verification: print block durations for key titles
    targets = {"the devil wears prada", "wonder woman"}
    for slot in schedule:
        if slot["title"].lower() in targets:
            print(
                f"  VERIFY  {slot['title']}: "
                f"actual={slot['content_runtime_min']} min  "
                f"block={slot['duration_min']} min  "
                f"({slot['start']} → {slot['end']})"
            )

    print("\nAll done.")


if __name__ == "__main__":
    main()
