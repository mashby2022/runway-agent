"""
generate_weekly_insights.py — Daily Strategic Themes builder.

Fabricates the 7-day editorial calendar for the Runway Inclusive demo.
Creates data/weekly_schedule.json with:
  - daily_themes: strategic metadata for each day of the week
  - weekly_plan:  empty list (populated by the generate_weekly_plan NAT tool)

Run standalone:
    python generate_weekly_insights.py

Re-running is safe — it overwrites the daily_themes block only, leaving
any existing weekly_plan intact.
"""

import json
import os
from datetime import date, timedelta

WEEKLY_SCHEDULE_PATH = os.path.join(os.path.dirname(__file__), "data", "weekly_schedule.json")

# Week anchored to the current project date
WEEK_START = date(2026, 4, 20)   # Monday
SUNDAY     = WEEK_START - timedelta(days=1)   # April 19

DAILY_THEMES = {
    "Sunday": {
        "theme":       "Heritage Weekend",
        "tribe":       "Heritage Couture",
        "peak_demo":   "Silver_Stylists",
        "secondary_demo": "Female",
        "location_bias":  "Affluent Suburban",
        "age_focus":   "50+",
        "markets":     ["Dallas (DMA 4)", "Atlanta (DMA 6)", "New York (DMA 1)"],
        "genre_keywords": [
            "drama", "classic", "period", "costume", "historical", "literary",
        ],
        "description_keywords": [
            "elegant", "timeless", "heritage", "tradition", "legacy",
            "couture", "dynasty", "aristocrat",
        ],
        "ds_rationale": (
            "Heritage Weekend opens Sunday with appointment viewing for our most loyal "
            "Silver Stylist audience. Affluent Suburban completion rates peak Sunday "
            "16:00–22:00 — Heritage Couture titles command the premium ad inventory. "
            "The stylistic arc of the week concludes here, giving the audience the "
            "editorial gravitas they came for. That is all."
        ),
    },

    "Monday": {
        "theme":       "Minimalist Monday",
        "tribe":       "Minimalist",
        "peak_demo":   "Female",
        "secondary_demo": "LGBT+",
        "location_bias":  "Urban Core",
        "age_focus":   "25-34",
        "markets":     ["New York (DMA 1)", "Chicago (DMA 3)"],
        "genre_keywords": [
            "drama", "independent", "art", "documentary", "design",
        ],
        "description_keywords": [
            "clean", "restrained", "modern", "minimal", "architecture",
            "contemporary", "urban", "spare",
        ],
        "ds_rationale": (
            "Monday mornings demand editorial restraint. Our Urban Core Female audience "
            "in New York and Chicago arrives ready for the week — Minimalist titles "
            "set the tone without demanding too much. Completion rates for clean, "
            "high-concept content spike 28% in the 08:00–12:00 NYC window on Mondays. "
            "This is not background television. This is the new black."
        ),
    },

    "Tuesday": {
        "theme":       "Romantic Tuesday",
        "tribe":       "Romantic Feminine",
        "peak_demo":   "Female",
        "secondary_demo": "Millennial",
        "location_bias":  "Exurban",
        "age_focus":   "25-34",
        "markets":     ["Atlanta (DMA 6)", "Dallas (DMA 4)"],
        "genre_keywords": [
            "romance", "drama", "romantic", "love", "relationship",
        ],
        "description_keywords": [
            "love", "passion", "heart", "feminine", "romantic",
            "sensual", "desire", "longing",
        ],
        "ds_rationale": (
            "Tuesday's Romantic Feminine programming anchors our Exurban Millennial "
            "Female audience. Mid-week dwell time for this segment peaks Tuesday "
            "evening — emotionally resonant cinema keeps completion rates above 0.70. "
            "Atlanta and Dallas Exurban markets lead this tribe's engagement curve."
        ),
    },

    "Wednesday": {
        "theme":       "Avant-Garde Wednesday",
        "tribe":       "Avant-Garde",
        "peak_demo":   "Gen_Z",
        "secondary_demo": "LGBT+",
        "location_bias":  "Urban Core",
        "age_focus":   "18-24",
        "markets":     ["New York (DMA 1)", "Los Angeles (DMA 2)"],
        "genre_keywords": [
            "experimental", "art", "indie", "avant-garde", "surreal", "conceptual",
        ],
        "description_keywords": [
            "bold", "experimental", "boundary", "subvert", "provocative",
            "artistic", "visionary", "transgressive",
        ],
        "ds_rationale": (
            "Wednesday is our mid-week Vanguard spike. Gen Z and LGBT+ audiences in "
            "NYC and LA engage most intensely on Wednesday evenings — this is where "
            "the algorithm leans into the uncomfortable and the extraordinary. "
            "Avant-Garde titles here consistently deliver our highest 18-24 completion "
            "rates of the week. The audience is dressed and ready. That is all."
        ),
    },

    "Thursday": {
        "theme":       "Global Couture Thursday",
        "tribe":       "Heritage Couture",
        "peak_demo":   "Female",
        "secondary_demo": "Silver_Stylists",
        "location_bias":  "Affluent Suburban",
        "age_focus":   "35-49",
        "markets":     ["New York (DMA 1)", "Dallas (DMA 4)"],
        "genre_keywords": [
            "fashion", "couture", "documentary", "biography", "luxury",
        ],
        "description_keywords": [
            "designer", "fashion", "couture", "luxury", "house",
            "atelier", "collection", "masterpiece",
        ],
        "ds_rationale": (
            "Thursday bridges Avant-Garde Wednesday into Heritage Weekend. "
            "Heritage Couture documentaries and biopics prime the 35-49 Female and "
            "Silver Stylist audiences for the weekend's premium programming. "
            "The Condé Nast Accelerated Intelligence Layer identifies Thursday "
            "16:00–20:00 as the highest-value interstitial window for fashion brand "
            "integrations across all Affluent Suburban markets."
        ),
    },

    "Friday": {
        "theme":       "Street & Youth Friday",
        "tribe":       "Street & Youth",
        "peak_demo":   "Gen_Z",
        "secondary_demo": "Millennial",
        "location_bias":  "Urban Core",
        "age_focus":   "18-24",
        "markets":     ["Los Angeles (DMA 2)", "Chicago (DMA 3)"],
        "genre_keywords": [
            "youth", "street", "hip-hop", "music", "subculture",
        ],
        "description_keywords": [
            "street", "youth", "urban", "culture", "rebel",
            "generation", "identity", "movement",
        ],
        "ds_rationale": (
            "Friday pre-weekend energy belongs to our youngest audience. "
            "Street & Youth titles drive the highest Gen Z engagement on Friday "
            "evenings — LA and Chicago Urban Core markets outperform all others "
            "in this slot. An unexpected accessory that steals the whole look."
        ),
    },

    "Saturday": {
        "theme":       "Heritage Weekend",
        "tribe":       "Heritage Couture",
        "peak_demo":   "Silver_Stylists",
        "secondary_demo": "Female",
        "location_bias":  "Affluent Suburban",
        "age_focus":   "50+",
        "markets":     ["Dallas (DMA 4)", "Atlanta (DMA 6)", "New York (DMA 1)"],
        "genre_keywords": [
            "drama", "classic", "period", "costume", "historical", "literary",
        ],
        "description_keywords": [
            "elegant", "timeless", "heritage", "tradition", "legacy",
            "couture", "dynasty", "aristocrat",
        ],
        "ds_rationale": (
            "Heritage Weekend opens Saturday with the season's premium offering "
            "for our Silver Stylist audience. Affluent Suburban completion rates "
            "peak Saturday 14:00–20:00 — Heritage Couture titles anchor the "
            "highest CPM ad inventory of the week. Nothing unravels here. That is all."
        ),
    },
}


def _load_existing() -> dict:
    try:
        with open(WEEKLY_SCHEDULE_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def main() -> None:
    existing = _load_existing()

    output = {
        "_meta": {
            "generated_by": "generate_weekly_insights.py",
            "version":      "1.0",
            "week_start":   WEEK_START.isoformat(),
            "sunday_date":  SUNDAY.isoformat(),
            "note": (
                "daily_themes defines the editorial strategy per day. "
                "weekly_plan (the actual slot grid) is populated by the "
                "generate_weekly_plan NAT tool."
            ),
        },
        "daily_themes": DAILY_THEMES,
        # Preserve any existing plan; generate_weekly_plan tool will overwrite
        "weekly_plan": existing.get("weekly_plan", []),
    }

    os.makedirs(os.path.dirname(WEEKLY_SCHEDULE_PATH), exist_ok=True)
    with open(WEEKLY_SCHEDULE_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved daily_themes for {len(DAILY_THEMES)} days → {WEEKLY_SCHEDULE_PATH}")
    print("\n── Daily Theme Summary ──")
    for day, data in DAILY_THEMES.items():
        print(
            f"  {day:<12} │ {data['theme']:<28} │ "
            f"tribe={data['tribe']:<20} │ age={data['age_focus']}"
        )
    print(
        "\nRun the generate_weekly_plan NAT tool (or call tools.generate_weekly_plan()) "
        "to populate the weekly_plan slot grid. That is all."
    )


if __name__ == "__main__":
    main()
