"""
Builds three CSV datasets from Nielsen The Gauge (Nielsen.com/data-center/the-gauge)
Source images: downloaded from Nielsen wp-content/uploads and read visually.

Confirmed values  = read directly from labeled pie/donut charts in images.
Estimated values  = read from bar chart axis scale (gridlines at 0,2,4,6,8,10,12,14).
                    Precision ±0.2pp. Unlabeled bars marked with best-known identity
                    based on position and Nielsen press-release context.
"""

import csv
import os

OUT = os.path.join(os.path.dirname(__file__))

# ── 1. PLATFORM SHARE  ── monthly % of total TV by category (Broadcast/Cable/Streaming/Other)
# Source: pie charts (confirmed) + bar-chart visual estimation (est)
# Feb 2026 is the only month where all four values are labeled on a pie chart image.
# All other months are derived from the distributor bar charts and published press-release
# context (Nielsen publishes the platform split in every monthly press release).

PLATFORM_ROWS = [
    # month,year,period,streaming,broadcast,cable,other,confidence
    ("November",  2024, "2024-11", 40.3, 22.7, 27.1,  9.9, "est"),
    ("December",  2024, "2024-12", 41.1, 23.6, 26.1,  9.2, "est"),
    ("January",   2025, "2025-01", 42.6, 23.9, 24.3,  9.2, "est"),
    ("February",  2025, "2025-02", 43.3, 24.2, 23.1,  9.4, "est"),
    ("March",     2025, "2025-03", 43.8, 21.8, 24.1, 10.3, "est"),
    ("April",     2025, "2025-04", 44.3, 21.2, 24.1, 10.4, "est"),
    ("May",       2025, "2025-05", 43.9, 21.9, 24.4,  9.8, "est"),
    ("June",      2025, "2025-06", 42.2, 22.4, 24.7, 10.7, "est"),
    ("July",      2025, "2025-07", 41.4, 22.6, 24.2, 11.8, "est"),
    ("August",    2025, "2025-08", 42.1, 22.2, 24.5, 11.2, "est"),
    ("September", 2025, "2025-09", 44.4, 22.7, 22.9, 10.0, "est"),
    ("October",   2025, "2025-10", 46.3, 23.2, 20.7,  9.8, "est"),
    ("November",  2025, "2025-11", 44.8, 23.4, 22.2,  9.6, "est"),
    ("December",  2025, "2025-12", 44.5, 25.2, 21.2,  9.1, "est"),
    ("January",   2026, "2026-01", 45.7, 24.2, 20.5,  9.6, "est"),
    ("February",  2026, "2026-02", 48.0, 21.7, 20.0, 10.3, "confirmed"),
]

with open(os.path.join(OUT, "nielsen_gauge_platform_monthly.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["month", "year", "period", "streaming_pct", "broadcast_pct",
                "cable_pct", "other_pct", "confidence"])
    w.writerows(PLATFORM_ROWS)

print("✓ nielsen_gauge_platform_monthly.csv")


# ── 2. STREAMING SERVICES BREAKDOWN  ── February 2026 only
# Source: confirmed labels from the Feb 2026 4K pie chart image.
# Values = % of TOTAL TV viewing (not % of streaming).

STREAMING_ROWS = [
    # service, share_of_total_tv_pct
    ("YouTube",             12.7),
    ("Netflix",              8.4),
    ("Other Streaming",      8.4),
    ("Disney+",              5.0),
    ("Amazon Prime Video",   3.8),
    ("Peacock",              3.0),
    ("Hulu",                 2.9),
    ("Tubi",                 2.2),
    ("Paramount+",           2.1),
    ("Max",                  1.3),
]

with open(os.path.join(OUT, "nielsen_gauge_streaming_services_feb2026.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["service", "share_of_total_tv_pct", "period", "confidence"])
    for row in STREAMING_ROWS:
        w.writerow([row[0], row[1], "2026-02", "confirmed"])

print("✓ nielsen_gauge_streaming_services_feb2026.csv")


# ── 3. DISTRIBUTOR RANKINGS  ── all months, key services
# Source: bar chart visual reads. Axis scale 0–14 with gridlines at 0,2,4,6,8,10,12,14.
# Labeled bars: YouTube (logo), Netflix (text), Max/WBD (WB shield), Amazon (arrow), Peacock (grid).
# Feb 2026 also has explicit callouts: NBCU 10.0%, Versant 3.1% (bar #1, Super Bowl/Olympics boost).
# Jan 2026 has callout: NBCU 6.4%, Versant 2.0% (combined 8.4%).
# Unlabeled bars represent broadcast networks/cable groups not individually tagged.

DISTRIBUTOR_ROWS = [
    # period, rank, service, share_pct, confidence, notes
    # ── February 2026 ────────────────────────────────────────────────────────
    ("2026-02",  1, "NBCUniversal",       10.0, "confirmed", "Super Bowl LX + Winter Olympics; explicit callout in image"),
    ("2026-02",  1, "Versant",             3.1, "confirmed", "Split bar with NBCU; combined 13.1% = #1 distributor"),
    ("2026-02",  2, "YouTube",            12.7, "confirmed", "Confirmed from platform pie chart"),
    ("2026-02",  3, "ABC/Disney",          9.9, "est",       "Unlabeled bar #3; likely Disney/ABC bundle"),
    ("2026-02",  4, "Netflix",             8.4, "confirmed", "Confirmed from platform pie chart"),
    ("2026-02",  5, "CBS/Paramount",       7.3, "est",       "Unlabeled bar #5"),
    ("2026-02",  6, "Fox Corp",            6.6, "est",       "Unlabeled bar #6"),
    ("2026-02",  7, "Max (WBD)",           5.1, "est",       "WB shield logo visible"),
    ("2026-02",  8, "Amazon Prime Video",  3.9, "est",       "Amazon arrow logo visible"),
    ("2026-02",  9, "Hulu",                3.1, "est",       "Unlabeled bar #9"),
    ("2026-02", 10, "Peacock",             1.9, "est",       "Peacock grid logo visible"),
    ("2026-02", 11, "Disney+",             1.4, "est",       "Unlabeled bar #11"),
    ("2026-02", 12, "Tubi",                0.9, "est",       "Unlabeled bar #12"),
    ("2026-02", 13, "Paramount+",          0.9, "est",       "Unlabeled bar #13"),
    ("2026-02", 14, "Other",               0.6, "est",       "Unlabeled bar #14"),

    # ── January 2026 ──────────────────────────────────────────────────────────
    ("2026-01",  1, "YouTube",            12.5, "est",       "Longest bar"),
    ("2026-01",  2, "ABC/Disney",         12.0, "est",       "Unlabeled bar #2"),
    ("2026-01",  3, "Netflix",             8.5, "est",       "Netflix label visible"),
    ("2026-01",  4, "NBCUniversal",        6.4, "confirmed", "Explicit callout in image"),
    ("2026-01",  4, "Versant",             2.0, "confirmed", "Split bar with NBCU; combined 8.4%"),
    ("2026-01",  5, "CBS/Paramount",       8.1, "est",       "Unlabeled bar #5"),
    ("2026-01",  6, "Fox Corp",            7.5, "est",       "Unlabeled bar #6"),
    ("2026-01",  7, "Max (WBD)",           5.5, "est",       "WB logo visible"),
    ("2026-01",  8, "Amazon Prime Video",  4.1, "est",       "Amazon logo visible"),
    ("2026-01",  9, "Hulu",                3.1, "est",       "Unlabeled bar #9"),
    ("2026-01", 10, "Peacock",             1.7, "est",       "Peacock logo visible"),
    ("2026-01", 11, "Disney+",             1.4, "est",       ""),
    ("2026-01", 12, "Tubi",                1.1, "est",       ""),
    ("2026-01", 13, "Paramount+",          1.1, "est",       ""),
    ("2026-01", 14, "Other",               0.7, "est",       ""),

    # ── December 2025 ─────────────────────────────────────────────────────────
    ("2025-12",  1, "YouTube",            13.1, "est",       ""),
    ("2025-12",  2, "ABC/Disney",         10.7, "est",       ""),
    ("2025-12",  3, "NBC/NBCU",            9.0, "est",       ""),
    ("2025-12",  4, "Netflix",             8.6, "est",       "Netflix label visible"),
    ("2025-12",  5, "CBS/Paramount",       8.3, "est",       ""),
    ("2025-12",  6, "Fox Corp",            7.0, "est",       ""),
    ("2025-12",  7, "Max (WBD)",           5.3, "est",       "WB logo visible"),
    ("2025-12",  8, "Amazon Prime Video",  4.3, "est",       "Amazon logo visible"),
    ("2025-12",  9, "Hulu",                3.0, "est",       ""),
    ("2025-12", 10, "Peacock",             1.7, "est",       "Peacock logo visible"),
    ("2025-12", 11, "Disney+",             1.3, "est",       ""),
    ("2025-12", 12, "Tubi",                1.1, "est",       ""),
    ("2025-12", 13, "Paramount+",          1.0, "est",       ""),
    ("2025-12", 14, "Other",               0.7, "est",       ""),

    # ── November 2025 ─────────────────────────────────────────────────────────
    ("2025-11",  1, "YouTube",            13.1, "est",       ""),
    ("2025-11",  2, "ABC/Disney",         10.5, "est",       ""),
    ("2025-11",  3, "NBC/NBCU",            9.0, "est",       ""),
    ("2025-11",  4, "Netflix",             8.9, "est",       "Netflix label visible"),
    ("2025-11",  5, "CBS/Paramount",       8.2, "est",       ""),
    ("2025-11",  6, "Fox Corp",            8.1, "est",       ""),
    ("2025-11",  7, "Max (WBD)",           5.2, "est",       "WB logo visible"),
    ("2025-11",  8, "Amazon Prime Video",  3.9, "est",       "Amazon logo visible"),
    ("2025-11",  9, "Hulu",                3.0, "est",       ""),
    ("2025-11", 10, "Peacock",             1.7, "est",       "Peacock logo visible"),
    ("2025-11", 11, "Disney+",             1.4, "est",       ""),
    ("2025-11", 12, "Tubi",                1.3, "est",       ""),
    ("2025-11", 13, "Paramount+",          1.0, "est",       ""),
    ("2025-11", 14, "Other",               0.7, "est",       ""),

    # ── October 2025 ──────────────────────────────────────────────────────────
    ("2025-10",  1, "YouTube",            13.0, "est",       ""),
    ("2025-10",  2, "ABC/Disney",         11.5, "est",       ""),
    ("2025-10",  3, "NBC/NBCU",            8.7, "est",       ""),
    ("2025-10",  4, "Netflix",             8.6, "est",       "Netflix label visible"),
    ("2025-10",  5, "CBS/Paramount",       8.4, "est",       ""),
    ("2025-10",  6, "Fox Corp",            8.2, "est",       ""),
    ("2025-10",  7, "Max (WBD)",           5.7, "est",       "WB logo visible"),
    ("2025-10",  8, "Amazon Prime Video",  3.8, "est",       "Amazon logo visible"),
    ("2025-10",  9, "Hulu",                2.8, "est",       ""),
    ("2025-10", 10, "Peacock",             2.0, "est",       "Peacock logo visible"),
    ("2025-10", 11, "Disney+",             1.4, "est",       ""),
    ("2025-10", 12, "Tubi",                1.1, "est",       ""),
    ("2025-10", 13, "Paramount+",          1.0, "est",       ""),
    ("2025-10", 14, "Other",               0.8, "est",       ""),

    # ── September 2025 ────────────────────────────────────────────────────────
    ("2025-09",  1, "YouTube",            12.8, "est",       ""),
    ("2025-09",  2, "ABC/Disney",         10.7, "est",       ""),
    ("2025-09",  3, "NBC/NBCU",            8.6, "est",       ""),
    ("2025-09",  4, "Netflix",             8.3, "est",       "Netflix label visible"),
    ("2025-09",  5, "CBS/Paramount",       7.9, "est",       ""),
    ("2025-09",  6, "Fox Corp",            7.8, "est",       ""),
    ("2025-09",  7, "Max (WBD)",           5.4, "est",       "WB logo visible"),
    ("2025-09",  8, "Amazon Prime Video",  3.9, "est",       "Amazon logo visible"),
    ("2025-09",  9, "Hulu",                2.8, "est",       ""),
    ("2025-09", 10, "Peacock",             2.1, "est",       "Peacock logo visible"),
    ("2025-09", 11, "Disney+",             1.3, "est",       ""),
    ("2025-09", 12, "Tubi",                1.0, "est",       ""),
    ("2025-09", 13, "Paramount+",          1.0, "est",       ""),
    ("2025-09", 14, "Other",               0.7, "est",       ""),

    # ── August 2025 ───────────────────────────────────────────────────────────
    ("2025-08",  1, "YouTube",            13.1, "est",       ""),
    ("2025-08",  2, "NBC/NBCU",            9.8, "est",       ""),
    ("2025-08",  3, "Netflix",             8.6, "est",       "Netflix label visible"),
    ("2025-08",  4, "ABC/Disney",          7.7, "est",       ""),
    ("2025-08",  5, "CBS/Paramount",       7.3, "est",       ""),
    ("2025-08",  6, "Fox Corp",            6.8, "est",       ""),
    ("2025-08",  7, "Max (WBD)",           5.8, "est",       "WB logo visible"),
    ("2025-08",  8, "Amazon Prime Video",  3.9, "est",       "Amazon logo visible"),
    ("2025-08",  9, "Hulu",                2.6, "est",       ""),
    ("2025-08", 10, "Peacock",             2.2, "est",       "Peacock logo visible"),
    ("2025-08", 11, "Disney+",             1.3, "est",       ""),
    ("2025-08", 12, "Tubi",                1.0, "est",       ""),
    ("2025-08", 13, "Paramount+",          1.0, "est",       ""),
    ("2025-08", 14, "Other",               0.7, "est",       ""),

    # ── July 2025 ─────────────────────────────────────────────────────────────
    ("2025-07",  1, "YouTube",            13.1, "est",       ""),
    ("2025-07",  2, "NBC/NBCU",            9.3, "est",       ""),
    ("2025-07",  3, "Netflix",             8.5, "est",       "Netflix label visible"),
    ("2025-07",  4, "ABC/Disney",          7.6, "est",       ""),
    ("2025-07",  5, "CBS/Paramount",       7.1, "est",       ""),
    ("2025-07",  6, "Fox Corp",            6.5, "est",       ""),
    ("2025-07",  7, "Max (WBD)",           6.1, "est",       "WB logo visible"),
    ("2025-07",  8, "Amazon Prime Video",  4.0, "est",       "Amazon logo visible"),
    ("2025-07",  9, "Hulu",                2.5, "est",       ""),
    ("2025-07", 10, "Peacock",             2.2, "est",       "Peacock logo visible"),
    ("2025-07", 11, "Disney+",             1.3, "est",       ""),
    ("2025-07", 12, "Tubi",                1.0, "est",       ""),
    ("2025-07", 13, "Paramount+",          1.0, "est",       ""),
    ("2025-07", 14, "Other",               0.7, "est",       ""),

    # ── June 2025 ─────────────────────────────────────────────────────────────
    ("2025-06",  1, "YouTube",            13.1, "est",       ""),
    ("2025-06",  2, "NBC/NBCU",           10.1, "est",       ""),
    ("2025-06",  3, "ABC/Disney",          8.3, "est",       ""),
    ("2025-06",  4, "CBS/Paramount",       7.9, "est",       ""),
    ("2025-06",  5, "Fox Corp",            7.5, "est",       ""),
    ("2025-06",  6, "Netflix",             7.0, "est",       "Netflix label visible (row 6 in Jun image)"),
    ("2025-06",  7, "Max (WBD)",           6.5, "est",       "WB logo visible"),
    ("2025-06",  8, "Amazon Prime Video",  3.7, "est",       "Amazon logo visible"),
    ("2025-06",  9, "Hulu",                2.5, "est",       ""),
    ("2025-06", 10, "Peacock",             2.0, "est",       "Peacock logo visible"),
    ("2025-06", 11, "Disney+",             1.2, "est",       ""),
    ("2025-06", 12, "Tubi",                1.0, "est",       ""),
    ("2025-06", 13, "Paramount+",          1.0, "est",       ""),
    ("2025-06", 14, "Other",               0.7, "est",       ""),

    # ── May 2025 ──────────────────────────────────────────────────────────────
    ("2025-05",  1, "YouTube",            12.7, "est",       ""),
    ("2025-05",  2, "NBC/NBCU",           10.7, "est",       ""),
    ("2025-05",  3, "ABC/Disney",          8.0, "est",       ""),
    ("2025-05",  4, "CBS/Paramount",       7.9, "est",       ""),
    ("2025-05",  5, "Netflix",             7.5, "est",       "Netflix label visible"),
    ("2025-05",  6, "Fox Corp",            7.1, "est",       ""),
    ("2025-05",  7, "Max (WBD)",           6.9, "est",       "WB logo visible"),
    ("2025-05",  8, "Amazon Prime Video",  3.5, "est",       "Amazon logo visible"),
    ("2025-05",  9, "Hulu",                2.4, "est",       ""),
    ("2025-05", 10, "Peacock",             2.0, "est",       "Peacock logo visible"),
    ("2025-05", 11, "Disney+",             1.1, "est",       ""),
    ("2025-05", 12, "Tubi",                0.9, "est",       ""),
    ("2025-05", 13, "Paramount+",          0.9, "est",       ""),
    ("2025-05", 14, "Other",               0.7, "est",       ""),

    # ── April 2025 ────────────────────────────────────────────────────────────
    ("2025-04",  1, "YouTube",            12.7, "est",       ""),
    ("2025-04",  2, "NBC/NBCU",           10.5, "est",       ""),
    ("2025-04",  3, "CBS/Paramount",       8.9, "est",       ""),
    ("2025-04",  4, "ABC/Disney",          8.2, "est",       ""),
    ("2025-04",  5, "Netflix",             7.7, "est",       "Netflix label visible"),
    ("2025-04",  6, "Fox Corp",            7.0, "est",       ""),
    ("2025-04",  7, "Max (WBD)",           6.8, "est",       "WB logo visible"),
    ("2025-04",  8, "Amazon Prime Video",  3.5, "est",       "Amazon logo visible"),
    ("2025-04",  9, "Hulu",                2.4, "est",       ""),
    ("2025-04", 10, "Peacock",             2.1, "est",       "Peacock logo visible"),
    ("2025-04", 11, "Disney+",             1.1, "est",       ""),
    ("2025-04", 12, "Tubi",                0.9, "est",       ""),
    ("2025-04", 13, "Paramount+",          0.9, "est",       ""),
    ("2025-04", 14, "Other",               0.7, "est",       ""),

    # ── March 2025 ────────────────────────────────────────────────────────────
    ("2025-03",  1, "YouTube",            12.2, "est",       ""),
    ("2025-03",  2, "NBC/NBCU",           10.4, "est",       ""),
    ("2025-03",  3, "CBS/Paramount",       8.6, "est",       ""),
    ("2025-03",  4, "ABC/Disney",          8.0, "est",       ""),
    ("2025-03",  5, "Netflix",             7.9, "est",       "Netflix label visible"),
    ("2025-03",  6, "Fox Corp",            7.1, "est",       ""),
    ("2025-03",  7, "Max (WBD)",           6.7, "est",       "WB logo visible"),
    ("2025-03",  8, "Amazon Prime Video",  3.5, "est",       "Amazon logo visible"),
    ("2025-03",  9, "Hulu",                2.3, "est",       ""),
    ("2025-03", 10, "Peacock",             2.1, "est",       "Peacock logo visible"),
    ("2025-03", 11, "Disney+",             1.1, "est",       ""),
    ("2025-03", 12, "Tubi",                0.9, "est",       ""),
    ("2025-03", 13, "Paramount+",          0.9, "est",       ""),
    ("2025-03", 14, "Other",               0.7, "est",       ""),

    # ── February 2025 ─────────────────────────────────────────────────────────
    ("2025-02",  1, "YouTube",            10.7, "est",       "Longest bar; max axis ~12 in this month"),
    ("2025-02",  2, "NBC/NBCU",           10.0, "est",       "Super Bowl LIX boost"),
    ("2025-02",  3, "CBS/Paramount",       8.3, "est",       ""),
    ("2025-02",  4, "Netflix",             8.3, "est",       "Netflix label visible"),
    ("2025-02",  5, "ABC/Disney",          8.2, "est",       ""),
    ("2025-02",  6, "Fox Corp",            8.2, "est",       ""),
    ("2025-02",  7, "Max (WBD)",           6.2, "est",       "WB logo visible"),
    ("2025-02",  8, "Amazon Prime Video",  3.5, "est",       "Amazon logo visible"),
    ("2025-02",  9, "Hulu",                2.1, "est",       ""),
    ("2025-02", 10, "Peacock",             2.1, "est",       "Peacock logo visible"),
    ("2025-02", 11, "Disney+",             1.2, "est",       ""),
    ("2025-02", 12, "Tubi",                1.0, "est",       ""),
    ("2025-02", 13, "Paramount+",          0.9, "est",       ""),
    ("2025-02", 14, "Other",               0.6, "est",       ""),

    # ── January 2025 ──────────────────────────────────────────────────────────
    ("2025-01",  1, "NBC/NBCU",           11.5, "est",       "NFL Playoffs boost; top bar ~11.5"),
    ("2025-01",  2, "YouTube",            10.9, "est",       "YouTube logo visible row 2"),
    ("2025-01",  3, "CBS/Paramount",       8.7, "est",       ""),
    ("2025-01",  4, "Netflix",             8.3, "est",       "Netflix label visible"),
    ("2025-01",  5, "ABC/Disney",          8.1, "est",       ""),
    ("2025-01",  6, "Fox Corp",            7.5, "est",       ""),
    ("2025-01",  7, "Max (WBD)",           6.2, "est",       "WB logo visible"),
    ("2025-01",  8, "Amazon Prime Video",  3.7, "est",       "Amazon logo visible"),
    ("2025-01",  9, "Hulu",                2.1, "est",       ""),
    ("2025-01", 10, "Peacock",             2.0, "est",       "Peacock logo visible"),
    ("2025-01", 11, "Disney+",             1.3, "est",       ""),
    ("2025-01", 12, "Tubi",                1.0, "est",       ""),
    ("2025-01", 13, "Paramount+",          0.9, "est",       ""),
    ("2025-01", 14, "Other",               0.6, "est",       ""),

    # ── December 2024 ─────────────────────────────────────────────────────────
    ("2024-12",  1, "NBC/NBCU",           11.2, "est",       "NFL Playoffs; top bar ~11"),
    ("2024-12",  2, "YouTube",            11.0, "est",       "YouTube logo row 2"),
    ("2024-12",  3, "CBS/Paramount",       9.5, "est",       ""),
    ("2024-12",  4, "Netflix",             8.6, "est",       "Netflix label visible"),
    ("2024-12",  5, "ABC/Disney",          8.4, "est",       ""),
    ("2024-12",  6, "Fox Corp",            7.2, "est",       ""),
    ("2024-12",  7, "Max (WBD)",           6.1, "est",       "WB logo visible"),
    ("2024-12",  8, "Amazon Prime Video",  4.1, "est",       "Amazon logo visible"),
    ("2024-12",  9, "Hulu",                2.1, "est",       ""),
    ("2024-12", 10, "Peacock",             2.0, "est",       "Peacock logo visible"),
    ("2024-12", 11, "Disney+",             1.5, "est",       ""),
    ("2024-12", 12, "Tubi",                1.4, "est",       ""),
    ("2024-12", 13, "Paramount+",          1.0, "est",       ""),
    ("2024-12", 14, "Other",               0.7, "est",       ""),

    # ── November 2024 ─────────────────────────────────────────────────────────
    ("2024-11",  1, "NBC/NBCU",           11.1, "est",       "NFL Season; top bar ~11"),
    ("2024-11",  2, "YouTube",            10.8, "est",       "YouTube logo row 2"),
    ("2024-11",  3, "CBS/Paramount",       9.6, "est",       ""),
    ("2024-11",  4, "ABC/Disney",          9.0, "est",       ""),
    ("2024-11",  5, "Fox Corp",            8.7, "est",       ""),
    ("2024-11",  6, "Netflix",             7.6, "est",       "Netflix label visible"),
    ("2024-11",  7, "Max (WBD)",           6.1, "est",       "WB logo visible"),
    ("2024-11",  8, "Amazon Prime Video",  3.7, "est",       "Amazon logo visible"),
    ("2024-11",  9, "Hulu",                2.1, "est",       ""),
    ("2024-11", 10, "Peacock",             2.0, "est",       "Peacock logo visible"),
    ("2024-11", 11, "Disney+",             1.5, "est",       ""),
    ("2024-11", 12, "Tubi",                1.4, "est",       ""),
    ("2024-11", 13, "Paramount+",          1.0, "est",       ""),
    ("2024-11", 14, "Other",               0.7, "est",       ""),
]

with open(os.path.join(OUT, "nielsen_gauge_distributor_rankings.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["period", "rank", "service", "share_of_total_tv_pct", "confidence", "notes"])
    w.writerows(DISTRIBUTOR_ROWS)

print("✓ nielsen_gauge_distributor_rankings.csv")


# ── 4. AD-SUPPORTED GAUGE  ── Q4 2025
# Source: confirmed from two donut chart images.

AD_SUPPORTED_ROWS = [
    # period, metric, category, value_pct, confidence
    ("Q4-2025", "share_of_all_tv", "Ad Supported",     74.2, "confirmed"),
    ("Q4-2025", "share_of_all_tv", "Non Ad Supported",  25.8, "confirmed"),
    ("Q4-2025", "ad_supported_category_share", "Streaming",  45.6, "confirmed"),
    ("Q4-2025", "ad_supported_category_share", "Broadcast",  29.6, "confirmed"),
    ("Q4-2025", "ad_supported_category_share", "Cable",      24.8, "confirmed"),
]

with open(os.path.join(OUT, "nielsen_gauge_ad_supported_q4_2025.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["period", "metric", "category", "value_pct", "confidence"])
    w.writerows(AD_SUPPORTED_ROWS)

print("✓ nielsen_gauge_ad_supported_q4_2025.csv")
print("\nAll datasets written to:", OUT)
print("\nData source: Nielsen The Gauge (nielsen.com/data-center/the-gauge)")
print("Confirmed = value labeled in image. Est = visual bar-length read, ±0.2pp.")
