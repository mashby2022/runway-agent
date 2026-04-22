"""
generate_friction_data.py — Strategic Friction Layer data builder.

Reads data/engagement_logs.json and data/catalog.json, applies the
Demographic Affinity Map to compute per-show friction scores, and writes
enriched output to data/friction_data.json.

Run standalone:
    python generate_friction_data.py

No GPU required — uses pandas throughout.
"""

import json
import os

# ── Constants (duplicated from tools.py to keep this script standalone) ─────

DEMOGRAPHIC_AFFINITY_MAP: dict[str, dict[str, float]] = {
    "18-24": {
        "Gen_Alpha":       1.3,
        "Gen_Z":           1.4,
        "Millennial":      0.9,
        "Gen_X":           0.7,
        "Silver_Stylists": 0.6,
    },
    "25-34": {
        "Gen_Alpha":       0.8,
        "Gen_Z":           1.1,
        "Millennial":      1.4,
        "Gen_X":           0.9,
        "Silver_Stylists": 0.7,
    },
    "35-49": {
        "Gen_Alpha":       0.5,
        "Gen_Z":           0.7,
        "Millennial":      1.1,
        "Gen_X":           1.4,
        "Silver_Stylists": 0.9,
    },
    "50+": {
        "Gen_Alpha":       0.4,
        "Gen_Z":           0.6,
        "Millennial":      0.8,
        "Gen_X":           1.1,
        "Silver_Stylists": 1.4,
    },
}

_GENERATIONS = ("Gen_Alpha", "Gen_Z", "Millennial", "Gen_X", "Silver_Stylists")


# ── Friction calculation ─────────────────────────────────────────────────────

def compute_friction(target_age: str) -> dict:
    """Return friction_index and segment lists for a given target_age bracket."""
    affinities = DEMOGRAPHIC_AFFINITY_MAP.get(target_age, DEMOGRAPHIC_AFFINITY_MAP["25-34"])
    repelled = {g: m for g, m in affinities.items() if m < 0.75}
    boosted  = {g: m for g, m in affinities.items() if m > 1.15}

    friction_index = (
        round(sum(1.0 - m for m in repelled.values()) / len(affinities) * 100, 1)
        if repelled else 0.0
    )

    if friction_index >= 40:
        trade_off = (
            "You've captured the Vanguard, but you've alienated the Silver Stylists. "
            "They've gone to watch the competition's boring linear broadcast. "
            "I hope those Gen Z clicks were worth the heritage loss. That is all."
        )
    elif friction_index >= 20:
        boosted_names  = ", ".join(boosted.keys())
        repelled_names = ", ".join(repelled.keys())
        trade_off = (
            f"A strategic trade-off: strong {boosted_names} acquisition at the cost "
            f"of {repelled_names} retention. Acceptable — if intentional."
        )
    else:
        trade_off = "Broad generational appeal. No significant friction detected."

    return {
        "friction_index":     friction_index,
        "repelled_segments":  repelled,
        "boosted_segments":   boosted,
        "affinity_map":       affinities,
        "trade_off_note":     trade_off,
    }


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load(path: str):
    with open(path) as f:
        return json.load(f)


def _save(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading source data …")
    logs    = _load("data/engagement_logs.json")
    catalog = _load("data/catalog.json")

    catalog_map = {item["show_id"]: item for item in catalog}

    # ── Per-show friction summary ────────────────────────────────────────────
    show_ids = sorted({r["show_id"] for r in logs})
    print(f"Computing friction for {len(show_ids)} unique shows …")

    friction_records = []
    for sid in show_ids:
        meta       = catalog_map.get(sid, {})
        target_age = meta.get("target_age", "25-34")
        friction   = compute_friction(target_age)

        show_logs = [r for r in logs if r["show_id"] == sid]
        avg_completion = (
            sum(r.get("completion_rate", 0) for r in show_logs) / len(show_logs)
            if show_logs else 0.0
        )

        # Weighted retention: multiply raw completion by each generation's multiplier
        weighted: dict[str, float] = {}
        for gen in _GENERATIONS:
            mult = friction["affinity_map"].get(gen, 1.0)
            weighted[gen] = round(avg_completion * mult, 4)

        friction_records.append({
            "show_id":              sid,
            "title":                meta.get("title", sid),
            "target_age":           target_age,
            "friction_index":       friction["friction_index"],
            "repelled_segments":    list(friction["repelled_segments"].keys()),
            "boosted_segments":     list(friction["boosted_segments"].keys()),
            "affinity_map":         friction["affinity_map"],
            "weighted_retention":   weighted,
            "avg_completion_rate":  round(avg_completion, 4),
            "trade_off_note":       friction["trade_off_note"],
            "log_count":            len(show_logs),
        })

    # Sort by friction_index descending (most polarising first)
    friction_records.sort(key=lambda x: x["friction_index"], reverse=True)

    # ── Aggregate affinity matrix across all shows ───────────────────────────
    print("Building aggregate affinity matrix …")
    affinity_matrix: dict[str, dict[str, float]] = {g: {} for g in _GENERATIONS}
    for rec in friction_records:
        for gen in _GENERATIONS:
            ta = rec["target_age"]
            if ta not in affinity_matrix[gen]:
                affinity_matrix[gen][ta] = rec["affinity_map"].get(gen, 1.0)

    # ── Per-generation average weighted retention across all shows ───────────
    gen_avg_retention: dict[str, float] = {}
    for gen in _GENERATIONS:
        vals = [r["weighted_retention"].get(gen, 0) for r in friction_records]
        gen_avg_retention[gen] = round(sum(vals) / len(vals), 4) if vals else 0.0

    output = {
        "_meta": {
            "generated_by":    "generate_friction_data.py",
            "source_compute":  "pandas (CPU — friction data generation does not require GPU)",
            "total_shows":     len(friction_records),
            "total_log_rows":  len(logs),
            "note": (
                "friction_index scores are computed offline for data generation only. "
                "Live Strategic Risk scores in the NAT agent require ONLINE (GPU) mode."
            ),
        },
        "affinity_matrix":          affinity_matrix,
        "gen_avg_weighted_retention": gen_avg_retention,
        "shows":                    friction_records,
    }

    out_path = "data/friction_data.json"
    _save(out_path, output)
    print(f"Saved {len(friction_records)} friction records → {out_path}")

    # ── Summary report ───────────────────────────────────────────────────────
    high_friction   = [r for r in friction_records if r["friction_index"] >= 40]
    medium_friction = [r for r in friction_records if 20 <= r["friction_index"] < 40]
    low_friction    = [r for r in friction_records if r["friction_index"] < 20]

    print("\n── Strategic Friction Summary ──")
    print(f"  High friction   (≥40): {len(high_friction)} shows")
    print(f"  Medium friction (20–39): {len(medium_friction)} shows")
    print(f"  Low friction    (<20):  {len(low_friction)} shows")
    print("\n── Top 5 Most Polarising Shows ──")
    for r in friction_records[:5]:
        print(
            f"  [{r['friction_index']:5.1f}] {r['title']:<40} "
            f"target={r['target_age']}  repelled={r['repelled_segments']}"
        )
    print("\n── Generation Average Weighted Retention ──")
    for gen, avg in sorted(gen_avg_retention.items(), key=lambda x: -x[1]):
        print(f"  {gen:<18} {avg:.4f}")
    print("\nDone. That is all.")


if __name__ == "__main__":
    main()
