"""
Runway Inclusive – Media Intelligence Command Center
Streamlit + Plotly dashboard · Gen Z / High-Fashion Chic aesthetic
"""

import json
import requests
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timezone
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────
NAT_SERVER = "http://localhost:8080/generate"
CHANNEL_ID = "ch_runway_01"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Couture Classics | Media Intel Suite",
    page_icon="🎀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS: Gen Z / High-Fashion Chic ────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500&display=swap');

  /* ── Base ── */
  [data-testid="stAppViewContainer"] {
    background-color: #FDFDFD;
    background-image:
      radial-gradient(circle at 15% 10%, rgba(255,182,235,0.18) 0%, transparent 45%),
      radial-gradient(circle at 85% 90%, rgba(200,180,255,0.15) 0%, transparent 45%);
  }
  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #2d1f2d;
  }
  .block-container { padding-top: 0.75rem; }
  p, li, span { color: #3a2a3a; }

  /* ── Sidebar: Millennial Pink → Soft Lavender gradient ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffe4f0 0%, #f0e6ff 60%, #e8f0ff 100%) !important;
    border-right: 1px solid rgba(255,105,180,0.15);
  }
  section[data-testid="stSidebar"] > div { padding-top: 1.25rem; }
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] span,
  [data-testid="stSidebar"] label { color: #5a2d5a; }

  /* ── Header ── */
  .main-header {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    color: #b5006e;
    text-align: center;
    padding: 0.75rem 0 0.2rem;
    background: linear-gradient(90deg, #FF69B4, #DA70D6, #9370DB);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .sub-header {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 0.35em;
    text-transform: uppercase;
    color: #b08cb0;
    text-align: center;
    margin-bottom: 1.5rem;
  }

  /* ── Section labels ── */
  .section-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.32em;
    text-transform: uppercase;
    color: #DA70D6;
    margin-bottom: 0.6rem;
  }

  /* ── Glassmorphism cards ── */
  .glass-card {
    background: rgba(255, 255, 255, 0.8);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-radius: 15px;
    border: 1px solid rgba(255, 182, 235, 0.35);
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.08);
    padding: 1.1rem 1.25rem;
    margin-bottom: 0.75rem;
  }
  .glass-card-title {
    font-family: 'Playfair Display', serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: #7b2f7b;
    margin-bottom: 0.4rem;
  }
  .glass-card-desc {
    font-size: 0.8rem;
    color: #7a5a7a;
    line-height: 1.5;
  }

  /* ── Metric containers ── */
  [data-testid="metric-container"] {
    background: rgba(255, 255, 255, 0.8) !important;
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,105,180,0.2);
    border-radius: 15px;
    box-shadow: 0 4px 30px rgba(0,0,0,0.07);
    padding: 0.75rem 1rem !important;
  }
  [data-testid="metric-container"] label { color: #9b59a0 !important; font-size: 0.72rem !important; letter-spacing: 0.1em; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #c2185b !important; font-family: 'Playfair Display', serif; }

  /* ── EPG cards ── */
  .epg-now {
    background: linear-gradient(135deg, #FF69B4 0%, #DA70D6 100%);
    color: #fff;
    padding: 12px 16px;
    border-radius: 15px;
    font-weight: 600;
    font-size: 0.83rem;
    margin-bottom: 10px;
    line-height: 1.6;
    box-shadow: 0 4px 20px rgba(255,105,180,0.35);
  }
  .epg-slot {
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(8px);
    color: #5a2d5a;
    padding: 10px 13px;
    border-radius: 12px;
    font-size: 0.8rem;
    margin-bottom: 6px;
    border-left: 3px solid #e8b4e8;
    line-height: 1.45;
    box-shadow: 0 2px 10px rgba(180,100,200,0.08);
    transition: border-left-color 0.2s;
  }
  .epg-slot:hover { border-left-color: #FF69B4; }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, #FF69B4 0%, #DA70D6 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 50px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    padding: 0.45rem 1.5rem !important;
    box-shadow: 0 4px 15px rgba(255,105,180,0.35) !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
  }
  .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(255,105,180,0.5) !important;
  }

  /* ── Dividers ── */
  hr { border-color: rgba(255,105,180,0.2); }

  /* ── Expander ── */
  [data-testid="stExpander"] {
    background: rgba(255,255,255,0.8);
    border-radius: 15px;
    border: 1px solid rgba(255,182,235,0.3);
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
  }

  /* ── Chat input ── */
  [data-testid="stChatInput"] textarea {
    background: rgba(255,255,255,0.9) !important;
    border: 1.5px solid rgba(255,105,180,0.4) !important;
    border-radius: 50px !important;
    color: #3a2a3a !important;
    font-family: 'DM Sans', sans-serif !important;
  }
  [data-testid="stChatInput"] textarea:focus {
    border-color: #FF69B4 !important;
    box-shadow: 0 0 0 3px rgba(255,105,180,0.12) !important;
  }

  /* ── Miranda bubble ── */
  .miranda-bubble {
    background: rgba(255,255,255,0.82);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(218,112,214,0.3);
    border-radius: 20px 20px 4px 20px;
    padding: 18px 22px;
    margin: 6px 0 6px 52px;
    box-shadow: 0 4px 30px rgba(180,80,200,0.10);
    line-height: 1.7;
    color: #3a2a3a;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.93rem;
  }
  .miranda-name {
    font-family: 'Playfair Display', serif;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: #c2185b;
    margin-bottom: 10px;
  }
  .miranda-ts {
    font-size: 0.62rem;
    color: #c4a0c4;
    margin-top: 12px;
    text-align: right;
    letter-spacing: 0.08em;
  }

  /* ── User bubble ── */
  .user-bubble-wrap {
    display: flex;
    justify-content: flex-end;
    margin: 6px 0;
  }
  .user-bubble {
    background: linear-gradient(135deg, #FF69B4 0%, #9b59b6 100%);
    border-radius: 20px 20px 20px 4px;
    padding: 12px 18px;
    color: #fff;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    max-width: 72%;
    box-shadow: 0 4px 15px rgba(255,105,180,0.3);
    line-height: 1.55;
  }

  /* ── General text inputs / selects ── */
  [data-testid="stTextInput"] input,
  [data-testid="stSelectbox"] { border-radius: 10px !important; }

  /* ── Info / warning boxes ── */
  [data-testid="stAlert"] { border-radius: 12px !important; }

  /* ── Caption text ── */
  .stCaption, caption { color: #a080a0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Data loaders ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_data():
    schedule    = json.loads(Path("data/schedule.json").read_text())
    catalog_raw = json.loads(Path("data/catalog.json").read_text())
    catalog     = {item["show_id"]: item for item in catalog_raw}
    telemetry   = json.loads(Path("data/telemetry.json").read_text())
    return schedule, catalog, telemetry


# ── Time helpers ───────────────────────────────────────────────────────────────
def simulated_now() -> datetime:
    real = datetime.now(timezone.utc)
    return real.replace(year=2026, month=4, day=20)


def find_current_slot(schedule: list, now: datetime) -> dict | None:
    for slot in schedule:
        start = datetime.fromisoformat(slot["start"])
        end   = datetime.fromisoformat(slot["end"])
        if start <= now < end:
            return slot
    return None


def find_next_slots(schedule: list, now: datetime, n: int = 5) -> list:
    return [s for s in schedule if datetime.fromisoformat(s["start"]) >= now][:n]


# ── Telemetry helper ───────────────────────────────────────────────────────────
def get_segment_viewers(telemetry: list, show_id: str, now: datetime) -> dict[str, int]:
    bucket = (now.hour // 2) * 2
    totals: dict[str, int] = {}
    for rec in telemetry:
        if rec.get("show_id") != show_id:
            continue
        try:
            rec_dt = datetime.fromisoformat(rec["timestamp"])
        except ValueError:
            continue
        if rec_dt.hour == bucket:
            seg = rec.get("viewer_type", "Unknown")
            totals[seg] = totals.get(seg, 0) + rec.get("viewers", 0)
    return totals


# ── Pastel-vivid palette ───────────────────────────────────────────────────────
SEG_PALETTE = {
    "Female_Viewers":      {"color": "#FF6EB4", "label": "💗 Female Viewers"},
    "LGBTQ_Core_Audience": {"color": "#A855F7", "label": "🌈 LGBTQ+ Core Audience"},
}
SPARKLINE_PALETTE = {
    "Female_Viewers":      {"color": "#FF6EB4", "label": "💗 Female Viewers"},
    "LGBTQ_Core_Audience": {"color": "#22D3EE", "label": "🌈 LGBTQ+ Core Audience"},
}


# ── Load data ──────────────────────────────────────────────────────────────────
try:
    schedule, catalog, telemetry = load_data()
    data_ok = True
except Exception as exc:
    st.error(f"Failed to load data files: {exc}")
    data_ok = False
    schedule, catalog, telemetry = [], {}, []

now = simulated_now()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="main-header">🎀 Couture Classics: The Media Intel Suite 🎀</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="sub-header">'
    f'ch_runway_01 &nbsp;·&nbsp; Runway Inclusive &nbsp;·&nbsp; '
    f'{now.strftime("%A, %B %d, %Y")} &nbsp;·&nbsp; 🕒 {now.strftime("%H:%M UTC")}'
    f'</div>',
    unsafe_allow_html=True,
)

if not data_ok:
    st.stop()

current_slot = find_current_slot(schedule, now)
next_slots   = find_next_slots(schedule, now, n=5)

# ── Sidebar EPG ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-label">📺 EPG · ch_runway_01</div>', unsafe_allow_html=True)

    if current_slot:
        start_fmt = datetime.fromisoformat(current_slot["start"]).strftime("%H:%M")
        end_fmt   = datetime.fromisoformat(current_slot["end"]).strftime("%H:%M")
        crt       = current_slot.get("content_runtime_min", "—")
        blk       = current_slot.get("duration_min", "—")
        st.markdown(
            f'<div class="epg-now">'
            f'✨ NOW PLAYING<br>'
            f'<span style="font-family:\'Playfair Display\',serif;font-size:1.05rem">'
            f'🎬 {current_slot["title"]}</span><br>'
            f'<span style="font-weight:400;opacity:0.9">'
            f'🕒 {start_fmt} – {end_fmt} &nbsp;·&nbsp; {blk} min block / {crt} min film'
            f'</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown("*No program currently airing.*")

    if next_slots:
        st.markdown('<div class="section-label" style="margin-top:0.9rem">🎞 Up Next</div>', unsafe_allow_html=True)
        for slot in next_slots:
            t = datetime.fromisoformat(slot["start"]).strftime("%H:%M")
            st.markdown(
                f'<div class="epg-slot">'
                f'🕒 <strong>{t}</strong> &nbsp;·&nbsp; 🎬 {slot["title"]}<br>'
                f'<span style="color:#b08cb0;font-size:0.74rem">{slot["duration_min"]} min block</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.caption(f"🕒 Simulated clock: {now.strftime('%H:%M UTC')}")
    if st.button("✦  Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ── Main: Telemetry + Show Info ────────────────────────────────────────────────
col_chart, col_info = st.columns([3, 1], gap="large")

with col_chart:
    st.markdown('<div class="section-label">💜 Live Viewership by Audience Segment</div>', unsafe_allow_html=True)

    if current_slot:
        seg_data = get_segment_viewers(telemetry, current_slot["show_id"], now)
        if seg_data:
            labels  = list(seg_data.keys())
            values  = list(seg_data.values())
            display = [SEG_PALETTE.get(l, {}).get("label", l) for l in labels]
            colors  = [SEG_PALETTE.get(l, {}).get("color", "#FF69B4") for l in labels]

            fig = go.Figure(go.Bar(
                x=display,
                y=values,
                marker=dict(
                    color=colors,
                    line_width=0,
                    opacity=0.88,
                ),
                text=[f"{v:,}" for v in values],
                textposition="outside",
                textfont=dict(color="#6d3880", size=14, family="DM Sans, sans-serif"),
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.5)",
                font=dict(color="#7b4f8a", family="DM Sans, sans-serif"),
                xaxis=dict(showgrid=False, tickfont=dict(size=13, color="#7b4f8a")),
                yaxis=dict(
                    showgrid=True,
                    gridcolor="rgba(218,112,214,0.15)",
                    tickformat=",",
                    tickfont=dict(size=11, color="#a080a0"),
                    zeroline=False,
                ),
                margin=dict(t=30, b=10, l=10, r=10),
                height=300,
                bargap=0.4,
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📈 Hourly trend — all segments"):
                hours_data: dict[str, dict[int, int]] = {}
                for rec in telemetry:
                    if rec.get("show_id") != current_slot["show_id"]:
                        continue
                    try:
                        h = datetime.fromisoformat(rec["timestamp"]).hour
                    except ValueError:
                        continue
                    seg = rec.get("viewer_type", "Unknown")
                    hours_data.setdefault(seg, {})[h] = (
                        hours_data.get(seg, {}).get(h, 0) + rec.get("viewers", 0)
                    )

                fig2 = go.Figure()
                for seg, style in SPARKLINE_PALETTE.items():
                    if seg not in hours_data:
                        continue
                    hd = hours_data[seg]
                    xs = sorted(hd.keys())
                    ys = [hd[h] for h in xs]
                    fig2.add_trace(go.Scatter(
                        x=xs, y=ys,
                        mode="lines+markers",
                        name=style["label"],
                        line=dict(color=style["color"], width=2.5),
                        marker=dict(size=6, color=style["color"]),
                        fill="tozeroy",
                        fillcolor=style["color"].replace(")", ", 0.08)").replace("rgb", "rgba") if style["color"].startswith("rgb") else style["color"] + "14",
                    ))
                fig2.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(255,255,255,0.5)",
                    font=dict(color="#7b4f8a", family="DM Sans, sans-serif"),
                    xaxis=dict(
                        showgrid=False,
                        tickmode="array",
                        tickvals=list(range(0, 24, 2)),
                        ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)],
                        tickfont=dict(size=10),
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor="rgba(218,112,214,0.12)",
                        tickformat=",",
                        zeroline=False,
                    ),
                    legend=dict(
                        bgcolor="rgba(255,255,255,0.7)",
                        bordercolor="rgba(255,105,180,0.2)",
                        borderwidth=1,
                        font=dict(color="#6d3880"),
                        orientation="h",
                        y=1.12,
                    ),
                    margin=dict(t=30, b=10, l=10, r=10),
                    height=260,
                )
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No telemetry data available for this hour.")
    else:
        st.info("No program currently scheduled.")

with col_info:
    st.markdown('<div class="section-label">🎬 Now Playing</div>', unsafe_allow_html=True)
    if current_slot:
        meta         = catalog.get(current_slot["show_id"], {})
        film_rt      = current_slot.get("content_runtime_min")
        blk_dur      = current_slot.get("duration_min")
        interstitial = (blk_dur - film_rt) if isinstance(film_rt, int) and isinstance(blk_dur, int) else None

        desc = meta.get("description", "")
        st.markdown(
            f'<div class="glass-card">'
            f'<div class="glass-card-title">🎬 {current_slot["title"]}</div>'
            f'<div class="glass-card-desc">{desc[:190] + "…" if len(desc) > 190 else desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.metric("🎞 Film Runtime",   f"{film_rt} min"      if film_rt      else "—")
        st.metric("📐 Block Duration", f"{blk_dur} min"      if blk_dur      else "—")
        st.metric("✨ Interstitial",   f"{interstitial} min" if interstitial else "—",
                  help="High-Fashion Ad Breaks + Exclusive Designer Interviews")

        if meta.get("release_year"):
            st.caption(f"📅 **Year:** {meta['release_year']}")
        if meta.get("listed_in"):
            st.caption(f"🎭 **Genres:** {meta['listed_in']}")
    else:
        st.caption("*Outside scheduled window.*")

st.markdown("---")

# ── Miranda Intelligence Console ───────────────────────────────────────────────
st.markdown(
    '<div class="section-label" style="margin-bottom:0.6rem">💬 Miranda Intelligence Console</div>',
    unsafe_allow_html=True,
)
st.caption("✦ Ask Miranda anything about ch_runway_01 — programming strategy, audience telemetry, scheduling.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history: list[dict] = []
if "last_raw" not in st.session_state:
    st.session_state.last_raw = None

# Render conversation history as styled bubbles
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-bubble-wrap">'
            f'<div class="user-bubble">💬 {msg["content"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        ts_label = "Miranda · ch_runway_01"
        st.markdown(
            f'<div class="miranda-bubble">'
            f'<div class="miranda-name">✦ Miranda Priestly &nbsp;·&nbsp; Runway Inclusive</div>'
            f'{msg["content"]}'
            f'<div class="miranda-ts">{ts_label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# Chat input
if user_input := st.chat_input("Ask Miranda something fierce…"):
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Show user bubble immediately
    st.markdown(
        f'<div class="user-bubble-wrap">'
        f'<div class="user-bubble">💬 {user_input}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("✦ Miranda is reviewing the data…"):
        try:
            payload = {
                "messages": [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.chat_history
                ]
            }
            resp = requests.post(NAT_SERVER, json=payload, timeout=180)
            resp.raise_for_status()

            try:
                raw = resp.json()
            except ValueError:
                raw = {"raw_text": resp.text}
            st.session_state.last_raw = raw

            if isinstance(raw, dict) and "choices" in raw:
                reply = (
                    raw["choices"][0].get("message", {}).get("content")
                    or raw["choices"][0].get("delta", {}).get("content")
                    or str(raw)
                )
            elif isinstance(raw, dict) and "raw_text" in raw:
                reply = raw["raw_text"]
            else:
                reply = str(raw)

            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.markdown(
                f'<div class="miranda-bubble">'
                f'<div class="miranda-name">✦ Miranda Priestly &nbsp;·&nbsp; Runway Inclusive</div>'
                f'{reply}'
                f'<div class="miranda-ts">Just now · ch_runway_01</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        except requests.exceptions.ConnectionError:
            err = "Miranda is unavailable — NAT server not reachable at http://localhost:8080."
            st.error(err)
            st.session_state.last_raw = {"error": err}
        except requests.exceptions.Timeout:
            err = "Request timed out. Miranda may be processing a complex query."
            st.warning(err)
            st.session_state.last_raw = {"error": err}
        except Exception as exc:
            err = f"Request failed: {exc}"
            st.error(err)
            st.session_state.last_raw = {"error": err}

# ── Architect View ─────────────────────────────────────────────────────────────
if st.session_state.last_raw:
    with st.expander("🔬 Architect View — Raw Server Response"):
        st.markdown(
            '<p style="font-size:0.7rem;letter-spacing:0.18em;color:#a080a0;'
            'text-transform:uppercase;margin-bottom:0.5rem">'
            'Full JSON from the NAT /generate endpoint — inspect tool routing, '
            'finish_reason, token usage &amp; model metadata.'
            '</p>',
            unsafe_allow_html=True,
        )
        st.json(st.session_state.last_raw)
