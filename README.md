# Runway Inclusive — Media Intelligence Agent

> *"We don't just play movies, we curate experiences."*
> Powered by NVIDIA NeMo Agent Toolkit · RAPIDS GPU acceleration 
---

## Overview

**Runway Inclusive** (`ch_runway_01`) is an AI-powered media intelligence agent built on the NVIDIA NeMo Agent Toolkit (NAT). It embodies the persona of **A vogue fashion editor** — a sharp, data-driven executive managing a FAST channel network whose primary audience is Female and LGBT+ viewers.

The agent answers natural-language questions about live scheduling, audience telemetry, programming strategy, fashion history, and designer intelligence — all backed by real data and a GPU-accelerated ML pipeline.

---


## Capabilities

### 1. Live EPG & Schedule Intelligence
The agent knows what is airing on `ch_runway_01` at any given moment.

> *"What's playing at 4 PM today?"*

- Queries `data/schedule.json` — a professional block-scheduled EPG
- Every slot is snapped to the nearest **30-minute boundary** (`block_duration_min`)
- Returns `content_runtime_min` (actual film length) alongside `block_duration_min`
- The gap between the two is reserved for **High-Fashion Ad Breaks** and **Exclusive Designer Interviews**

---

### 2. Audience Telemetry & Demographic Analysis
Real-time viewership broken out by the two primary audience segments.

> *"How is The Devil Wears Prada performing with our LGBT+ audience at prime time?"*

| Segment | Baseline multiplier | Prime-time spike (high-fashion titles) |
|---|---|---|
| `Female_Viewers` | 1.4× | +20% (16:00–22:00) |
| `LGBTQ_Core_Audience` | 0.55× | +15% (16:00–22:00) |

---

### 3. CMS Catalog Metadata
Full movie metadata for every title in the female-led catalog.

> *"Tell me about Cruella — runtime, genres, description."*

- 1,378 female-led titles filtered from the TMDB v11 dataset
- Blocklist enforced: Zoolander and male-led titles excluded
- Returns `content_runtime_min`, `block_duration_min`, and `interstitial_min`

---

### 4. Channel Strategy Blueprint
The channel's identity, audience mandate, and programming philosophy.

> *"What is Runway Inclusive's brand strategy?"*

```
Identity  : Runway Inclusive
Audience  : Primary — Female / LGBT+  ·  Secondary — 18-35
Strategy  : Curating iconic female performances that define style
            and subvert the status quo
```

---

### 5. Fashion Designer Knowledge Base
49 designers and houses from Alaïa to Zoran — career history, hallmarks, active era.

> *"What are Vivienne Westwood's hallmarks?"*  
> *"Which French designers worked for Balenciaga?"*

- Full-text search across name, career, hallmarks, nationality, and era
- Sourced from the *Fashion Design from A–Z* reference document

---

### 6. Met Gala Theme History (1973–2025)
Complete Costume Institute theme archive.

> *"What was the Met Gala theme in 2019?"*  
> *"Show me all themes related to American fashion."*

- 49 entries from *The World of Balenciaga* (1973) to *Superfine: Tailoring Black Style* (2025)
- Recent years include dress code, co-chairs, and curatorial notes

---

### 7. Style Tribe Intelligence (GPU-Accelerated)
The **Condé Nast Accelerated Intelligence Layer** — a K-Means ML pipeline that clusters every designer into one of five Style Tribes.

> *"What tribe does Schiaparelli belong to? What should I pair her documentary with?"*

| Tribe | Representative Designers |
|---|---|
| **Avant-Garde** | Balenciaga, Hamnett, Westwood, Kawakubo |
| **Minimalist** | Beene, Halston, de la Renta, Ellis |
| **Heritage Couture** | Balmain, Dior, Fendi, Valentino |
| **Romantic Feminine** | Armani, Chanel, Schiaparelli, Mori |
| **Street & Youth** | Fiorucci, Gaultier, Kenzo, Quant |

Miranda cites the tribe by name in every scheduling justification:
> *"I've paired this documentary with The Devil Wears Prada because both fall into our GPU-clustered Heritage Couture tribe, maximising stylistic coherence for our audience."*

**Pipeline:** TF-IDF vectorisation → `cuml.KMeans` (GPU) / `sklearn.KMeans` (CPU fallback) → cosine-distance KNN index

---

### 8. Designer Similarity Search (cuML KNN)
Find mathematically similar designers by cosine distance on TF-IDF feature vectors.

> *"Who are the designers most similar to Chanel?"*

Returns up to 5 nearest neighbours with tribe labels and similarity scores — powered by the `knn_index.pkl` built by `ml_engine.py`.

---

## Data Files

| File | Size | Description |
|---|---|---|
| `data/catalog.json` | 794 KB | 1,378 female-led movie entries (TMDB v11) |
| `data/schedule.json` | 2.3 KB | 24-hour block-scheduled EPG (10 slots) |
| `data/telemetry.json` | 7.3 MB | 33,072 viewership records (2 segments × 12 hours × 1,378 titles) |
| `data/designers.json` | 14 KB | 49 designer/house profiles A–Z |
| `data/designers_clustered.json` | 28 KB | Designers enriched with Style Tribe labels |
| `data/tribe_manifest.json` | 1.1 KB | Tribe name → member designer list |
| `data/knn_index.pkl` | 40 KB | Fitted NearestNeighbors model + feature matrix |
| `data/met_gala_themes.json` | 3.6 KB | Met Gala themes 1973–2025 |

All generated files are committed — no raw dataset download required on the Brev instance.

---

## Tools Reference

| `_type` in config | Function | Description |
|---|---|---|
| `runway_blueprint` | `get_channel_blueprint` | Channel identity, audience, strategy |
| `runway_schedule` | `get_current_schedule` | What's airing at a given timestamp |
| `runway_metadata` | `lookup_content_metadata` | Movie metadata + block/interstitial breakdown |
| `runway_telemetry` | `get_audience_telemetry` | Female + LGBTQ+ viewership at a given hour |
| `runway_designers` | `search_fashion_designers` | Free-text designer knowledge base search |
| `runway_met_gala` | `get_met_gala_themes` | Met Gala theme by year or full history |
| `runway_tribe_intel` | `get_strategic_programming_insight` | GPU-clustered tribe + catalog recommendations |
| `runway_knn` | `find_similar_designers` | Cosine-similarity nearest neighbours |

---

## Deployment

### Local
```bash
export NVIDIA_API_KEY=<your-key>
nat serve --config_file workflow-config.yml
```

### Brev GPU Instance (one-shot)
```bash
git clone https://github.com/mashby2022/runway-agent.git
cd runway-agent
chmod +x setup_brev.sh && ./setup_brev.sh

source .venv/bin/activate
export NVIDIA_API_KEY=<your-key>
nat serve --config_file workflow-config.yml
```

`setup_brev.sh` auto-detects CUDA version and installs the matching `cudf-cu12` / `cuml-cu12` build from `pypi.nvidia.com`.

### Frontend (Lovable + ngrok)
```bash
ngrok http 8080
# paste the https tunnel URL into your Lovable project as the API base
```

CORS is pre-configured in `workflow-config.yml`:
```yaml
cors:
  allow_origins: ["*"]
  allow_methods: ["*"]
  allow_headers: ["*"]
```

---

## ML Engine

Run standalone to regenerate clustered data (runs on GPU if RAPIDS is available):

```bash
python ml_engine.py
```

```
════════════════════════════════════════════════════
  RUNWAY INCLUSIVE — STRATEGIC BRAIN (ML ENGINE)
  Condé Nast Accelerated Intelligence Layer
════════════════════════════════════════════════════
[1] Data ingestion
[2] DataFrame build + GPU-merge (cudf.pandas)
[3] TF-IDF vectorisation
[4] K-Means — CPU baseline vs GPU speedup
[5] Style Tribe assignment
[6] KNN index build
[7] Outputs saved
    PERFORMANCE REPORT (for Head of ML)
```

---

## Interstitial Policy

Every movie airs inside a block snapped to the nearest 30-minute boundary. The gap between `content_runtime_min` and `block_duration_min` is **never dead air**:

- **High-Fashion Ad Breaks** — premium sponsors aligned to Female/LGBT+ audience
- **Exclusive Designer Interviews** — original interstitial content


