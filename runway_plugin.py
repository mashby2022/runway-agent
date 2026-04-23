"""
NAT plugin: registers Runway FAST-channel Python tools so they are
discoverable by nvidia-nat as _type values in workflow-config.yml.

Registered _type names
  runway_blueprint       – get_channel_blueprint(channel_id)
  runway_schedule        – get_current_schedule(channel_id, timestamp)
  runway_metadata        – lookup_content_metadata(content_id)
  runway_telemetry       – get_audience_telemetry(content_id, current_time)
  runway_designers       – search_fashion_designers(query)
  runway_met_gala        – get_met_gala_themes(year)
  runway_tribe_intel     – get_strategic_programming_insight(query)
  runway_knn             – find_similar_designers(brand_name)
  runway_update_schedule – update_schedule_slot(slot_time, new_title)
  runway_toggle_mode     – toggle_system_mode(mode)
"""

import json

from pydantic import BaseModel, Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig


# ── Config types (each _type string is the name= arg) ──────────────────────

class BlueprintToolConfig(FunctionBaseConfig, name="runway_blueprint"):
    """Channel strategy blueprint lookup."""


class ScheduleToolConfig(FunctionBaseConfig, name="runway_schedule"):
    """EPG schedule lookup for a channel at a given timestamp."""


class MetadataToolConfig(FunctionBaseConfig, name="runway_metadata"):
    """CMS catalog metadata lookup by show_id."""


class TelemetryToolConfig(FunctionBaseConfig, name="runway_telemetry"):
    """Audience telemetry lookup for a show at a given hour."""


class DesignersToolConfig(FunctionBaseConfig, name="runway_designers"):
    """Fashion designer knowledge-base search."""


class MetGalaToolConfig(FunctionBaseConfig, name="runway_met_gala"):
    """Met Gala theme lookup by year or full history."""


class TribeIntelToolConfig(FunctionBaseConfig, name="runway_tribe_intel"):
    """Strategic programming insight from the GPU-clustered Style Tribe index."""


class KNNToolConfig(FunctionBaseConfig, name="runway_knn"):
    """cuML KNN similarity search across the fashion designer index."""


class UpdateScheduleToolConfig(FunctionBaseConfig, name="runway_update_schedule"):
    """Update a schedule slot with a new title and recalculate block padding."""


class ToggleModeToolConfig(FunctionBaseConfig, name="runway_toggle_mode"):
    """Switch pipeline execution between ONLINE (GPU) and OFFLINE (CPU mock) modes."""


class MovieCatalogToolConfig(FunctionBaseConfig, name="runway_movie_catalog"):
    """Return the full dropdown-ready movie catalog with show_id, title, genres, runtime."""


class GenerateCandidatesToolConfig(FunctionBaseConfig, name="runway_generate_candidates"):
    """RAPIDS-first recommender: top-5 Female/LGBT+ candidates by tribe + engagement."""


class SystemHealthToolConfig(FunctionBaseConfig, name="runway_system_health"):
    """Heartbeat tool — returns full compute performance metadata for the dashboard."""


class GenerateWeeklyPlanToolConfig(FunctionBaseConfig, name="runway_generate_weekly_plan"):
    """Generate a 7-day programming grid aligned to Daily Strategic Themes."""


class UpdateWeeklySlotToolConfig(FunctionBaseConfig, name="runway_update_weekly_slot"):
    """Move a title between weekly slots with demographic friction validation."""


class StrategicFrictionToolConfig(FunctionBaseConfig, name="runway_strategic_friction"):
    """Pre-flight Event Exclusivity check — returns is_final_conflict before a slot move."""


class QuadAnalysisToolConfig(FunctionBaseConfig, name="runway_quad_analysis"):
    """Audience Composition (Gold/Silver/Occasional) for a DMA market. Never returns empty."""


class AudienceMetricsToolConfig(FunctionBaseConfig, name="runway_audience_metrics"):
    """Channel KPI metrics: active_viewers, avg_watch_time, peak_concurrent, engagement_rate."""


class RetrievalToolConfig(FunctionBaseConfig, name="runway_retrieval"):
    """Semantic FAISS vector retrieval from the historical insights vault."""


class TfValidationToolConfig(FunctionBaseConfig, name="runway_tf_validation"):
    """Validate LLM thematic scores against TF supervised demographic reach models."""


class GraphRelationshipToolConfig(FunctionBaseConfig, name="runway_graph_relationship"):
    """RAPIDS cuGraph — map script tag relationships to catalog; returns 500x acceleration metadata."""


class DistillScriptToolConfig(FunctionBaseConfig, name="runway_distill_script"):
    """100k Tag Universe distiller — extract 200-300 canonical story elements from script text."""


# ── Input schemas for multi-parameter tools ─────────────────────────────────
# FunctionInfo.from_fn requires exactly one parameter; use Pydantic models
# for tools that originally took two arguments.

class ScheduleQuery(BaseModel):
    channel_id: str = Field(description="Channel identifier, e.g. 'ch_runway_01'")
    timestamp: str = Field(description="ISO-8601 timestamp, e.g. '2026-04-20T16:00:00+00:00'")


class UpdateScheduleQuery(BaseModel):
    slot_time: str = Field(description="ISO-8601 start time or HH:MM shorthand, e.g. '16:00' or '2026-04-20T16:00:00+00:00'")
    new_title: str = Field(description="Title of the replacement film from the catalog, e.g. 'Clueless'")


class TelemetryQuery(BaseModel):
    content_id: str = Field(description="Show identifier, e.g. 's0001'")
    current_time: str = Field(description="ISO-8601 timestamp used to select the time window")
    demographic: str = Field(default="", description="Optional segment — 'Female', 'LGBT+', 'Gen_Z', 'Millennial', 'Male'. Defaults to core Female+LGBTQ+.")


class GenerateCandidatesQuery(BaseModel):
    context_tribe: str = Field(default="", description="Style Tribe name or keyword, e.g. 'Heritage Couture'. Leave blank for all tribes.")
    demographic: str = Field(default="", description="Target segment — 'Female', 'LGBT+', 'Gen_Z', 'Millennial', 'Silver_Stylists', 'Male'. Defaults to Female+LGBTQ+ core.")
    location_segment: str = Field(default="", description="DMA market — 'New York (DMA 1)', 'Dallas (DMA 4)', 'Paris', 'Europe', etc. Leave blank for all markets.")
    density_tier: str = Field(default="", description="'Urban Core', 'Affluent Suburban', or 'Exurban'. Affluent Suburban applies a Heritage Couture bias.")


class GenerateWeeklyPlanQuery(BaseModel):
    week_start: str = Field(
        default="",
        description="ISO date for the week's Monday, e.g. '2026-04-20'. Leave blank to use the current project week.",
    )


class UpdateWeeklySlotQuery(BaseModel):
    from_day: str = Field(description="Source day name, e.g. 'Monday'")
    from_time: str = Field(description="Source slot time HH:MM, e.g. '20:00'")
    to_day: str = Field(description="Target day name, e.g. 'Wednesday'")
    to_time: str = Field(description="Target slot time HH:MM, e.g. '20:00'")


class StrategicFrictionQuery(BaseModel):
    title: str = Field(description="Title of the content to check, e.g. 'The First Monday in May'")
    target_day: str = Field(description="Destination day name, e.g. 'Saturday'")


class QuadAnalysisQuery(BaseModel):
    market:  str = Field(default="", description="Market name or DMA string, e.g. 'Dallas' or 'Dallas (DMA 4)'")
    segment: str = Field(default="", description="Optional demographic filter, e.g. 'Female' or 'LGBT+'")


class AudienceMetricsQuery(BaseModel):
    market:  str = Field(default="", description="Market name or DMA string, e.g. 'Dallas' or 'Dallas (DMA 4)'")
    segment: str = Field(default="", description="Optional demographic filter, e.g. 'Female' or 'LGBT+'")


class TfValidationQuery(BaseModel):
    title:               str   = Field(description="Content title to validate, e.g. 'Roxanne Lowit Magic Moments'")
    llm_thematic_score:  float = Field(default=0.0, description="LLM resonance score 0.0–1.0. Pass 0.0 to auto-derive from historical completion rate.")
    target_network:      str   = Field(default="cbs_broadcast", description="Network key: 'cbs_broadcast', 'nbc_broadcast', 'netflix_streaming', 'hulu_streaming', 'tubi_avod', 'prime_streaming'")
    demographic:         str   = Field(default="25-34", description="Target age bracket: '18-24', '25-34', '35-49', '50+'")


class GraphRelationshipQuery(BaseModel):
    script_tags: str = Field(description="Comma- or space-separated story tags, e.g. 'empowerment career drama female_led'")
    top_k:       int = Field(default=10, description="Number of top catalog matches to return (default 10, max 50)")


class DistillScriptQuery(BaseModel):
    script_text: str = Field(description="Script summary, logline, or description to analyse against the 100k tag universe")
    max_tags:    int = Field(default=250, description="Maximum canonical tags to return (200–300 recommended)")


# ── Function registrations ───────────────────────────────────────────────────

@register_function(config_type=BlueprintToolConfig)
async def _register_blueprint(_config: BlueprintToolConfig, _builder: Builder):
    from tools import get_channel_blueprint

    async def fn(channel_id: str) -> str:
        return get_channel_blueprint(channel_id)

    yield FunctionInfo.from_fn(
        fn,
        description="Return the channel strategy blueprint for a given channel_id (e.g. 'ch_runway_01').",
    )


@register_function(config_type=ScheduleToolConfig)
async def _register_schedule(_config: ScheduleToolConfig, _builder: Builder):
    from tools import get_current_schedule

    async def fn(query: ScheduleQuery) -> str:
        result = get_current_schedule(query.channel_id, query.timestamp)
        return json.dumps(result)

    yield FunctionInfo.from_fn(
        fn,
        description="Return the program airing on a FAST channel at a given ISO-8601 timestamp.",
    )


@register_function(config_type=MetadataToolConfig)
async def _register_metadata(_config: MetadataToolConfig, _builder: Builder):
    from tools import lookup_content_metadata

    async def fn(content_id: str) -> str:
        result = lookup_content_metadata(content_id)
        return json.dumps(result)

    yield FunctionInfo.from_fn(
        fn,
        description="Return full CMS catalog metadata for an asset identified by show_id (e.g. 's0001').",
    )


@register_function(config_type=TelemetryToolConfig)
async def _register_telemetry(_config: TelemetryToolConfig, _builder: Builder):
    from tools import get_audience_telemetry

    async def fn(query: TelemetryQuery) -> str:
        result = get_audience_telemetry(query.content_id, query.current_time, query.demographic)
        return json.dumps(result)

    yield FunctionInfo.from_fn(
        fn,
        description=(
            "Return viewership data for an asset at a given ISO-8601 hour. "
            "Always returns Female and LGBTQ+ core segment stats. "
            "Optionally pass a demographic ('Gen_Z', 'Millennial', 'Male') "
            "to get that segment alongside the core for comparison. "
            "Response includes top_performing_market (DMA with highest completion rate)."
        ),
    )


@register_function(config_type=DesignersToolConfig)
async def _register_designers(_config: DesignersToolConfig, _builder: Builder):
    from tools import search_fashion_designers

    async def fn(query: str) -> str:
        result = search_fashion_designers(query)
        return json.dumps(result)

    yield FunctionInfo.from_fn(
        fn,
        description=(
            "Search the fashion designer knowledge base. "
            "Pass a name, house, hallmark keyword, nationality, or era "
            "(e.g. 'Chanel', 'leather', '1960s', 'Japanese'). "
            "Returns matching designer profiles including career history and hallmarks."
        ),
    )


@register_function(config_type=MetGalaToolConfig)
async def _register_met_gala(_config: MetGalaToolConfig, _builder: Builder):
    from tools import get_met_gala_themes

    async def fn(year: int = 0) -> str:
        result = get_met_gala_themes(year if year else None)
        return json.dumps(result)

    yield FunctionInfo.from_fn(
        fn,
        description=(
            "Return Met Gala Costume Institute theme records from 1973 to 2025. "
            "Pass a specific year (e.g. 2024) for a single entry, "
            "or 0 to retrieve the full history."
        ),
    )


@register_function(config_type=TribeIntelToolConfig)
async def _register_tribe_intel(_config: TribeIntelToolConfig, _builder: Builder):
    from tools import get_strategic_programming_insight

    async def fn(query: str) -> str:
        result = get_strategic_programming_insight(query)
        return json.dumps(result)

    yield FunctionInfo.from_fn(
        fn,
        description=(
            "Condé Nast Accelerated Intelligence Layer. "
            "Returns Style Tribe alignment and catalog recommendations for a given "
            "aesthetic query, designer name, or tribe keyword. "
            "Powered by GPU-clustered K-Means (cuML/sklearn). "
            "Use this to justify scheduling decisions with tribe consistency."
        ),
    )


@register_function(config_type=KNNToolConfig)
async def _register_knn(_config: KNNToolConfig, _builder: Builder):
    from tools import find_similar_designers

    async def fn(brand_name: str) -> str:
        result = find_similar_designers(brand_name)
        return json.dumps(result)

    yield FunctionInfo.from_fn(
        fn,
        description=(
            "Find the 5 most stylistically similar designers to a given name "
            "using the cuML KNN similarity index (cosine distance on TF-IDF vectors). "
            "Returns neighbours with their Style Tribe and similarity score."
        ),
    )


@register_function(config_type=ToggleModeToolConfig)
async def _register_toggle_mode(_config: ToggleModeToolConfig, _builder: Builder):
    from tools import toggle_system_mode

    async def fn(mode: str) -> str:
        result = toggle_system_mode(mode)
        return json.dumps(result)

    yield FunctionInfo.from_fn(
        fn,
        description=(
            "Switch the analytics pipeline between ONLINE (NVIDIA A10G GPU via cuDF/cuML) "
            "and OFFLINE (MacBook CPU via pandas mock) execution modes. "
            "Pass 'ONLINE' or 'OFFLINE'. "
            "All subsequent tool calls will reflect the active compute profile in their "
            "source_compute and engine metadata fields."
        ),
    )


@register_function(config_type=UpdateScheduleToolConfig)
async def _register_update_schedule(_config: UpdateScheduleToolConfig, _builder: Builder):
    from tools import update_schedule_slot

    async def fn(query: UpdateScheduleQuery) -> str:
        return update_schedule_slot(query.slot_time, query.new_title)

    yield FunctionInfo.from_fn(
        fn,
        description=(
            "Miranda's Strategic Authority tool. "
            "Replace the film in a specific schedule slot with a new title from the catalog. "
            "Automatically looks up the replacement film's runtime, recalculates the "
            "block_duration_min (nearest 30-min boundary), and updates interstitial padding. "
            "Pass slot_time as ISO-8601 or HH:MM (e.g. '16:00'), and new_title as the film name."
        ),
    )


@register_function(config_type=MovieCatalogToolConfig)
async def _register_movie_catalog(_config: MovieCatalogToolConfig, _builder: Builder):
    from tools import get_movie_catalog

    async def fn(channel_id: str = "ch_runway_01") -> str:  # noqa: ARG001
        result = get_movie_catalog()
        return json.dumps(result)

    yield FunctionInfo.from_fn(
        fn,
        description=(
            "Return the full movie catalog as a clean, dropdown-ready list. "
            "Each entry has show_id, title, genres, and runtime_min. "
            "Use this to populate UI dropdowns or to check available titles before "
            "making scheduling decisions."
        ),
    )


@register_function(config_type=GenerateCandidatesToolConfig)
async def _register_generate_candidates(_config: GenerateCandidatesToolConfig, _builder: Builder):
    from tools import generate_candidates

    async def fn(query: GenerateCandidatesQuery) -> str:
        result = generate_candidates(query.context_tribe, query.demographic, query.location_segment, query.density_tier)
        return json.dumps(result)

    yield FunctionInfo.from_fn(
        fn,
        description=(
            "Condé Nast Accelerated Intelligence Layer — RAPIDS-first recommender. "
            "Accepts context_tribe (Style Tribe), demographic ('Female','LGBT+','Gen_Z',"
            "'Millennial','Male'), and location_segment (DMA market or region like 'Paris',"
            "'New York','Europe'). Defaults to Female+LGBTQ+ core across all markets. "
            "OFFLINE: pandas with European fashion title boost for EU markets. "
            "ONLINE: cuDF + co-visitation scoring. "
            "Always call before making a scheduling recommendation."
        ),
    )


@register_function(config_type=SystemHealthToolConfig)
async def _register_system_health(_config: SystemHealthToolConfig, _builder: Builder):
    from tools import get_system_health

    async def fn(channel_id: str = "ch_runway_01") -> str:  # noqa: ARG001
        result = get_system_health()
        return json.dumps(result)

    yield FunctionInfo.from_fn(
        fn,
        description=(
            "Heartbeat tool — returns the current compute performance metadata: "
            "execution_mode (ONLINE/OFFLINE), source_compute, engine name, "
            "gpu_boost factor, latency_ms, and HAS_GPU detection. "
            "Use this when the user asks about system status, compute mode, "
            "or which engine is currently active."
        ),
    )


@register_function(config_type=GenerateWeeklyPlanToolConfig)
async def _register_generate_weekly_plan(
    _config: GenerateWeeklyPlanToolConfig, _builder: Builder
):
    from tools import generate_weekly_plan

    async def fn(query: GenerateWeeklyPlanQuery) -> str:
        result = generate_weekly_plan(query.week_start)
        return json.dumps(result)

    yield FunctionInfo.from_fn(
        fn,
        description=(
            "Weekly Architect tool — generates a 7-day programming grid (08:00–00:00) "
            "aligned to Daily Strategic Themes (Minimalist Monday, Avant-Garde Wednesday, "
            "Heritage Weekend, etc.). "
            "OFFLINE: matches catalog titles to themes by genre/description keyword alignment "
            "and average completion rate. "
            "ONLINE: uses cuDF to rank by market-specific completion rate per day's peak demo. "
            "Persists the plan to data/weekly_schedule.json. "
            "Call generate_weekly_insights.py first to seed the daily_themes data. "
            "Pass week_start as ISO date (e.g. '2026-04-20') or leave blank for current week."
        ),
    )


@register_function(config_type=QuadAnalysisToolConfig)
async def _register_quad_analysis(_config: QuadAnalysisToolConfig, _builder: Builder):
    from tools import get_quad_analysis

    async def fn(query: QuadAnalysisQuery) -> str:
        return get_quad_analysis(query.market, query.segment)

    yield FunctionInfo.from_fn(
        fn,
        description=(
            "Audience Composition analysis — Gold / Silver / Occasional loyalty tiers "
            "for a given DMA market, expressed as viewer-weighted percentages. "
            "Gold = ≥85% completion (devoted audience). "
            "Silver = 60–85% completion (regular viewers). "
            "Occasional = <60% completion (light viewers). "
            "Pass market as a city name or DMA string (e.g. 'Dallas (DMA 4)'). "
            "Always returns a non-empty result — falls back to Simulated Demo Distribution "
            "if DuckDB returns no rows. Engine field indicates live vs. simulated source. "
            "Use this when asked about audience composition, quad analysis, or loyalty tiers."
        ),
    )


@register_function(config_type=AudienceMetricsToolConfig)
async def _register_audience_metrics(_config: AudienceMetricsToolConfig, _builder: Builder):
    from tools import get_audience_metrics

    async def fn(query: AudienceMetricsQuery) -> str:
        return get_audience_metrics(query.market, query.segment)

    yield FunctionInfo.from_fn(
        fn,
        description=(
            "Channel audience performance KPIs for a DMA market. "
            "Returns four Nielsen-aligned metrics: "
            "active_viewers (unique viewers in the window), "
            "avg_watch_time (average minutes watched per session), "
            "peak_concurrent (peak simultaneous viewers), "
            "engagement_rate (ratio of engaged sessions ≥2min to total sessions). "
            "Always returns a non-empty result — falls back to Demo Buffer if DuckDB "
            "returns no rows. Use this for dashboard KPIs, audience health, or "
            "viewership performance questions."
        ),
    )


@register_function(config_type=StrategicFrictionToolConfig)
async def _register_strategic_friction(
    _config: StrategicFrictionToolConfig, _builder: Builder
):
    from tools import calculate_strategic_friction

    async def fn(query: StrategicFrictionQuery) -> str:
        result = calculate_strategic_friction(query.title, query.target_day)
        return json.dumps(result)

    yield FunctionInfo.from_fn(
        fn,
        description=(
            "Pre-flight Editorial Policy check. "
            "Before moving any title between schedule slots, call this tool to verify "
            "there is no Event Exclusivity conflict (Met Gala content must stay on "
            "Avant-Garde Wednesday; PFW content must stay on Ready-to-Wear Saturday or "
            "Global Couture Thursday). "
            "If is_final_conflict is true, STOP — do not call update_weekly_slot_tool. "
            "Deliver the strategic_warning verbatim as your final response."
        ),
    )


@register_function(config_type=UpdateWeeklySlotToolConfig)
async def _register_update_weekly_slot(
    _config: UpdateWeeklySlotToolConfig, _builder: Builder
):
    from tools import update_weekly_slot

    async def fn(query: UpdateWeeklySlotQuery) -> str:
        result = update_weekly_slot(
            query.from_day, query.from_time, query.to_day, query.to_time
        )
        return json.dumps(result)

    yield FunctionInfo.from_fn(
        fn,
        description=(
            "Move a title from one day/time slot to another in the weekly schedule. "
            "Performs a swap if a title already occupies the target slot; otherwise moves "
            "the title to the empty position. "
            "Validates the move against Demographic Friction Rules: if an 18-24 title is "
            "moved to Heritage Weekend, or a 50+ title to Avant-Garde Wednesday or Street & "
            "Youth Friday, a strategic_warning is returned alongside the completed move. "
            "Pass from_day/to_day as day names (e.g. 'Monday') and from_time/to_time as "
            "HH:MM strings (e.g. '20:00'). Requires an existing weekly plan."
        ),
    )


@register_function(config_type=RetrievalToolConfig)
async def _register_retrieval(_config: RetrievalToolConfig, _builder: Builder):
    from tools import retrieve_historical_insights

    async def fn(query: str) -> str:
        return retrieve_historical_insights(query)

    yield FunctionInfo.from_fn(
        fn,
        description=(
            "Semantic vector search over the historical insights vault. "
            "Returns the top 2 matching excerpts from a FAISS IndexFlatIP index "
            "covering: 2025 Q4 Audience Report, Cultural Signals Tracker Mar 2026, "
            "and Met Gala 2024 Post-Mortem. "
            "Use this to ground recommendations in historical evidence and cultural signals."
        ),
    )


@register_function(config_type=TfValidationToolConfig)
async def _register_tf_validation(_config: TfValidationToolConfig, _builder: Builder):
    from tools import validate_tf_predictions

    async def fn(query: TfValidationQuery) -> str:
        return validate_tf_predictions(
            query.title,
            query.llm_thematic_score,
            query.target_network,
            query.demographic,
        )

    yield FunctionInfo.from_fn(
        fn,
        description=(
            "Validation Bridge — compares the generative agent's thematic reasoning "
            "against Vault's TF supervised demographic reach model (tf-demographic-reach-v2). "
            "Returns confidence_score (0–100), variance_delta (% divergence between LLM "
            "prediction and TF model), and tf_reach_prediction (e.g. '78% reach on CBS'). "
            "validation_status is VALIDATED if confidence ≥ 70, else DIVERGENCE_DETECTED. "
            "Use before finalising any acquisition or scheduling recommendation."
        ),
    )


@register_function(config_type=GraphRelationshipToolConfig)
async def _register_graph_relationship(_config: GraphRelationshipToolConfig, _builder: Builder):
    from tools import map_graph_relationships

    async def fn(query: GraphRelationshipQuery) -> str:
        return map_graph_relationships(query.script_tags, query.top_k)

    yield FunctionInfo.from_fn(
        fn,
        description=(
            "RAPIDS cuGraph relationship mapper — builds a bipartite graph between "
            "script/story tags and existing catalog titles. "
            "Returns top_catalog_matches (by edge weight), tag_adjacency_summary, "
            "and acceleration_metadata: 'Acceleration: 500x (RAPIDS cuGraph) | "
            "Speed: 12ms (GPU) vs 6000ms (CPU)'. "
            "GPU path active on Brev L4 (cuGraph); CPU adjacency fallback on local. "
            "Use when analysing how a new script fits the existing content graph."
        ),
    )


@register_function(config_type=DistillScriptToolConfig)
async def _register_distill_script(_config: DistillScriptToolConfig, _builder: Builder):
    from tools import distill_script_tags

    async def fn(query: DistillScriptQuery) -> str:
        return distill_script_tags(query.script_text, query.max_tags)

    yield FunctionInfo.from_fn(
        fn,
        description=(
            "100k Tag Universe distiller — extracts 200-300 unique canonical story elements "
            "from a script summary or logline, mapped back to the Vault Tag Universe v3. "
            "Each tag has a tag_id (TU-XXXXXX), canonical_tag, category "
            "(narrative_theme / character_archetype / setting / emotional_beat / "
            "cultural_signal / genre_marker / demographic_indicator / platform_fit), "
            "match_confidence, and universe_ref. "
            "Use this as the first step in any new content acquisition analysis."
        ),
    )
