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
        result = get_audience_telemetry(query.content_id, query.current_time)
        return json.dumps(result)

    yield FunctionInfo.from_fn(
        fn,
        description="Return viewership demographic data for an asset at a given ISO-8601 hour.",
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
