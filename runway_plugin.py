"""
NAT plugin: registers Runway FAST-channel Python tools so they are
discoverable by nvidia-nat as _type values in workflow-config.yml.

Registered _type names
  runway_blueprint  – get_channel_blueprint(channel_id)
  runway_schedule   – get_current_schedule(channel_id, timestamp)
  runway_metadata   – lookup_content_metadata(content_id)
  runway_telemetry  – get_audience_telemetry(content_id, current_time)
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


# ── Input schemas for multi-parameter tools ─────────────────────────────────
# FunctionInfo.from_fn requires exactly one parameter; use Pydantic models
# for tools that originally took two arguments.

class ScheduleQuery(BaseModel):
    channel_id: str = Field(description="Channel identifier, e.g. 'ch_runway_01'")
    timestamp: str = Field(description="ISO-8601 timestamp, e.g. '2026-04-20T16:00:00+00:00'")


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
