"""Normalized telemetry schema.

All units are SI / metric: speed in kph, distances in m, fuel in kg,
temperatures in °C, pressures in psi, times in seconds.

Adapters are responsible for converting from sim-native units before
emitting frames. Agents and storage assume one unit system.

Bump SCHEMA_VERSION on any breaking change to TelemetryFrame or
SessionInfo (added/removed/renamed/retyped field). The storage layer
refuses to load files written under an incompatible version.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0.0"


SimName = Literal["acc", "f1", "iracing", "simhub"]
SessionType = Literal["practice", "qualifying", "race", "hotlap"]
TyreKey = Literal["fl", "fr", "rl", "rr"]


class TelemetryFrame(BaseModel):
    """One telemetry sample. Adapters emit at ~30Hz by default."""

    model_config = ConfigDict(extra="forbid")

    timestamp_s: float = Field(..., description="Seconds since session start; monotonic.")
    lap_number: int = Field(..., description="1-indexed lap number; 0 during out-lap before SF crossing.")

    lap_distance_m: float | None = None
    lap_distance_pct: float | None = Field(default=None, description="0.0–1.0 of current lap.")

    speed_kph: float | None = None
    throttle_pct: float | None = Field(default=None, description="0–100.")
    brake_pct: float | None = Field(default=None, description="0–100.")
    clutch_pct: float | None = Field(default=None, description="0–100.")
    steering_norm: float | None = Field(default=None, description="-1 (full left) to +1 (full right).")
    gear: int | None = Field(default=None, description="-1 reverse, 0 neutral, 1+ forward.")
    rpm: float | None = None

    fuel_kg: float | None = None
    fuel_laps_remaining: float | None = None

    tyre_wear_pct: dict[str, float] | None = Field(
        default=None, description="Keys: fl, fr, rl, rr. Values 0–100."
    )
    tyre_temp_c: dict[str, float] | None = Field(
        default=None, description="Keys: fl, fr, rl, rr."
    )
    tyre_pressure_psi: dict[str, float] | None = Field(
        default=None, description="Keys: fl, fr, rl, rr."
    )
    tyre_compound: str | None = None

    position: int | None = None
    gap_ahead_s: float | None = None
    gap_behind_s: float | None = None

    last_lap_time_s: float | None = None
    best_lap_time_s: float | None = None
    current_sector: int | None = Field(default=None, description="0-indexed sector.")

    extras: dict[str, Any] = Field(default_factory=dict, description="Sim-specific richness.")


class SessionInfo(BaseModel):
    """One row per session, written at start, updated at end."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(..., description="UUID4 string.")
    sim: SimName
    started_at: datetime

    track: str | None = None
    car: str | None = None
    session_type: SessionType | None = None
    ended_at: datetime | None = None
    weather: str | None = None
    track_temp_c: float | None = None
    air_temp_c: float | None = None
    total_laps: int | None = None
