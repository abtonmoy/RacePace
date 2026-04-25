"""SituationReport — the live structured-state contract.

Pure function over (SessionInfo, recent frames, optional strategy state)
returning a flat dataclass. The Phase 2 engineer agent consumes this; the
Phase 3 coach will consume the same shape (extended).

Field names are a public API. Renaming later is painful — be deliberate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from racepace.features import strategy
from racepace.features.laps import Lap, split_into_laps
from racepace.schema import SessionInfo, TelemetryFrame


PaceTrend = Literal["improving", "stable", "degrading"]


@dataclass
class SituationReport:
    # Where we are
    lap: int
    laps_remaining: int | None
    session_type: str | None
    track: str | None
    car: str | None

    # How we're doing
    position: int | None
    last_lap_time_s: float | None
    best_lap_time_s: float | None
    avg_recent_lap_time_s: float | None
    pace_trend: PaceTrend | None

    # Tyres
    tyre_compound: str | None
    avg_tyre_wear_pct: float | None
    tyre_wear_per_lap_pct: float | None
    laps_until_critical_wear: float | None

    # Fuel
    fuel_kg: float | None
    fuel_per_lap_kg: float | None
    fuel_laps_remaining: float | None
    fuel_margin_laps: float | None

    # Traffic
    gap_ahead_s: float | None
    gap_behind_s: float | None
    gap_ahead_trend_s_per_lap: float | None
    gap_behind_trend_s_per_lap: float | None
    undercut_threat: bool
    overtake_opportunity: bool

    # Conditions
    weather: str | None
    track_temp_c: float | None
    rain_incoming: bool | None

    # Strategic
    pit_window_open: bool
    laps_since_last_pit: int | None
    optimal_pit_lap: int | None

    # Recent events (last 30s)
    events: list[str] = field(default_factory=list)


@dataclass
class StrategyState:
    """Persistent state the situation builder uses but cannot derive from a snapshot.

    Carried across ticks by the engineer; resets only at session start.
    """
    last_pit_lap: int | None = None
    fresh_tyre_pace_estimate_s: float | None = None  # if we've ever seen a fresh stint


def build_situation(
    info: SessionInfo,
    snapshot: list[TelemetryFrame],
    strategy_state: StrategyState | None = None,
) -> SituationReport | None:
    """Build a SituationReport from a snapshot of recent frames. Returns None if empty."""
    if not snapshot:
        return None
    state = strategy_state or StrategyState()
    latest = snapshot[-1]

    laps = split_into_laps(snapshot)
    complete_laps = [l for l in laps if l.is_complete and l.lap_time_s is not None]

    # Pace
    last_lap_time = complete_laps[-1].lap_time_s if complete_laps else latest.last_lap_time_s
    best_lap_time = latest.best_lap_time_s
    if best_lap_time is None and complete_laps:
        best_lap_time = min(l.lap_time_s for l in complete_laps if l.lap_time_s)

    recent_3 = [l.lap_time_s for l in complete_laps[-3:] if l.lap_time_s]
    avg_recent = sum(recent_3) / len(recent_3) if recent_3 else None
    pace_trend = _pace_trend(complete_laps)

    # Tyres
    tyre_compound = latest.tyre_compound
    avg_wear = _avg_dict(latest.tyre_wear_pct)
    wear_per_lap = strategy.tyre_wear_per_lap(complete_laps)
    laps_until_crit = strategy.laps_until_critical_wear(avg_wear, wear_per_lap)

    # Fuel
    fuel_kg = latest.fuel_kg
    fpl = strategy.fuel_per_lap(complete_laps)
    fuel_laps_left = (fuel_kg / fpl) if (fuel_kg is not None and fpl and fpl > 0) else latest.fuel_laps_remaining

    laps_left = strategy.laps_to_finish(
        total_laps=info.total_laps,
        current_lap=latest.lap_number,
        avg_lap_time_s=avg_recent,
    )
    margin = strategy.fuel_margin_laps(fuel_kg, fpl, laps_left)

    # Traffic
    gap_ahead, gap_ahead_trend = _gap_with_trend(complete_laps, snapshot, "gap_ahead_s")
    gap_behind, gap_behind_trend = _gap_with_trend(complete_laps, snapshot, "gap_behind_s")

    pit_loss = strategy.pit_loss_seconds(info.track, info.sim)
    undercut = strategy.is_undercut_threat(
        gap_behind_s=gap_behind,
        self_pace_old_s=avg_recent,
        behind_pace_old_s=avg_recent,  # approx — sim rarely exposes opponent pace cleanly
        pit_loss_s=pit_loss,
    )
    overtake = strategy.is_overtake_opportunity(
        gap_ahead_s=gap_ahead,
        self_pace_s=avg_recent,
        ahead_pace_s=avg_recent,  # same caveat
    )

    # Strategic
    window_open = strategy.pit_window_open(margin, laps_until_crit)
    optimal = strategy.optimal_pit_lap(latest.lap_number, laps_left, margin, laps_until_crit)
    laps_since_pit = (latest.lap_number - state.last_pit_lap) if state.last_pit_lap else None

    return SituationReport(
        lap=latest.lap_number,
        laps_remaining=laps_left,
        session_type=info.session_type,
        track=info.track,
        car=info.car,
        position=latest.position,
        last_lap_time_s=last_lap_time,
        best_lap_time_s=best_lap_time,
        avg_recent_lap_time_s=avg_recent,
        pace_trend=pace_trend,
        tyre_compound=tyre_compound,
        avg_tyre_wear_pct=avg_wear,
        tyre_wear_per_lap_pct=wear_per_lap,
        laps_until_critical_wear=laps_until_crit,
        fuel_kg=fuel_kg,
        fuel_per_lap_kg=fpl,
        fuel_laps_remaining=fuel_laps_left,
        fuel_margin_laps=margin,
        gap_ahead_s=gap_ahead,
        gap_behind_s=gap_behind,
        gap_ahead_trend_s_per_lap=gap_ahead_trend,
        gap_behind_trend_s_per_lap=gap_behind_trend,
        undercut_threat=undercut,
        overtake_opportunity=overtake,
        weather=info.weather,
        track_temp_c=info.track_temp_c,
        rain_incoming=None,  # ACC SHM doesn't expose forecast; UDP broadcast does (Phase 4)
        pit_window_open=window_open,
        laps_since_last_pit=laps_since_pit,
        optimal_pit_lap=optimal,
        events=_recent_events(snapshot),
    )


def _avg_dict(d: dict[str, float] | None) -> float | None:
    if not d:
        return None
    vals = list(d.values())
    return sum(vals) / len(vals) if vals else None


def _pace_trend(complete_laps: list[Lap]) -> PaceTrend | None:
    times = [l.lap_time_s for l in complete_laps if l.lap_time_s]
    if len(times) < 3:
        return None
    recent = times[-3:]
    earlier = times[-6:-3] if len(times) >= 6 else times[:-3]
    if not earlier:
        return "stable"
    delta = sum(recent) / len(recent) - sum(earlier) / len(earlier)
    if delta < -0.15:
        return "improving"
    if delta > 0.15:
        return "degrading"
    return "stable"


def _gap_with_trend(
    complete_laps: list[Lap],
    snapshot: list[TelemetryFrame],
    attr: str,
) -> tuple[float | None, float | None]:
    latest_val = getattr(snapshot[-1], attr, None)
    if not complete_laps or len(complete_laps) < 2:
        return latest_val, None
    # Use the gap value at the start-finish crossing of the last few completed laps.
    samples: list[float] = []
    for lap in complete_laps[-3:]:
        # First frame of lap = closest to SF crossing.
        v = getattr(lap.frames[0], attr, None)
        if v is not None:
            samples.append(v)
    if len(samples) < 2:
        return latest_val, None
    # Linear: (last - first) / (n - 1) gives per-lap delta.
    trend = (samples[-1] - samples[0]) / (len(samples) - 1)
    return latest_val, trend


def _recent_events(snapshot: list[TelemetryFrame]) -> list[str]:
    """Last-30s notable events: off-tracks, gear-grab anomalies, etc.

    Kept conservative; the engineer prefers strategic events over trivia.
    """
    if not snapshot:
        return []
    cutoff = snapshot[-1].timestamp_s - 30.0
    recent = [f for f in snapshot if f.timestamp_s >= cutoff]
    events: list[str] = []
    # Off-track heuristic: same as features/laps._is_clean — sample-to-sample
    # speed drop > 30 kph above 50 kph.
    prev: float | None = None
    for f in recent:
        s = f.speed_kph
        if s is None:
            prev = None
            continue
        if prev is not None and prev > 50.0 and (prev - s) > 30.0:
            events.append(f"speed loss at lap {f.lap_number}")
            prev = s
            continue
        prev = s
    return events
