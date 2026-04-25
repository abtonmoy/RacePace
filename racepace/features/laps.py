"""Lap segmentation and per-lap summary stats.

Pure functions over (SessionInfo, list[TelemetryFrame]) — no side effects,
no LLM, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from racepace.schema import TelemetryFrame


@dataclass
class Lap:
    """One lap, with its frames and summary stats.

    `lap_time_s` is taken from the *next* lap's `last_lap_time_s` if
    available (the source of truth from the sim), otherwise computed as
    the elapsed time of the lap's frames.
    """

    lap_number: int
    frames: list[TelemetryFrame]

    lap_time_s: float | None = None
    sector_times_s: list[float] = field(default_factory=list)

    is_clean: bool | None = None      # None = unknown, True = no off-track detected
    is_complete: bool = False         # crossed start-finish at both ends

    avg_speed_kph: float | None = None
    max_speed_kph: float | None = None
    min_speed_kph: float | None = None

    fuel_used_kg: float | None = None
    fuel_start_kg: float | None = None
    fuel_end_kg: float | None = None

    tyre_wear_delta_pct: dict[str, float] | None = None
    tyre_temp_avg_c: dict[str, float] | None = None

    compound: str | None = None


# Heuristic thresholds. Conservative — better to call a lap dirty than miss
# an off-track when judging consistency. Threshold is rate-per-second so
# the heuristic is sample-rate agnostic; 200 kph/s is well above the ~180
# kph/s peak F1 braking rate but well below an off-track-driven discontinuity.
_OFFTRACK_DECEL_KPH_PER_S = 250.0
_OFFTRACK_MIN_SPEED_KPH = 50.0    # below this, ignore (pit/start)


def split_into_laps(frames: list[TelemetryFrame]) -> list[Lap]:
    """Split a flat frame stream into per-lap groups.

    Frames must be sorted by `timestamp_s` (the storage layer guarantees
    this). Frames with `lap_number <= 0` are treated as out-lap and
    grouped under their own pseudo-lap if any are present.
    """
    if not frames:
        return []

    laps: list[Lap] = []
    current: list[TelemetryFrame] = [frames[0]]

    for f in frames[1:]:
        if f.lap_number != current[-1].lap_number:
            laps.append(_finalize_lap(current))
            current = [f]
        else:
            current.append(f)
    laps.append(_finalize_lap(current))

    # Cross-reference last_lap_time_s from the *next* lap to fill in lap_time_s.
    for i, lap in enumerate(laps):
        if lap.lap_time_s is not None:
            continue
        if i + 1 < len(laps):
            nxt = laps[i + 1]
            # The first few frames of the next lap should carry last_lap_time_s.
            for f in nxt.frames[:30]:
                if f.last_lap_time_s and f.last_lap_time_s > 0:
                    lap.lap_time_s = f.last_lap_time_s
                    lap.is_complete = True
                    break

    return laps


def _finalize_lap(frames: list[TelemetryFrame]) -> Lap:
    lap = Lap(lap_number=frames[0].lap_number, frames=frames)

    speeds = [f.speed_kph for f in frames if f.speed_kph is not None]
    if speeds:
        lap.avg_speed_kph = sum(speeds) / len(speeds)
        lap.max_speed_kph = max(speeds)
        lap.min_speed_kph = min(speeds)

    fuels = [f.fuel_kg for f in frames if f.fuel_kg is not None]
    if fuels:
        lap.fuel_start_kg = fuels[0]
        lap.fuel_end_kg = fuels[-1]
        lap.fuel_used_kg = max(0.0, fuels[0] - fuels[-1])

    lap.tyre_wear_delta_pct = _wear_delta(frames)
    lap.tyre_temp_avg_c = _avg_tyre_temp(frames)

    for f in frames:
        if f.tyre_compound:
            lap.compound = f.tyre_compound
            break

    lap.is_clean = _is_clean(frames)
    return lap


def _wear_delta(frames: list[TelemetryFrame]) -> dict[str, float] | None:
    start = next((f.tyre_wear_pct for f in frames if f.tyre_wear_pct), None)
    end = next((f.tyre_wear_pct for f in reversed(frames) if f.tyre_wear_pct), None)
    if not start or not end:
        return None
    return {k: end.get(k, 0.0) - start.get(k, 0.0) for k in start}


def _avg_tyre_temp(frames: list[TelemetryFrame]) -> dict[str, float] | None:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for f in frames:
        if not f.tyre_temp_c:
            continue
        for k, v in f.tyre_temp_c.items():
            sums[k] = sums.get(k, 0.0) + v
            counts[k] = counts.get(k, 0) + 1
    if not counts:
        return None
    return {k: sums[k] / counts[k] for k in sums}


def _is_clean(frames: list[TelemetryFrame]) -> bool | None:
    """True if no instantaneous deceleration exceeds _OFFTRACK_DECEL_KPH_PER_S
    above the min-speed gate. Rate-based (kph/s) so this works across sample
    rates — F1's ~4 Hz vs ACC's ~30 Hz both behave consistently.
    None if speed not available.
    """
    if all(f.speed_kph is None for f in frames):
        return None
    prev_s: float | None = None
    prev_t: float | None = None
    for f in frames:
        s = f.speed_kph
        t = f.timestamp_s
        if s is None:
            prev_s = None
            prev_t = None
            continue
        if prev_s is not None and prev_t is not None and prev_s > _OFFTRACK_MIN_SPEED_KPH:
            dt = t - prev_t
            if dt > 0:
                decel = (prev_s - s) / dt
                if decel > _OFFTRACK_DECEL_KPH_PER_S:
                    return False
        prev_s = s
        prev_t = t
    return True
