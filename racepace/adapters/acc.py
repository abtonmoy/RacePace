"""Assetto Corsa Competizione adapter via shared memory.

ACC publishes three pages of shared memory on Windows:

- **Physics**  — high-frequency dynamics: speed, throttle, brake, gear,
  rpm, steering, fuel, tyre temps/pressures/wear, g-forces.
- **Graphics** — slower, UI-level state: lap number, position, lap times,
  current sector, session type, flag state.
- **Static**   — written once at session start: car model, track name,
  max fuel, sector count, session-level metadata.

This module reads all three each tick via `pyaccsharedmemory` and folds
them into a single normalized `TelemetryFrame`.

Per-field mapping for `TelemetryFrame`
--------------------------------------

- ``timestamp_s``         monotonic clock, sample 0 == 0.0 s
- ``lap_number``          graphics.completedLaps + 1
- ``lap_distance_m``      graphics.normalizedCarPosition * static.trackSPlineLength
- ``lap_distance_pct``    graphics.normalizedCarPosition (0.0–1.0)
- ``speed_kph``           physics.speedKmh (already kph in ACC SHM)
- ``throttle_pct``        physics.gas * 100
- ``brake_pct``           physics.brake * 100
- ``clutch_pct``          (1.0 - physics.clutch) * 100  (ACC reports 1.0=released)
- ``steering_norm``       physics.steerAngle (already -1..+1)
- ``gear``                physics.gear - 1   (ACC: 0=R, 1=N, 2=1st...)
- ``rpm``                 physics.rpms
- ``fuel_kg``             physics.fuel  (ACC reports litres; converted with
                          car-specific density — defaults to 0.75 kg/L for
                          racing fuel; override per car if needed)
- ``fuel_laps_remaining`` graphics.fuelXLap > 0 ? fuel_kg / (graphics.fuelXLap * 0.75) : None
- ``tyre_wear_pct``       100 - physics.tyreWear[i] (ACC reports remaining %, we report worn %)
- ``tyre_temp_c``         physics.tyreCoreTemperature[i]   (FL,FR,RL,RR)
- ``tyre_pressure_psi``   physics.wheelsPressure[i]   (ACC SHM reports psi already)
- ``tyre_compound``       graphics.tyreCompound  ("dry"/"wet")
- ``position``            graphics.position
- ``gap_ahead_s``         **unavailable** in shared memory; broadcast API only
- ``gap_behind_s``        **unavailable** in shared memory; broadcast API only
- ``last_lap_time_s``     graphics.iLastTime / 1000.0   (ACC reports ms)
- ``best_lap_time_s``     graphics.iBestTime / 1000.0
- ``current_sector``      graphics.currentSectorIndex
- ``extras``              g-forces, brake bias, TC/ABS levels, flag state,
                          air/track temp, surface grip — see _extract_extras

Fields ACC does NOT expose via SHM (use UDP broadcast in a future phase):
  - precise inter-car gaps (gap_ahead_s, gap_behind_s)
  - opponent lap times / sector splits
  - safety car / VSC state with leadtime

Platform: shared memory is Windows-only. On macOS/Linux, use MockAdapter.
"""

from __future__ import annotations

import platform
import time
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone

from racepace.adapters.base import AbstractAdapter, RateLimiter
from racepace.schema import SessionInfo, TelemetryFrame


_LITRES_TO_KG = 0.75   # racing-fuel density; close enough for all road/GT cars

_TYRE_KEYS = ("fl", "fr", "rl", "rr")

# ACC session-type codes (graphics.session)
_SESSION_TYPE_MAP = {
    0: "practice",
    1: "qualifying",
    2: "race",
    3: "hotlap",
    4: "hotlap",   # time attack
    5: "hotlap",   # drift
    6: "hotlap",   # drag
}


class AccAdapter(AbstractAdapter):
    sim_name = "acc"

    def __init__(self, target_hz: float = 30.0) -> None:
        self.target_hz = target_hz
        self._reader = None
        self._limiter = RateLimiter(self.target_hz)
        self._t0_monotonic: float | None = None
        self._session_id = str(uuid.uuid4())

    def connect(self) -> None:
        if platform.system() != "Windows":
            raise RuntimeError(
                "ACC shared memory is Windows-only. "
                "On macOS/Linux, replay a recorded session with MockAdapter."
            )
        # Imported lazily so the package is installable on non-Windows hosts.
        from pyaccsharedmemory import accSharedMemory  # type: ignore[import-not-found]

        self._reader = accSharedMemory()
        self._t0_monotonic = time.monotonic()

    def disconnect(self) -> None:
        if self._reader is not None:
            try:
                self._reader.close()
            except Exception:
                pass
            self._reader = None

    def read_session_info(self) -> SessionInfo:
        if self._reader is None:
            raise RuntimeError("connect() before read_session_info()")
        sm = self._reader.read_shared_memory()
        static = sm.Static
        graphics = sm.Graphics
        return SessionInfo(
            session_id=self._session_id,
            sim=self.sim_name,
            started_at=datetime.now(timezone.utc),
            track=getattr(static, "track", None),
            car=getattr(static, "car_model", None),
            session_type=_SESSION_TYPE_MAP.get(int(getattr(graphics, "session", -1))),
            air_temp_c=getattr(static, "air_temp", None),
            track_temp_c=getattr(static, "road_temp", None),
        )

    def stream_frames(self) -> Iterator[TelemetryFrame]:
        if self._reader is None:
            raise RuntimeError("connect() before stream_frames()")
        if self._t0_monotonic is None:
            self._t0_monotonic = time.monotonic()

        while True:
            sm = self._reader.read_shared_memory()
            if sm is None:
                # Source paused (game in menu); back off briefly.
                time.sleep(0.05)
                continue
            if not self._limiter.should_emit():
                continue
            frame = self._build_frame(sm)
            if frame is not None:
                yield frame

    def _build_frame(self, sm) -> TelemetryFrame | None:
        physics = sm.Physics
        graphics = sm.Graphics
        static = sm.Static
        if physics is None or graphics is None:
            return None

        ts = time.monotonic() - (self._t0_monotonic or time.monotonic())

        spline = _safe_float(getattr(graphics, "normalized_car_position", None))
        track_len = _safe_float(getattr(static, "track_spline_length", None))

        fuel_litres = _safe_float(getattr(physics, "fuel", None))
        fuel_per_lap_l = _safe_float(getattr(graphics, "fuel_per_lap", None))
        fuel_kg = fuel_litres * _LITRES_TO_KG if fuel_litres is not None else None
        if fuel_litres is not None and fuel_per_lap_l and fuel_per_lap_l > 0:
            fuel_laps_remaining = fuel_litres / fuel_per_lap_l
        else:
            fuel_laps_remaining = None

        tyre_temp_arr = getattr(physics, "tyre_core_temp", None)
        tyre_press_arr = getattr(physics, "wheel_pressure", None)
        tyre_wear_arr = getattr(physics, "tyre_wear", None)

        tyre_temp = _quad_to_dict(tyre_temp_arr)
        tyre_press = _quad_to_dict(tyre_press_arr)
        # ACC reports remaining tread; convert to worn %
        tyre_wear_remaining = _quad_to_dict(tyre_wear_arr)
        tyre_wear = (
            {k: max(0.0, 100.0 - v) for k, v in tyre_wear_remaining.items()}
            if tyre_wear_remaining is not None
            else None
        )

        gear_raw = _safe_int(getattr(physics, "gear", None))
        gear = (gear_raw - 1) if gear_raw is not None else None

        clutch_raw = _safe_float(getattr(physics, "clutch", None))
        clutch_pct = (1.0 - clutch_raw) * 100.0 if clutch_raw is not None else None

        last_ms = _safe_float(getattr(graphics, "last_time", None))
        best_ms = _safe_float(getattr(graphics, "best_time", None))

        return TelemetryFrame(
            timestamp_s=ts,
            lap_number=(_safe_int(getattr(graphics, "completed_lap", 0)) or 0) + 1,
            lap_distance_m=(spline * track_len) if (spline is not None and track_len) else None,
            lap_distance_pct=spline,
            speed_kph=_safe_float(getattr(physics, "speed_kmh", None)),
            throttle_pct=_pct(getattr(physics, "gas", None)),
            brake_pct=_pct(getattr(physics, "brake", None)),
            clutch_pct=clutch_pct,
            steering_norm=_safe_float(getattr(physics, "steer_angle", None)),
            gear=gear,
            rpm=_safe_float(getattr(physics, "rpm", None)),
            fuel_kg=fuel_kg,
            fuel_laps_remaining=fuel_laps_remaining,
            tyre_wear_pct=tyre_wear,
            tyre_temp_c=tyre_temp,
            tyre_pressure_psi=tyre_press,
            tyre_compound=getattr(graphics, "tyre_compound", None),
            position=_safe_int(getattr(graphics, "position", None)),
            gap_ahead_s=None,
            gap_behind_s=None,
            last_lap_time_s=(last_ms / 1000.0) if last_ms and last_ms > 0 else None,
            best_lap_time_s=(best_ms / 1000.0) if best_ms and best_ms > 0 else None,
            current_sector=_safe_int(getattr(graphics, "current_sector_index", None)),
            extras=_extract_extras(physics, graphics, static),
        )


def _safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f


def _safe_int(v) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _pct(v) -> float | None:
    f = _safe_float(v)
    return f * 100.0 if f is not None else None


def _quad_to_dict(arr) -> dict[str, float] | None:
    """Map a 4-element FL/FR/RL/RR array (or .fl/.fr/.rl/.rr object) to a dict."""
    if arr is None:
        return None
    # Support both indexable arrays and pyaccsharedmemory's Wheels objects.
    if hasattr(arr, "fl"):
        try:
            return {
                "fl": float(arr.fl),
                "fr": float(arr.fr),
                "rl": float(arr.rl),
                "rr": float(arr.rr),
            }
        except (TypeError, ValueError, AttributeError):
            return None
    try:
        return {k: float(arr[i]) for i, k in enumerate(_TYRE_KEYS)}
    except (TypeError, ValueError, IndexError):
        return None


def _extract_extras(physics, graphics, static) -> dict:
    """Sim-specific richness that doesn't fit the normalized schema."""
    out: dict = {}
    for src, prefix, fields in (
        (physics, "physics", ("g_force", "brake_bias", "tc", "abs", "tc_in_action", "abs_in_action")),
        (graphics, "graphics", ("flag", "session_time_left", "rain_intensity", "wind_speed")),
        (static, "static", ("max_fuel", "sector_count", "max_rpm", "player_name")),
    ):
        if src is None:
            continue
        for f in fields:
            v = getattr(src, f, None)
            if v is None:
                continue
            try:
                # Coerce to JSON-safe primitives.
                if hasattr(v, "x") and hasattr(v, "y") and hasattr(v, "z"):
                    out[f"{prefix}.{f}"] = [float(v.x), float(v.y), float(v.z)]
                elif isinstance(v, (int, float, str, bool)):
                    out[f"{prefix}.{f}"] = v
                else:
                    out[f"{prefix}.{f}"] = float(v)
            except (TypeError, ValueError):
                continue
    return out
