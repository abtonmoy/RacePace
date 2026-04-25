"""Import a real F1 session from FastF1 into a RacePace SQLite session DB.

Maps each FastF1 telemetry sample → TelemetryFrame.

What FastF1 gives us (per sample): SessionTime, Distance, RelativeDistance,
Speed (kph), Throttle (0-100), Brake (bool), nGear, RPM, DRS, X/Y/Z (m).

Per-lap (constant across frames in that lap): LapTime, LapNumber, Compound,
TyreLife, Position, Sector1Time/Sector2Time/Sector3Time.

What F1 broadcast does NOT give us:
- Fuel mass/laps remaining   → frame.fuel_kg = None
- Tyre wear per corner        → frame.tyre_wear_pct = None  (we expose TyreLife in extras)
- Tyre temps / pressures      → None
- Steering wheel position     → derived as a proxy from X/Y heading deltas
- Brake pressure              → boolean → 0.0 or 100.0
- Per-frame gap to ahead/behind → DistanceToDriverAhead is in metres, not seconds; left in extras

The Phase 1 contract (missing data is None, not zero) means the agents
already handle this gracefully — the engineer's fuel triggers stay silent,
the analyst says "not reported" for missing fields.
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from racepace.schema import SessionInfo, TelemetryFrame
from racepace.storage.session_store import SessionWriter


def import_session(
    year: int,
    gp: str,
    session_type: str,
    driver: str,
    output_db: str | Path,
    lap_range: tuple[int, int] | None = None,
    cache_dir: str | Path = ".fastf1_cache",
    car_label: str | None = None,
) -> tuple[str, int, int]:
    """Pull a session from FastF1 and write it to a RacePace session DB.

    Returns (session_id, frames_written, laps_written).
    """
    import fastf1   # imported here so importing this module is cheap

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_path))

    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=True, weather=False, messages=False)

    laps = session.laps.pick_drivers(driver)
    if lap_range is not None:
        lo, hi = lap_range
        laps = laps[(laps["LapNumber"] >= lo) & (laps["LapNumber"] <= hi)]
    if len(laps) == 0:
        raise RuntimeError(f"No laps for {driver} in {year} {gp} {session_type} (range={lap_range}).")

    # Find a per-session "best lap time" from the full session.
    all_laps = session.laps.pick_drivers(driver)
    fastest = all_laps.pick_fastest()
    best_lap_s = (
        fastest["LapTime"].total_seconds()
        if fastest is not None and pd.notna(fastest.get("LapTime"))
        else None
    )

    track_name = _safe_track_name(session)
    car_name = car_label or _team_name(laps)
    session_id = str(uuid.uuid4())

    info = SessionInfo(
        session_id=session_id,
        sim="f1",
        started_at=_session_start_dt(session),
        track=track_name,
        car=car_name,
        session_type=_session_type_label(session_type),
        weather="dry",  # FastF1 has weather frames; we keep it simple here
        total_laps=int(session.total_laps) if hasattr(session, "total_laps") and session.total_laps else None,
    )

    frames_written = 0
    laps_written = 0
    t0 = None
    last_lap_time_s: float | None = None
    pos_for_lap: int | None = None

    with SessionWriter(output_db) as w:
        w.write_session(info)
        for _, lap in laps.iterlaps():
            try:
                tel = lap.get_car_data().add_distance()
            except Exception:
                continue
            if len(tel) == 0:
                continue

            # Add position trace (X, Y) from pos_data; align by SessionTime.
            try:
                pos = lap.get_pos_data()
                merged = tel.copy()
                merged["X"] = np.interp(
                    tel["SessionTime"].dt.total_seconds(),
                    pos["SessionTime"].dt.total_seconds(),
                    pos["X"].astype(float),
                )
                merged["Y"] = np.interp(
                    tel["SessionTime"].dt.total_seconds(),
                    pos["SessionTime"].dt.total_seconds(),
                    pos["Y"].astype(float),
                )
            except Exception:
                merged = tel.copy()
                merged["X"] = np.nan
                merged["Y"] = np.nan

            steering_proxy = _steering_proxy(merged.get("X").to_numpy(), merged.get("Y").to_numpy())

            # Compute rel_dist from Distance — get_car_data().add_distance() doesn't
            # add a RelativeDistance column, so derive it ourselves.
            try:
                _max_dist = float(merged["Distance"].max())
            except Exception:
                _max_dist = 0.0

            lap_number = int(lap.get("LapNumber", laps_written + 1))
            lap_time_obj = lap.get("LapTime")
            this_lap_time_s = lap_time_obj.total_seconds() if pd.notna(lap_time_obj) else None
            sector1 = lap.get("Sector1Time")
            sector2 = lap.get("Sector2Time")
            sector1_s = sector1.total_seconds() if pd.notna(sector1) else None
            sector2_s = sector2.total_seconds() if pd.notna(sector2) else None
            compound = lap.get("Compound")
            if isinstance(compound, str):
                compound = compound.lower()
            tyre_life = lap.get("TyreLife")
            try:
                pos_for_lap = int(lap.get("Position")) if pd.notna(lap.get("Position")) else pos_for_lap
            except Exception:
                pass

            for i, row in enumerate(merged.itertuples(index=False)):
                row_t = float(row.SessionTime.total_seconds())
                if t0 is None:
                    t0 = row_t
                ts = row_t - t0

                lap_distance_m = _safe_float(getattr(row, "Distance", None))
                rel_dist = _safe_float(getattr(row, "RelativeDistance", None))
                if rel_dist is None and lap_distance_m is not None and _max_dist > 0:
                    rel_dist = lap_distance_m / _max_dist
                speed_kph = _safe_float(getattr(row, "Speed", None))
                throttle_pct = _safe_float(getattr(row, "Throttle", None))
                brake_raw = getattr(row, "Brake", None)
                brake_pct = 100.0 if bool(brake_raw) else 0.0
                gear = _safe_int(getattr(row, "nGear", None))
                rpm = _safe_float(getattr(row, "RPM", None))
                drs = _safe_int(getattr(row, "DRS", None))

                # Sector estimate from elapsed time inside the lap, given we have lap time + sector splits.
                cur_sector = _estimate_sector(rel_dist, sector1_s, sector2_s, this_lap_time_s)

                extras: dict = {}
                if drs is not None:
                    extras["drs"] = drs
                if tyre_life is not None and pd.notna(tyre_life):
                    extras["tyre_age_laps"] = int(tyre_life)
                if hasattr(row, "X") and hasattr(row, "Y") and not (math.isnan(row.X) or math.isnan(row.Y)):
                    extras["pos_x_m"] = float(row.X)
                    extras["pos_y_m"] = float(row.Y)

                steering = float(steering_proxy[i]) if i < len(steering_proxy) else None
                if steering is not None and math.isnan(steering):
                    steering = None

                w.write_frame(TelemetryFrame(
                    timestamp_s=ts,
                    lap_number=lap_number,
                    lap_distance_m=lap_distance_m,
                    lap_distance_pct=rel_dist,
                    speed_kph=speed_kph,
                    throttle_pct=throttle_pct,
                    brake_pct=brake_pct,
                    gear=gear,
                    rpm=rpm,
                    steering_norm=steering,
                    fuel_kg=None,
                    fuel_laps_remaining=None,
                    tyre_wear_pct=None,
                    tyre_temp_c=None,
                    tyre_pressure_psi=None,
                    tyre_compound=compound if isinstance(compound, str) else None,
                    position=pos_for_lap,
                    last_lap_time_s=last_lap_time_s,
                    best_lap_time_s=best_lap_s,
                    current_sector=cur_sector,
                    extras=extras,
                ))
                frames_written += 1

            last_lap_time_s = this_lap_time_s
            laps_written += 1

        w.update_session_end(
            ended_at=datetime.now(timezone.utc),
            total_laps=laps_written,
        )

    return session_id, frames_written, laps_written


# --- helpers -----------------------------------------------------------------

def _safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f


def _safe_int(v) -> int | None:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def _safe_track_name(session) -> str | None:
    for attr in ("event", "weekend"):
        ev = getattr(session, attr, None)
        if ev is not None:
            for k in ("Location", "EventName", "OfficialEventName"):
                v = getattr(ev, k, None) or (ev[k] if hasattr(ev, "__getitem__") and k in ev else None)
                if v:
                    return str(v).lower().replace(" ", "_")
    return None


def _team_name(laps) -> str | None:
    if "Team" in laps.columns:
        teams = laps["Team"].dropna().unique()
        if len(teams):
            return str(teams[0])
    return None


def _session_start_dt(session) -> datetime:
    for attr in ("session_start_time", "date"):
        v = getattr(session, attr, None)
        if v is not None:
            try:
                if hasattr(v, "to_pydatetime"):
                    v = v.to_pydatetime()
                if v.tzinfo is None:
                    v = v.replace(tzinfo=timezone.utc)
                return v
            except Exception:
                continue
    return datetime.now(timezone.utc)


def _session_type_label(s: str) -> str | None:
    s = s.lower()
    if s in ("race", "r"):
        return "race"
    if s in ("q", "qualifying", "sq", "sprint qualifying"):
        return "qualifying"
    if s in ("fp1", "fp2", "fp3", "p1", "p2", "p3", "practice"):
        return "practice"
    if s in ("sprint", "ss"):
        return "race"
    return None


def _estimate_sector(
    rel_dist: float | None,
    sector1_s: float | None,
    sector2_s: float | None,
    lap_time_s: float | None,
) -> int | None:
    """Approximate sector index from RelativeDistance — F1 sectors are
    distance-based, so this is close to ground truth for clean laps."""
    if rel_dist is None:
        return None
    if rel_dist < 1.0 / 3.0:
        return 0
    if rel_dist < 2.0 / 3.0:
        return 1
    return 2


def _steering_proxy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Approximate steering as the change in heading along the X/Y trajectory.

    Negative = turning left, positive = right. Normalized by max |value| in
    the lap to fit roughly in [-1, +1]. Returns NaNs where either coord is
    missing.
    """
    n = len(x)
    out = np.full(n, np.nan)
    if n < 3:
        return out
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() < 3:
        return out
    headings = np.arctan2(np.diff(y), np.diff(x))
    # Wrap heading delta into [-pi, pi]
    delta = np.diff(headings)
    delta = (delta + np.pi) % (2 * np.pi) - np.pi
    if np.nanmax(np.abs(delta)) > 0:
        norm = delta / np.nanmax(np.abs(delta))
    else:
        norm = delta
    # Pad with NaNs at start and end so length matches n
    out[1:-1] = norm
    return out
