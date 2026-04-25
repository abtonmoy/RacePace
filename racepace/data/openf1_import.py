"""Import a real F1 session from the OpenF1 REST API.

OpenF1 is a free public API at api.openf1.org. Endpoints used:
- /v1/sessions  — session metadata (track, year, type)
- /v1/drivers   — team, name code
- /v1/laps      — per-lap durations + sector splits
- /v1/car_data  — speed/throttle/brake(0|100)/gear/rpm/drs at ~4Hz
- /v1/location  — X/Y/Z (track-local metres) at ~4Hz
- /v1/intervals — gap to leader and interval (gap to car ahead) per timestamp

Distance is NOT exposed by OpenF1, so we derive it per-lap by integrating
sqrt(dx²+dy²) along the location trace. That gives a reasonable
`lap_distance_m` and `lap_distance_pct`.

Steering is NOT exposed either; we derive a heading-delta proxy from X/Y,
same as the FastF1 importer. Brake is boolean (0|100). Fuel and tyre
wear/temps/pressures are not in the broadcast — None on every frame.
"""

from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from racepace.schema import SessionInfo, TelemetryFrame
from racepace.storage.session_store import SessionWriter


_BASE = "https://api.openf1.org/v1"


def _get(path: str, **params) -> list[dict]:
    """GET a JSON list from OpenF1. Raises on non-2xx.

    OpenF1 filter keys carry the operator: e.g. `date>=` is a key meaning
    "date greater-than-or-equal". We recognize keys ending in an operator
    char and skip adding an extra `=`. Date string special chars are kept
    literal — OpenF1 doesn't tolerate `+` percent-encoded as `%2B` here.
    """
    parts = []
    for k, v in params.items():
        if v is None:
            continue
        sep = "" if (k.endswith("=") or k.endswith("<") or k.endswith(">")) else "="
        parts.append(f"{k}{sep}{urllib.parse.quote(str(v), safe='<>=:+T,-.')}")
    qs = "&".join(parts)
    url = f"{_BASE}/{path.lstrip('/')}" + (f"?{qs}" if qs else "")
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"OpenF1 {e.code} on {url}") from e


def find_session_key(year: int, country_or_circuit: str, session_name: str = "Race") -> dict:
    """Find one session by year + country/circuit + session_name.

    Prefers exact `location` / `circuit_short_name` match before the broader
    `country_name` match (one country can host two circuits, e.g. Imola + Monza),
    and skips cancelled sessions.
    """
    needle = country_or_circuit.lower()
    sessions = [s for s in _get("sessions", year=year, session_name=session_name) if not s.get("is_cancelled")]
    # Pass 1: exact location / circuit short name match.
    for s in sessions:
        if needle == (s.get("location") or "").lower() or needle == (s.get("circuit_short_name") or "").lower():
            return s
    # Pass 2: substring match on location or circuit.
    for s in sessions:
        if needle in (s.get("location") or "").lower() or needle in (s.get("circuit_short_name") or "").lower():
            return s
    # Pass 3: country.
    for s in sessions:
        if needle in (s.get("country_name") or "").lower():
            return s
    raise RuntimeError(f"No {session_name} session in {year} matching {country_or_circuit!r}")


def import_session(
    year: int,
    country_or_circuit: str,
    session_name: str,
    driver_number: int,
    output_db: str | Path,
    lap_range: tuple[int, int] | None = None,
    car_label: str | None = None,
) -> tuple[str, int, int]:
    """Pull an OpenF1 session and write it to a RacePace session DB.

    Returns (session_id, frames_written, laps_written).
    """
    s = find_session_key(year, country_or_circuit, session_name)
    session_key = s["session_key"]

    drivers = _get("drivers", session_key=session_key, driver_number=driver_number)
    driver = drivers[0] if drivers else {}
    team = driver.get("team_name") or car_label
    name_acronym = driver.get("name_acronym")

    laps_raw = _get("laps", session_key=session_key, driver_number=driver_number)
    laps_raw = [l for l in laps_raw if l.get("date_start") and l.get("lap_duration")]
    laps_raw.sort(key=lambda l: l["lap_number"])
    if lap_range is not None:
        lo, hi = lap_range
        laps_raw = [l for l in laps_raw if lo <= l["lap_number"] <= hi]
    if not laps_raw:
        raise RuntimeError(f"No usable laps for driver {driver_number} in {year} {country_or_circuit} {session_name}")

    # Pull car_data + location in one shot covering all selected laps.
    win_start = laps_raw[0]["date_start"]
    last_lap = laps_raw[-1]
    win_end = _add_seconds_iso(last_lap["date_start"], float(last_lap["lap_duration"]) + 5.0)
    print(f"  pulling car_data + location {win_start} → {win_end}")
    car = _get(
        "car_data",
        session_key=session_key,
        driver_number=driver_number,
        **{"date>=": win_start, "date<=": win_end},
    )
    loc = _get(
        "location",
        session_key=session_key,
        driver_number=driver_number,
        **{"date>=": win_start, "date<=": win_end},
    )
    intervals = _get(
        "intervals",
        session_key=session_key,
        driver_number=driver_number,
        **{"date>=": win_start, "date<=": win_end},
    )
    # /v1/position only records position *changes* — pull ALL of them for
    # this driver so we have the seed value (typically the grid position
    # set at session_start, well before our race window).
    positions = _get(
        "position",
        session_key=session_key,
        driver_number=driver_number,
    )

    # Convert to (epoch_seconds, payload) tuples for fast time-based lookup.
    car_t = [(_iso_to_epoch(c["date"]), c) for c in car]
    loc_t = [(_iso_to_epoch(l["date"]), l) for l in loc]
    int_t = [(_iso_to_epoch(i["date"]), i) for i in intervals]
    pos_t = [(_iso_to_epoch(p["date"]), p) for p in positions]
    car_t.sort(); loc_t.sort(); int_t.sort(); pos_t.sort()

    # Find best lap for best_lap_time_s
    best_lap_time_s = min((float(l["lap_duration"]) for l in laps_raw if l.get("lap_duration")), default=None)

    track_name = (s.get("location") or s.get("circuit_short_name") or "unknown").lower().replace(" ", "_")
    session_id = str(uuid.uuid4())
    info = SessionInfo(
        session_id=session_id,
        sim="f1",
        started_at=_iso_to_dt(s["date_start"]),
        track=track_name,
        car=team,
        session_type=_session_type_label(session_name),
        weather="dry",
        total_laps=len(laps_raw),
    )

    frames_written = 0
    laps_written = 0
    t0_session: float | None = None
    last_lap_time_s: float | None = None

    with SessionWriter(output_db) as w:
        w.write_session(info)
        for lap in laps_raw:
            lap_number = int(lap["lap_number"])
            lap_start_t = _iso_to_epoch(lap["date_start"])
            lap_dur = float(lap["lap_duration"])
            lap_end_t = lap_start_t + lap_dur
            sector_1 = _safe_float(lap.get("duration_sector_1"))
            sector_2 = _safe_float(lap.get("duration_sector_2"))

            if t0_session is None:
                t0_session = lap_start_t

            # Slice car/loc samples that fall in this lap.
            car_in = [c for t, c in car_t if lap_start_t <= t <= lap_end_t]
            loc_in = [l for t, l in loc_t if lap_start_t <= t <= lap_end_t]
            if not car_in:
                continue

            # Build a per-sample distance trace from location.
            loc_xs = np.array([l["x"] for l in loc_in], dtype=float) if loc_in else np.array([])
            loc_ys = np.array([l["y"] for l in loc_in], dtype=float) if loc_in else np.array([])
            loc_ts = np.array([_iso_to_epoch(l["date"]) for l in loc_in], dtype=float) if loc_in else np.array([])
            distances_loc = _cumulative_distance(loc_xs, loc_ys)
            max_dist = float(distances_loc[-1]) if len(distances_loc) else 0.0
            steering_proxy = _steering_proxy(loc_xs, loc_ys)

            # Compound + tyre age for this lap from /stints (one shot per session would be cleaner,
            # but the OpenF1 stints endpoint expects a date range that covers the lap):
            # Skip per-lap fetch to stay polite — we'll grab nothing for now and rely on extras.

            for c in car_in:
                car_epoch = _iso_to_epoch(c["date"])
                lap_t = car_epoch - lap_start_t
                rel_dist = lap_t / lap_dur if lap_dur > 0 else 0.0
                rel_dist = max(0.0, min(1.0, rel_dist))

                # Find the location sample nearest in time to this car sample.
                lap_distance_m: float | None = None
                steer: float | None = None
                pos_x: float | None = None
                pos_y: float | None = None
                if len(loc_ts):
                    idx = int(np.argmin(np.abs(loc_ts - car_epoch)))
                    lap_distance_m = float(distances_loc[idx])
                    pos_x = float(loc_xs[idx])
                    pos_y = float(loc_ys[idx])
                    if 0 < idx < len(steering_proxy) - 1:
                        s_val = float(steering_proxy[idx])
                        steer = None if math.isnan(s_val) else s_val

                # Sector by elapsed time (more accurate than rel_dist).
                cur_sector = _estimate_sector(lap_t, sector_1, sector_2)

                # Gap to car ahead from /intervals — most-recent-at-or-before
                # this frame's time (intervals samples regularly so closest-in-time
                # would also work, but at-or-before is safe regardless).
                gap_ahead_s = _at_or_before(int_t, car_epoch, "interval")

                # Position from /position — IMPORTANT: this endpoint only
                # records position *changes*, so closest-in-time picks future
                # positions. Use most-recent-at-or-before for correct semantics.
                pos_for_lap = _safe_int(_at_or_before(pos_t, car_epoch, "position"))

                extras: dict[str, Any] = {}
                if c.get("drs") is not None:
                    extras["drs"] = int(c["drs"])
                if pos_x is not None and pos_y is not None:
                    extras["pos_x_m"] = pos_x
                    extras["pos_y_m"] = pos_y

                w.write_frame(TelemetryFrame(
                    timestamp_s=car_epoch - t0_session,
                    lap_number=lap_number,
                    lap_distance_m=lap_distance_m,
                    lap_distance_pct=rel_dist if max_dist > 0 else None,
                    speed_kph=_safe_float(c.get("speed")),
                    throttle_pct=_safe_float(c.get("throttle")),
                    brake_pct=_safe_float(c.get("brake")),
                    gear=_safe_int(c.get("n_gear")),
                    rpm=_safe_float(c.get("rpm")),
                    steering_norm=steer,
                    fuel_kg=None,
                    fuel_laps_remaining=None,
                    tyre_wear_pct=None,
                    tyre_temp_c=None,
                    tyre_pressure_psi=None,
                    tyre_compound=None,  # could pull /stints; left for now
                    position=pos_for_lap,
                    gap_ahead_s=gap_ahead_s,
                    last_lap_time_s=last_lap_time_s,
                    best_lap_time_s=best_lap_time_s,
                    current_sector=cur_sector,
                    extras=extras,
                ))
                frames_written += 1

            last_lap_time_s = lap_dur
            laps_written += 1

        w.update_session_end(
            ended_at=datetime.now(timezone.utc),
            total_laps=laps_written,
        )

    return session_id, frames_written, laps_written


# --- helpers -----------------------------------------------------------------

def _iso_to_epoch(s: str) -> float:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).timestamp()


def _iso_to_dt(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _add_seconds_iso(s: str, secs: float) -> str:
    dt = _iso_to_dt(s)
    return (dt.fromtimestamp(dt.timestamp() + secs, tz=timezone.utc)).isoformat()


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
        return int(v)
    except (TypeError, ValueError):
        return None


def _session_type_label(name: str) -> str | None:
    n = name.lower()
    if "race" in n:
        return "race"
    if "qual" in n or "shootout" in n:
        return "qualifying"
    if "practice" in n or n.startswith("fp") or n.startswith("p"):
        return "practice"
    if "sprint" in n:
        return "race"
    return None


def _at_or_before(records: list[tuple[float, dict]], t: float, field: str):
    """Most recent record at-or-before time `t`. Returns the value of `field`,
    or None if no record qualifies. Records must be sorted by timestamp."""
    if not records:
        return None
    # Binary search for the largest index with ts <= t.
    lo, hi = 0, len(records) - 1
    if records[0][0] > t:
        return None
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if records[mid][0] <= t:
            lo = mid
        else:
            hi = mid - 1
    return records[lo][1].get(field)


def _estimate_sector(lap_t: float, s1: float | None, s2: float | None) -> int | None:
    """Sector index from elapsed time within the lap. Uses real splits when available."""
    if s1 is None:
        return None
    if lap_t <= s1:
        return 0
    if s2 is None:
        return 1
    if lap_t <= s1 + s2:
        return 1
    return 2


def _cumulative_distance(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Integrate the X/Y trajectory to a per-sample arc length in meters.

    OpenF1's /v1/location reports X/Y in decimeters (1/10 m), so we divide
    the integrated distance by 10 to land in meters.
    """
    if len(xs) < 2:
        return np.zeros_like(xs)
    dx = np.diff(xs)
    dy = np.diff(ys)
    seg = np.sqrt(dx * dx + dy * dy) / 10.0   # decimeters → meters
    return np.concatenate(([0.0], np.cumsum(seg)))


def _steering_proxy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan)
    if n < 3:
        return out
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() < 3:
        return out
    headings = np.arctan2(np.diff(y), np.diff(x))
    delta = np.diff(headings)
    delta = (delta + np.pi) % (2 * np.pi) - np.pi
    if np.nanmax(np.abs(delta)) > 0:
        norm = delta / np.nanmax(np.abs(delta))
    else:
        norm = delta
    out[1:-1] = norm
    return out
