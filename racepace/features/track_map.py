"""Corner extraction from a clean reference lap.

Algorithm:
1. Resample the lap onto a uniform 1m grid by lap_distance_m.
2. Find local minima in speed (apex candidates) using a windowed scan.
3. For each apex, walk backward until both throttle was 100% and brake 0%
   (the brake point) — gives flat corners a brake_point_m == apex_m.
4. Walk forward from the apex until throttle returns to 100% (exit point).
5. Direction from steering_norm sign at the apex (negative = left).
6. Severity 1 (flat-out near-corner) through 6 (heavy braking, low gear).

The output `TrackMap` is JSON-serializable and stored under
`racepace/tracks/{sim}/{track}.json`. After auto-extraction, hand-edit
to add `name` and `notes` fields per corner.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from racepace.features.laps import Lap


# Tunables — exposed so callers can experiment per car.
APEX_LOCAL_WINDOW_M = 80.0      # ± this distance window for local-min check
APEX_MIN_PROMINENCE_KPH = 8.0    # apex must be this much slower than the surrounding window
APEX_MIN_GAP_M = 60.0            # corners closer than this collapse into one apex
THROTTLE_OPEN_PCT = 95.0         # "throttle == 100%" tolerance
BRAKE_OFF_PCT = 5.0              # "brake == 0%" tolerance


@dataclass
class Corner:
    id: int
    name: str | None
    direction: Literal["L", "R"]
    severity: int
    brake_point_m: float
    apex_m: float
    exit_m: float
    target_min_speed_kph: float
    target_gear: int | None
    flat: bool
    notes: str | None = None


@dataclass
class TrackMap:
    track: str
    sim: str
    car: str | None
    total_length_m: float
    corners: list[Corner] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "track": self.track,
                "sim": self.sim,
                "car": self.car,
                "total_length_m": self.total_length_m,
                "corners": [asdict(c) for c in self.corners],
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, s: str) -> "TrackMap":
        d = json.loads(s)
        return cls(
            track=d["track"],
            sim=d["sim"],
            car=d.get("car"),
            total_length_m=d["total_length_m"],
            corners=[Corner(**c) for c in d.get("corners", [])],
        )

    @classmethod
    def load(cls, path: str | Path) -> "TrackMap":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.to_json(), encoding="utf-8")


def extract_track_map(
    lap: Lap,
    track: str,
    sim: str,
    car: str | None = None,
    grid_m: float = 1.0,
) -> TrackMap:
    """Extract a TrackMap from a single reference lap.

    The lap should be clean and at competitive pace. Drivers should
    re-extract whenever the racing line they want coached changes.
    """
    grid = _resample_to_grid(lap, grid_m=grid_m)
    if grid is None:
        return TrackMap(track=track, sim=sim, car=car, total_length_m=0.0, corners=[])

    distance, speed, throttle, brake, steering, gear = grid
    apex_idxs = _find_apex_indices(distance, speed)
    corners: list[Corner] = []
    for i, idx in enumerate(apex_idxs):
        bp_idx = _walk_back_to_brake_point(idx, throttle, brake)
        ex_idx = _walk_forward_to_exit(idx, throttle)
        steer_at_apex = steering[idx] if not np.isnan(steering[idx]) else 0.0
        direction: Literal["L", "R"] = "L" if steer_at_apex < 0 else "R"
        flat = bool(np.nanmax(brake[bp_idx:idx + 1]) < BRAKE_OFF_PCT) if idx > bp_idx else True
        sev = _severity(speed[idx], np.nanmax(brake[bp_idx:idx + 1]) if idx > bp_idx else 0.0)
        target_gear = int(gear[idx]) if not np.isnan(gear[idx]) else None
        corners.append(Corner(
            id=i + 1,
            name=None,
            direction=direction,
            severity=sev,
            brake_point_m=float(distance[bp_idx]),
            apex_m=float(distance[idx]),
            exit_m=float(distance[ex_idx]),
            target_min_speed_kph=float(speed[idx]),
            target_gear=target_gear,
            flat=flat,
            notes=None,
        ))

    total_length = float(distance[-1] - distance[0])
    return TrackMap(track=track, sim=sim, car=car, total_length_m=total_length, corners=corners)


# --- internals ---------------------------------------------------------------

def _resample_to_grid(lap: Lap, grid_m: float):
    """Returns (distance, speed, throttle, brake, steering, gear) numpy arrays
    on a uniform 1m grid. None if the lap lacks distance data."""
    pts = [
        (
            f.lap_distance_m,
            f.speed_kph if f.speed_kph is not None else float("nan"),
            f.throttle_pct if f.throttle_pct is not None else float("nan"),
            f.brake_pct if f.brake_pct is not None else float("nan"),
            f.steering_norm if f.steering_norm is not None else float("nan"),
            f.gear if f.gear is not None else float("nan"),
        )
        for f in lap.frames if f.lap_distance_m is not None
    ]
    if len(pts) < 5:
        return None
    pts.sort(key=lambda p: p[0])
    deduped = [pts[0]]
    for p in pts[1:]:
        if p[0] > deduped[-1][0]:
            deduped.append(p)
    if len(deduped) < 5:
        return None

    raw_d = np.array([p[0] for p in deduped], dtype=float)
    raw_s = np.array([p[1] for p in deduped], dtype=float)
    raw_t = np.array([p[2] for p in deduped], dtype=float)
    raw_b = np.array([p[3] for p in deduped], dtype=float)
    raw_st = np.array([p[4] for p in deduped], dtype=float)
    raw_g = np.array([p[5] for p in deduped], dtype=float)

    d_min, d_max = raw_d[0], raw_d[-1]
    grid = np.arange(d_min, d_max, grid_m)
    if len(grid) < 5:
        return None

    def interp(values: np.ndarray) -> np.ndarray:
        valid = ~np.isnan(values)
        if valid.sum() < 2:
            return np.full_like(grid, np.nan, dtype=float)
        return np.interp(grid, raw_d[valid], values[valid])

    return grid, interp(raw_s), interp(raw_t), interp(raw_b), interp(raw_st), interp(raw_g)


def _find_apex_indices(distance: np.ndarray, speed: np.ndarray) -> list[int]:
    """Local minima with prominence filtering and a minimum gap between apexes."""
    n = len(speed)
    if n < 5:
        return []
    grid_m = float(distance[1] - distance[0]) if n >= 2 else 1.0
    half_window = max(1, int(APEX_LOCAL_WINDOW_M / grid_m))
    min_gap_idx = max(1, int(APEX_MIN_GAP_M / grid_m))

    candidates: list[tuple[int, float]] = []   # (idx, prominence)
    for i in range(half_window, n - half_window):
        window = speed[i - half_window:i + half_window + 1]
        if np.isnan(window).any():
            continue
        if speed[i] != window.min():
            continue
        prominence = float(window.max() - speed[i])
        if prominence < APEX_MIN_PROMINENCE_KPH:
            continue
        candidates.append((i, prominence))

    if not candidates:
        return []

    # Greedy: sort by prominence, accept in order, reject any candidate within min_gap of an accepted one.
    candidates.sort(key=lambda x: -x[1])
    accepted: list[int] = []
    for idx, _ in candidates:
        if all(abs(idx - a) >= min_gap_idx for a in accepted):
            accepted.append(idx)
    accepted.sort()
    return accepted


def _walk_back_to_brake_point(apex_idx: int, throttle: np.ndarray, brake: np.ndarray) -> int:
    """The earliest contiguous index before apex where throttle was 100% and brake 0%.

    For a flat corner (brake never applied), this returns apex_idx itself.
    """
    i = apex_idx
    while i > 0:
        t_ok = (not np.isnan(throttle[i - 1])) and throttle[i - 1] >= THROTTLE_OPEN_PCT
        b_ok = (not np.isnan(brake[i - 1])) and brake[i - 1] < BRAKE_OFF_PCT
        if t_ok and b_ok:
            return i
        i -= 1
    return 0


def _walk_forward_to_exit(apex_idx: int, throttle: np.ndarray) -> int:
    n = len(throttle)
    i = apex_idx
    while i < n - 1:
        t = throttle[i + 1]
        if (not np.isnan(t)) and t >= THROTTLE_OPEN_PCT:
            return i + 1
        i += 1
    return n - 1


def _severity(min_speed_kph: float, max_brake_pct: float) -> int:
    if min_speed_kph >= 220:
        return 1
    if min_speed_kph >= 180:
        return 2 if max_brake_pct < 30 else 3
    if min_speed_kph >= 140:
        return 3 if max_brake_pct < 50 else 4
    if min_speed_kph >= 100:
        return 4 if max_brake_pct < 70 else 5
    if min_speed_kph >= 60:
        return 5
    return 6
