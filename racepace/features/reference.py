"""Reference laps: a fast clean lap stored on a 1m grid for the coach.

Used by the fast loop (live_delta — instant pace check) and the slow
loop (sector_delta — diagnose where time is leaking). Stored as Parquet
for compactness and for cheap columnar reads.

Path layout:
    references/{sim}/{track}/{car}.parquet
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from racepace.features.laps import Lap


@dataclass
class Reference:
    sim: str
    track: str
    car: str | None
    df: pl.DataFrame   # columns: distance_m, speed_kph, throttle_pct, brake_pct, time_s, sector

    @property
    def total_length_m(self) -> float:
        return float(self.df["distance_m"][-1])

    def speed_at(self, distance_m: float) -> float | None:
        return _interp_at(self.df, "speed_kph", distance_m)

    def throttle_at(self, distance_m: float) -> float | None:
        return _interp_at(self.df, "throttle_pct", distance_m)

    def brake_at(self, distance_m: float) -> float | None:
        return _interp_at(self.df, "brake_pct", distance_m)

    def time_at(self, distance_m: float) -> float | None:
        return _interp_at(self.df, "time_s", distance_m)


def _interp_at(df: pl.DataFrame, col: str, x: float) -> float | None:
    xs = df["distance_m"].to_numpy()
    ys = df[col].to_numpy()
    if len(xs) == 0:
        return None
    if x < xs[0] or x > xs[-1]:
        return None
    return float(np.interp(x, xs, ys))


@dataclass
class LiveDelta:
    distance_m: float
    speed_delta_kph: float | None     # current - reference
    throttle_delta_pct: float | None
    brake_delta_pct: float | None
    cumulative_time_delta_s: float | None  # current_time_at_distance - reference_time


@dataclass
class SectorDelta:
    sector: int
    delta_s: float                    # negative = faster than reference
    worst_distance_m: float | None    # where the largest local loss occurred
    worst_local_delta_s: float | None


# --- save / load --------------------------------------------------------------

def save_reference(
    lap: Lap,
    out_path: str | Path,
    sim: str,
    track: str,
    car: str | None = None,
    grid_m: float = 1.0,
) -> Reference:
    """Resample a Lap to a 1m grid and write Parquet."""
    df = _lap_to_grid_df(lap, grid_m=grid_m)
    if df is None or df.height == 0:
        raise ValueError("Lap has no usable distance/speed data; cannot build a reference.")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(str(out))
    return Reference(sim=sim, track=track, car=car, df=df)


def load_reference(path: str | Path, sim: str, track: str, car: str | None = None) -> Reference:
    df = pl.read_parquet(str(path))
    return Reference(sim=sim, track=track, car=car, df=df)


def reference_path(root: str | Path, sim: str, track: str, car: str | None) -> Path:
    """Conventional path: {root}/{sim}/{track}/{car or default}.parquet"""
    car_part = (car or "default").replace("/", "_")
    return Path(root) / sim / track / f"{car_part}.parquet"


# --- live deltas (fast loop) --------------------------------------------------

def live_delta(reference: Reference, frame) -> LiveDelta | None:
    """Compute instantaneous deltas at the driver's current track position."""
    if frame is None or frame.lap_distance_m is None:
        return None
    d = float(frame.lap_distance_m)
    if d < 0 or d > reference.total_length_m:
        return None

    ref_speed = reference.speed_at(d)
    ref_throttle = reference.throttle_at(d)
    ref_brake = reference.brake_at(d)

    sd = (frame.speed_kph - ref_speed) if (frame.speed_kph is not None and ref_speed is not None) else None
    td = (frame.throttle_pct - ref_throttle) if (frame.throttle_pct is not None and ref_throttle is not None) else None
    bd = (frame.brake_pct - ref_brake) if (frame.brake_pct is not None and ref_brake is not None) else None

    return LiveDelta(distance_m=d, speed_delta_kph=sd, throttle_delta_pct=td, brake_delta_pct=bd, cumulative_time_delta_s=None)


# --- sector deltas (slow loop) -----------------------------------------------

def sector_delta(reference: Reference, lap: Lap, sector: int) -> SectorDelta | None:
    """Cumulative time delta over `sector` (0-indexed) vs the reference.

    Computes a per-meter time series from each side, restricts to the
    distance range belonging to `sector` per the reference's sector
    column, and returns the cumulative delta plus the worst local point.
    """
    if reference.df.height < 2:
        return None
    ref_df = reference.df
    sector_mask = ref_df["sector"] == sector
    if sector_mask.sum() == 0:
        return None
    ref_sector = ref_df.filter(sector_mask).sort("distance_m")

    cur_df = _lap_to_grid_df(lap, grid_m=1.0)
    if cur_df is None or cur_df.height == 0:
        return None

    # Restrict current to the sector's distance window.
    d_min = float(ref_sector["distance_m"][0])
    d_max = float(ref_sector["distance_m"][-1])
    cur_sector = cur_df.filter((pl.col("distance_m") >= d_min) & (pl.col("distance_m") <= d_max))
    if cur_sector.height == 0:
        return None

    # Interpolate both onto a common 1m grid spanning the overlap.
    overlap_min = max(d_min, float(cur_sector["distance_m"][0]))
    overlap_max = min(d_max, float(cur_sector["distance_m"][-1]))
    if overlap_max <= overlap_min:
        return None
    xs = np.arange(overlap_min, overlap_max + 1.0, 1.0)
    ref_t = np.interp(xs, ref_df["distance_m"].to_numpy(), ref_df["time_s"].to_numpy())
    cur_t = np.interp(xs, cur_df["distance_m"].to_numpy(), cur_df["time_s"].to_numpy())
    cum_dt = (cur_t - cur_t[0]) - (ref_t - ref_t[0])
    if len(cum_dt) == 0:
        return None

    # Worst local loss = largest positive jump in cumulative delta over a 50m window.
    window = min(50, len(cum_dt) - 1)
    worst_idx = None
    worst_local = -1e9
    if window > 0:
        for i in range(0, len(cum_dt) - window):
            local = float(cum_dt[i + window] - cum_dt[i])
            if local > worst_local:
                worst_local = local
                worst_idx = i + window // 2

    return SectorDelta(
        sector=sector,
        delta_s=float(cum_dt[-1]),
        worst_distance_m=float(xs[worst_idx]) if worst_idx is not None else None,
        worst_local_delta_s=float(worst_local) if worst_idx is not None else None,
    )


# --- shared --------------------------------------------------------------------

def _lap_to_grid_df(lap: Lap, grid_m: float = 1.0) -> pl.DataFrame | None:
    pts = [
        (
            f.lap_distance_m,
            f.speed_kph if f.speed_kph is not None else float("nan"),
            f.throttle_pct if f.throttle_pct is not None else float("nan"),
            f.brake_pct if f.brake_pct is not None else float("nan"),
            f.timestamp_s,
            f.current_sector if f.current_sector is not None else 0,
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
    raw_ts = np.array([p[4] for p in deduped], dtype=float)
    raw_sec = np.array([p[5] for p in deduped], dtype=float)

    grid = np.arange(raw_d[0], raw_d[-1], grid_m)
    if len(grid) < 5:
        return None

    def interp(values: np.ndarray) -> np.ndarray:
        valid = ~np.isnan(values)
        if valid.sum() < 2:
            return np.zeros_like(grid)
        return np.interp(grid, raw_d[valid], values[valid])

    sec_interp = np.interp(grid, raw_d, raw_sec).round().astype(int)

    return pl.DataFrame({
        "distance_m": grid,
        "speed_kph": interp(raw_s),
        "throttle_pct": interp(raw_t),
        "brake_pct": interp(raw_b),
        "time_s": np.interp(grid, raw_d, raw_ts),
        "sector": sec_interp,
    })
