"""Resample two laps onto a common track-distance axis and produce a
per-meter delta of speed, throttle, brake.

The resampled axis goes from 0 to L meters in 1m steps, where L is the
shorter of the two laps' max `lap_distance_m`. (Lap distance can vary
slightly across laps because the racing line differs — using the shorter
of the two is conservative.) Frames missing `lap_distance_m` are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from racepace.features.laps import Lap


@dataclass
class DeltaTrace:
    distance_m: list[float]                # x-axis, 0..L in 1m steps
    speed_delta_kph: list[float | None]    # target - reference (positive = target faster)
    throttle_delta_pct: list[float | None]
    brake_delta_pct: list[float | None]
    cumulative_time_delta_s: list[float | None]   # ahead/behind reference at each x

    reference_lap_number: int
    target_lap_number: int

    @property
    def total_time_delta_s(self) -> float | None:
        """Total time difference at end of lap; positive = target slower."""
        for v in reversed(self.cumulative_time_delta_s):
            if v is not None:
                return v
        return None


def compare_laps(reference: Lap, target: Lap, step_m: float = 1.0) -> DeltaTrace:
    """Per-meter delta of target vs reference."""
    ref_arr = _frames_to_arrays(reference)
    tgt_arr = _frames_to_arrays(target)

    if ref_arr is None or tgt_arr is None:
        return DeltaTrace(
            distance_m=[],
            speed_delta_kph=[],
            throttle_delta_pct=[],
            brake_delta_pct=[],
            cumulative_time_delta_s=[],
            reference_lap_number=reference.lap_number,
            target_lap_number=target.lap_number,
        )

    ref_dist, ref_speed, ref_throttle, ref_brake, ref_t = ref_arr
    tgt_dist, tgt_speed, tgt_throttle, tgt_brake, tgt_t = tgt_arr

    L = min(ref_dist[-1], tgt_dist[-1])
    if L <= 0:
        return DeltaTrace(
            distance_m=[],
            speed_delta_kph=[],
            throttle_delta_pct=[],
            brake_delta_pct=[],
            cumulative_time_delta_s=[],
            reference_lap_number=reference.lap_number,
            target_lap_number=target.lap_number,
        )

    xs = np.arange(0.0, L + step_m, step_m)

    ref_speed_i = _interp_or_none(xs, ref_dist, ref_speed)
    tgt_speed_i = _interp_or_none(xs, tgt_dist, tgt_speed)
    ref_throttle_i = _interp_or_none(xs, ref_dist, ref_throttle)
    tgt_throttle_i = _interp_or_none(xs, tgt_dist, tgt_throttle)
    ref_brake_i = _interp_or_none(xs, ref_dist, ref_brake)
    tgt_brake_i = _interp_or_none(xs, tgt_dist, tgt_brake)

    ref_t_i = np.interp(xs, ref_dist, ref_t)
    tgt_t_i = np.interp(xs, tgt_dist, tgt_t)
    cum_dt = (tgt_t_i - tgt_t_i[0]) - (ref_t_i - ref_t_i[0])

    return DeltaTrace(
        distance_m=xs.tolist(),
        speed_delta_kph=_subtract(tgt_speed_i, ref_speed_i),
        throttle_delta_pct=_subtract(tgt_throttle_i, ref_throttle_i),
        brake_delta_pct=_subtract(tgt_brake_i, ref_brake_i),
        cumulative_time_delta_s=cum_dt.tolist(),
        reference_lap_number=reference.lap_number,
        target_lap_number=target.lap_number,
    )


def _frames_to_arrays(lap: Lap):
    """Strip frames lacking lap_distance_m or timestamp; return sorted-by-distance arrays."""
    pts = [
        (f.lap_distance_m, f.speed_kph, f.throttle_pct, f.brake_pct, f.timestamp_s)
        for f in lap.frames
        if f.lap_distance_m is not None
    ]
    if not pts:
        return None
    pts.sort(key=lambda p: p[0])
    # Drop duplicates / non-monotonic samples (np.interp requires strictly increasing xp).
    deduped = [pts[0]]
    for p in pts[1:]:
        if p[0] > deduped[-1][0]:
            deduped.append(p)
    if len(deduped) < 2:
        return None
    dist = np.array([p[0] for p in deduped], dtype=float)
    speed = np.array([_nan_if_none(p[1]) for p in deduped], dtype=float)
    throttle = np.array([_nan_if_none(p[2]) for p in deduped], dtype=float)
    brake = np.array([_nan_if_none(p[3]) for p in deduped], dtype=float)
    t = np.array([p[4] for p in deduped], dtype=float)
    return dist, speed, throttle, brake, t


def _nan_if_none(v) -> float:
    return float("nan") if v is None else float(v)


def _interp_or_none(xs: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> list[float | None]:
    """np.interp but returns None where the source is NaN at any contributing sample."""
    if np.all(np.isnan(fp)):
        return [None] * len(xs)
    # np.interp ignores NaNs poorly — replace NaNs by interpolation of the rest.
    valid = ~np.isnan(fp)
    if valid.sum() < 2:
        return [None] * len(xs)
    interp = np.interp(xs, xp[valid], fp[valid])
    return interp.tolist()


def _subtract(a: list[float | None], b: list[float | None]) -> list[float | None]:
    out: list[float | None] = []
    for ai, bi in zip(a, b):
        if ai is None or bi is None:
            out.append(None)
        else:
            out.append(ai - bi)
    return out
