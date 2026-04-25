"""Tyre degradation: linear fit of lap_time vs lap_number per stint.

A "stint" is a contiguous run of laps with no compound change and no
implausible lap-time gap (>10× the median lap time → assumed pit / out
lap break). We report slope (s/lap), intercept, and a 95% CI on the
slope estimated from the OLS standard error. The CI requires ≥ 3 laps
in the stint.

Out / in / pit / dirty laps are excluded from the fit but kept in the
stint definition so neighbouring laps remain contiguous.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from racepace.features.laps import Lap


_OPTIMAL_TYRE_TEMP_C = (70.0, 100.0)   # rough GT3 racing window


@dataclass
class StintDeg:
    stint_index: int                       # 0-based
    lap_numbers: list[int]                 # all laps in the stint, in order
    fit_lap_numbers: list[int]             # laps used in the fit (clean only)
    compound: str | None
    laps: int                              # len(lap_numbers)

    slope_s_per_lap: float | None = None   # positive = degrading
    intercept_s: float | None = None
    slope_ci95_s: float | None = None      # half-width; slope ± this
    r_squared: float | None = None

    tyre_temp_drift_c: dict[str, float] | None = None    # avg(last lap) - avg(first lap)
    out_of_window_pct: dict[str, float] | None = None    # frac of stint above/below window


@dataclass
class DegProfile:
    stints: list[StintDeg] = field(default_factory=list)


def tyre_degradation(laps: list[Lap]) -> DegProfile:
    stints = _split_into_stints(laps)
    out: list[StintDeg] = []
    for i, stint in enumerate(stints):
        out.append(_fit_stint(i, stint))
    return DegProfile(stints=out)


def _split_into_stints(laps: list[Lap]) -> list[list[Lap]]:
    if not laps:
        return []
    stints: list[list[Lap]] = [[laps[0]]]
    for prev, lap in zip(laps, laps[1:]):
        if lap.compound and prev.compound and lap.compound != prev.compound:
            stints.append([lap])
            continue
        # If both lap times exist and lap is wildly slower than previous, treat as in-lap break.
        if prev.lap_time_s and lap.lap_time_s and lap.lap_time_s > prev.lap_time_s * 3:
            stints.append([lap])
            continue
        stints[-1].append(lap)
    return stints


def _fit_stint(idx: int, stint: list[Lap]) -> StintDeg:
    lap_nums = [l.lap_number for l in stint]
    compound = next((l.compound for l in stint if l.compound), None)

    # Use only clean, complete laps with valid lap_time_s for the fit.
    clean = [
        l for l in stint
        if l.lap_time_s is not None and l.is_complete and l.is_clean is not False
    ]
    fit_nums = [l.lap_number for l in clean]

    s = StintDeg(
        stint_index=idx,
        lap_numbers=lap_nums,
        fit_lap_numbers=fit_nums,
        compound=compound,
        laps=len(stint),
    )

    if len(clean) >= 2:
        x = np.array([l.lap_number for l in clean], dtype=float)
        y = np.array([l.lap_time_s for l in clean], dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        y_hat = slope * x + intercept
        ss_res = float(((y - y_hat) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else None

        ci = None
        if len(clean) >= 3:
            n = len(clean)
            sigma2 = ss_res / (n - 2) if n > 2 else 0.0
            sxx = float(((x - x.mean()) ** 2).sum())
            if sxx > 0 and sigma2 >= 0:
                se_slope = float(np.sqrt(sigma2 / sxx))
                # t-multiplier ≈ 1.96 for 95% CI; coarse but fine for n typically <30
                ci = 1.96 * se_slope

        s.slope_s_per_lap = float(slope)
        s.intercept_s = float(intercept)
        s.slope_ci95_s = ci
        s.r_squared = r2

    s.tyre_temp_drift_c = _temp_drift(stint)
    s.out_of_window_pct = _temp_out_of_window(stint)

    return s


def _temp_drift(stint: list[Lap]) -> dict[str, float] | None:
    first = next((l.tyre_temp_avg_c for l in stint if l.tyre_temp_avg_c), None)
    last = next((l.tyre_temp_avg_c for l in reversed(stint) if l.tyre_temp_avg_c), None)
    if not first or not last:
        return None
    return {k: last.get(k, 0.0) - first.get(k, 0.0) for k in first}


def _temp_out_of_window(stint: list[Lap]) -> dict[str, float] | None:
    """Fraction of frames where each tyre's temperature was outside the optimal window."""
    lo, hi = _OPTIMAL_TYRE_TEMP_C
    sums: dict[str, int] = {}
    counts: dict[str, int] = {}
    for lap in stint:
        for f in lap.frames:
            if not f.tyre_temp_c:
                continue
            for k, v in f.tyre_temp_c.items():
                counts[k] = counts.get(k, 0) + 1
                if v < lo or v > hi:
                    sums[k] = sums.get(k, 0) + 1
    if not counts:
        return None
    return {k: (sums.get(k, 0) / counts[k]) * 100.0 for k in counts}
