"""Build synthetic multi-corner reference laps and committed track-map JSONs.

Used by the corner-extraction test and the coach smoke test. Two output
shapes:

- A `Lap` constructable in-process for tests that need raw frames.
- JSON track maps under `racepace/tracks/{acc,f1}/synthetic_test.json`
  for the coach to load.

Run as: `python -m racepace.tests.fixtures.build_track_fixtures`
"""

from __future__ import annotations

import math
from pathlib import Path

from racepace.features.laps import Lap
from racepace.features.track_map import extract_track_map
from racepace.schema import TelemetryFrame


TRACK_LENGTH_M = 4500.0
HZ = 50.0
LAP_TIME_S = 100.0


# Three synthetic corners — (apex_distance_m, severity, direction_sign, target_min_kph)
_CORNERS = [
    (1000.0, 4, -1, 100.0),  # tight left
    (2400.0, 2, +1, 200.0),  # fast right
    (3600.0, 5, -1, 70.0),   # slow left hairpin
]


def _speed_profile(distance_m: float) -> tuple[float, float, float, float]:
    """Returns (speed_kph, throttle_pct, brake_pct, steering_norm) at `distance_m`."""
    base = 240.0
    speed = base
    brake = 0.0
    throttle = 100.0
    steering = 0.0
    for apex, _, sign, min_kph in _CORNERS:
        # Speed dip: gaussian centered on apex with width depending on severity.
        sigma = 80.0
        dip = (base - min_kph) * math.exp(-((distance_m - apex) ** 2) / (2 * sigma * sigma))
        speed -= dip
        # Brake before apex
        if apex - 200 < distance_m < apex - 30:
            brake = max(brake, 80.0)
            throttle = 0.0
        elif apex - 30 <= distance_m <= apex + 30:
            brake = max(brake, 20.0)
            throttle = 0.0
        # Steering near apex
        if apex - 100 < distance_m < apex + 100:
            steering = sign * 0.6
    speed = max(min_kph_overall(distance_m), speed)
    return speed, throttle, brake, steering


def min_kph_overall(distance_m: float) -> float:
    return min(min_kph for _, _, _, min_kph in _CORNERS)


def build_synthetic_lap() -> Lap:
    n_frames = int(LAP_TIME_S * HZ)
    frames: list[TelemetryFrame] = []
    for i in range(n_frames):
        progress = i / n_frames
        d = progress * TRACK_LENGTH_M
        speed, throttle, brake, steering = _speed_profile(d)
        frames.append(TelemetryFrame(
            timestamp_s=i / HZ,
            lap_number=1,
            lap_distance_m=d,
            lap_distance_pct=progress,
            speed_kph=speed,
            throttle_pct=throttle,
            brake_pct=brake,
            steering_norm=steering,
            gear=4,
            rpm=7000.0,
            current_sector=int(progress * 3),
        ))
    lap = Lap(lap_number=1, frames=frames, lap_time_s=LAP_TIME_S, is_clean=True, is_complete=True)
    return lap


def build_and_save() -> dict[str, Path]:
    lap = build_synthetic_lap()
    out: dict[str, Path] = {}
    base = Path(__file__).parent.parent.parent / "tracks"
    for sim in ("acc", "f1"):
        tm = extract_track_map(lap, track="synthetic_test", sim=sim, car="fixture-car")
        path = base / sim / "synthetic_test.json"
        tm.save(path)
        out[sim] = path
    return out


if __name__ == "__main__":
    paths = build_and_save()
    for sim, p in paths.items():
        print(f"Wrote {sim} → {p}")
