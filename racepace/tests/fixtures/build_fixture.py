"""Generate a tiny deterministic fixture session.

5 laps of synthetic telemetry on a 4000 m track at ~30 Hz. Lap 3 has a
simulated off-track (sudden speed drop). Lap times degrade slightly to
exercise the deg fit. Run as:

    python -m racepace.tests.fixtures.build_fixture
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

from racepace.schema import SessionInfo, TelemetryFrame
from racepace.storage.session_store import SessionWriter


SESSION_ID = "00000000-0000-4000-8000-000000000001"
TRACK_LEN_M = 4000.0
HZ = 30.0
N_LAPS = 5
BASE_LAP_TIME_S = 90.0
DEG_S_PER_LAP = 0.20


def build(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    info = SessionInfo(
        session_id=SESSION_ID,
        sim="acc",
        started_at=datetime(2026, 4, 25, 12, 0, 0, tzinfo=timezone.utc),
        track="fixture-track",
        car="fixture-car",
        session_type="practice",
        weather="dry",
        track_temp_c=28.0,
        air_temp_c=22.0,
    )

    with SessionWriter(out_path) as w:
        w.write_session(info)
        t = 0.0
        fuel_kg = 60.0
        wear_pct = 0.0
        last_lap_time = 0.0
        best_lap_time = 0.0
        for lap_idx in range(N_LAPS):
            lap_number = lap_idx + 1
            lap_time = BASE_LAP_TIME_S + DEG_S_PER_LAP * lap_idx
            n_frames = int(lap_time * HZ)
            for i in range(n_frames):
                progress = i / n_frames
                dist = progress * TRACK_LEN_M
                # Synthetic speed: sinusoidal, ~120-220 kph.
                speed = 170.0 + 50.0 * math.sin(progress * 2 * math.pi)
                throttle = max(0.0, 100.0 * math.cos(progress * 4 * math.pi))
                brake = max(0.0, -50.0 * math.cos(progress * 4 * math.pi))
                # Lap 3 has a sharp speed drop at ~50% (off-track).
                if lap_number == 3 and 0.49 < progress < 0.51:
                    speed = 60.0
                w.write_frame(TelemetryFrame(
                    timestamp_s=t,
                    lap_number=lap_number,
                    lap_distance_m=dist,
                    lap_distance_pct=progress,
                    speed_kph=speed,
                    throttle_pct=throttle,
                    brake_pct=brake,
                    gear=4,
                    rpm=7000.0,
                    fuel_kg=fuel_kg,
                    fuel_laps_remaining=fuel_kg / 2.5,
                    tyre_wear_pct={"fl": wear_pct, "fr": wear_pct, "rl": wear_pct, "rr": wear_pct},
                    tyre_temp_c={"fl": 80.0, "fr": 80.0, "rl": 78.0, "rr": 78.0},
                    tyre_pressure_psi={"fl": 27.5, "fr": 27.5, "rl": 27.0, "rr": 27.0},
                    tyre_compound="dry",
                    position=1,
                    last_lap_time_s=last_lap_time if last_lap_time > 0 else None,
                    best_lap_time_s=best_lap_time if best_lap_time > 0 else None,
                    current_sector=int(progress * 3),
                ))
                t += 1.0 / HZ
                fuel_kg -= 2.5 / n_frames
                wear_pct += 0.5 / n_frames
            last_lap_time = lap_time
            best_lap_time = lap_time if best_lap_time == 0 else min(best_lap_time, lap_time)
        w.update_session_end(
            ended_at=datetime(2026, 4, 25, 12, 7, 30, tzinfo=timezone.utc),
            total_laps=N_LAPS,
        )


if __name__ == "__main__":
    out = Path(__file__).parent / "fixture_session.db"
    build(out)
    print(f"Wrote {out}")
