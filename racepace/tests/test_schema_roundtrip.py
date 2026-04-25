"""Schema round-trip: write a frame, read it back, assert equality."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from racepace.schema import SessionInfo, TelemetryFrame
from racepace.storage.session_store import SessionReader, SessionWriter


def test_roundtrip_frame_preserves_all_fields(tmp_path: Path) -> None:
    db = tmp_path / "rt.db"
    info = SessionInfo(
        session_id="11111111-1111-4111-8111-111111111111",
        sim="acc",
        started_at=datetime(2026, 4, 25, 10, 0, tzinfo=timezone.utc),
        track="rt-track",
        car="rt-car",
        session_type="practice",
    )
    frame = TelemetryFrame(
        timestamp_s=12.5,
        lap_number=2,
        lap_distance_m=1234.5,
        lap_distance_pct=0.31,
        speed_kph=180.4,
        throttle_pct=88.0,
        brake_pct=0.0,
        clutch_pct=0.0,
        steering_norm=-0.1,
        gear=4,
        rpm=7100.0,
        fuel_kg=42.3,
        fuel_laps_remaining=12.0,
        tyre_wear_pct={"fl": 1.2, "fr": 1.0, "rl": 0.9, "rr": 0.8},
        tyre_temp_c={"fl": 82.0, "fr": 81.5, "rl": 79.0, "rr": 78.5},
        tyre_pressure_psi={"fl": 27.5, "fr": 27.4, "rl": 27.0, "rr": 27.0},
        tyre_compound="dry",
        position=3,
        gap_ahead_s=1.4,
        gap_behind_s=0.8,
        last_lap_time_s=89.123,
        best_lap_time_s=88.901,
        current_sector=1,
        extras={"physics.brake_bias": 56.5, "graphics.flag": "green"},
    )

    with SessionWriter(db) as w:
        w.write_session(info)
        w.write_frame(frame)

    with SessionReader(db) as r:
        loaded_info, frames = r.load_session(info.session_id)

    assert loaded_info == info
    assert len(frames) == 1
    assert frames[0] == frame
