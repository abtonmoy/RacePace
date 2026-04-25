"""Reference-lap save/load + live_delta + sector_delta."""

from __future__ import annotations

from pathlib import Path

from racepace.features.laps import Lap
from racepace.features.reference import (
    live_delta,
    load_reference,
    save_reference,
    sector_delta,
)
from racepace.schema import TelemetryFrame
from racepace.tests.fixtures.build_track_fixtures import build_synthetic_lap


def test_save_and_load_roundtrip(tmp_path: Path):
    lap = build_synthetic_lap()
    out = tmp_path / "ref.parquet"
    ref = save_reference(lap, out, sim="acc", track="synthetic_test", car="fixture-car")
    assert out.exists()
    loaded = load_reference(out, sim="acc", track="synthetic_test", car="fixture-car")
    assert loaded.df.height == ref.df.height
    assert abs(loaded.total_length_m - ref.total_length_m) < 1.0


def test_live_delta_zero_against_self(tmp_path: Path):
    """Driving the reference exactly should yield ~zero delta everywhere."""
    lap = build_synthetic_lap()
    ref = save_reference(lap, tmp_path / "ref.parquet", sim="acc", track="synthetic_test")
    # Pick a frame mid-lap and ask for its delta against itself
    f = lap.frames[len(lap.frames) // 2]
    d = live_delta(ref, f)
    assert d is not None
    assert abs(d.speed_delta_kph) < 0.5
    assert abs(d.throttle_delta_pct) < 0.5


def test_live_delta_returns_none_outside_track():
    lap = build_synthetic_lap()
    ref = save_reference(lap, Path("/tmp/_ref_outside.parquet"), sim="acc", track="synthetic_test")
    bad_frame = TelemetryFrame(timestamp_s=0, lap_number=1, lap_distance_m=999999.0)
    assert live_delta(ref, bad_frame) is None


def test_sector_delta_against_self_is_near_zero(tmp_path: Path):
    lap = build_synthetic_lap()
    ref = save_reference(lap, tmp_path / "ref.parquet", sim="acc", track="synthetic_test")
    sd = sector_delta(ref, lap, sector=1)
    assert sd is not None
    assert abs(sd.delta_s) < 0.1


def test_sector_delta_positive_when_slower(tmp_path: Path):
    """Slowing down (timestamps stretched) should give a positive delta."""
    fast = build_synthetic_lap()
    ref = save_reference(fast, tmp_path / "ref.parquet", sim="acc", track="synthetic_test")
    # Make a slower lap by stretching timestamps 10%.
    slow_frames: list[TelemetryFrame] = []
    for f in fast.frames:
        slow_frames.append(TelemetryFrame(
            timestamp_s=f.timestamp_s * 1.1,
            lap_number=f.lap_number,
            lap_distance_m=f.lap_distance_m,
            speed_kph=f.speed_kph,
            throttle_pct=f.throttle_pct,
            brake_pct=f.brake_pct,
            current_sector=f.current_sector,
        ))
    slow_lap = Lap(lap_number=1, frames=slow_frames, lap_time_s=fast.lap_time_s * 1.1, is_clean=True, is_complete=True)
    sd = sector_delta(ref, slow_lap, sector=1)
    assert sd is not None
    assert sd.delta_s > 0.5  # easily detectable
