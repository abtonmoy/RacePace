"""Corner extraction golden test."""

from __future__ import annotations

from pathlib import Path

from racepace.features.track_map import TrackMap, extract_track_map
from racepace.tests.fixtures.build_track_fixtures import build_synthetic_lap


def test_extract_finds_three_corners_with_correct_directions():
    lap = build_synthetic_lap()
    tm = extract_track_map(lap, track="synthetic_test", sim="acc", car="fixture-car")
    assert len(tm.corners) == 3
    assert [c.direction for c in tm.corners] == ["L", "R", "L"]


def test_extracted_apexes_match_designed_positions():
    lap = build_synthetic_lap()
    tm = extract_track_map(lap, track="synthetic_test", sim="acc", car="fixture-car")
    apex_ms = [c.apex_m for c in tm.corners]
    expected = [1000.0, 2400.0, 3600.0]
    for got, want in zip(apex_ms, expected):
        assert abs(got - want) < 30.0, f"apex {got} not within 30m of {want}"


def test_extracted_target_min_speed_matches_designed():
    lap = build_synthetic_lap()
    tm = extract_track_map(lap, track="synthetic_test", sim="acc", car="fixture-car")
    speeds = [c.target_min_speed_kph for c in tm.corners]
    expected = [100.0, 200.0, 70.0]
    for got, want in zip(speeds, expected):
        assert abs(got - want) < 6.0, f"min_speed {got} not within 6 kph of {want}"


def test_brake_point_precedes_apex():
    lap = build_synthetic_lap()
    tm = extract_track_map(lap, track="synthetic_test", sim="acc", car="fixture-car")
    for c in tm.corners:
        assert c.brake_point_m <= c.apex_m
        assert c.exit_m >= c.apex_m


def test_track_map_json_roundtrip():
    lap = build_synthetic_lap()
    tm = extract_track_map(lap, track="synthetic_test", sim="acc")
    js = tm.to_json()
    rt = TrackMap.from_json(js)
    assert rt.track == tm.track
    assert len(rt.corners) == len(tm.corners)
    assert rt.corners[0].direction == tm.corners[0].direction
    assert rt.corners[0].apex_m == tm.corners[0].apex_m


def test_committed_acc_track_map_loads():
    p = Path(__file__).parent.parent / "tracks" / "acc" / "synthetic_test.json"
    tm = TrackMap.load(p)
    assert tm.sim == "acc"
    assert len(tm.corners) == 3


def test_committed_f1_track_map_loads():
    p = Path(__file__).parent.parent / "tracks" / "f1" / "synthetic_test.json"
    tm = TrackMap.load(p)
    assert tm.sim == "f1"
    assert len(tm.corners) == 3
