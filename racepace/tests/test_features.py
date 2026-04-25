"""Unit tests for feature extractors using the fixture session."""

from __future__ import annotations

from racepace.features.deg import tyre_degradation
from racepace.features.deltas import compare_laps
from racepace.features.laps import split_into_laps
from racepace.storage.session_store import SessionReader


def _load(db):
    with SessionReader(db) as r:
        return r.load_session(r.latest_session_id())


def test_split_into_laps_returns_expected_laps(fixture_db_path):
    info, frames = _load(fixture_db_path)
    laps = split_into_laps(frames)
    assert [l.lap_number for l in laps] == [1, 2, 3, 4, 5]


def test_lap_3_is_marked_dirty_due_to_speed_drop(fixture_db_path):
    info, frames = _load(fixture_db_path)
    laps = split_into_laps(frames)
    by_num = {l.lap_number: l for l in laps}
    assert by_num[3].is_clean is False
    assert by_num[1].is_clean is True
    assert by_num[2].is_clean is True


def test_lap_times_are_recovered_from_next_laps_last_lap_time(fixture_db_path):
    info, frames = _load(fixture_db_path)
    laps = split_into_laps(frames)
    # Laps 1..4 should have lap_time_s populated from next-lap last_lap_time_s.
    # Lap 5 has no successor, so lap_time_s stays None.
    for l in laps[:-1]:
        assert l.lap_time_s is not None and l.lap_time_s > 0


def test_compare_laps_emits_per_meter_trace(fixture_db_path):
    info, frames = _load(fixture_db_path)
    laps = split_into_laps(frames)
    delta = compare_laps(laps[0], laps[1])
    assert len(delta.distance_m) > 100
    assert len(delta.speed_delta_kph) == len(delta.distance_m)
    assert len(delta.cumulative_time_delta_s) == len(delta.distance_m)


def test_tyre_degradation_fit_has_positive_slope(fixture_db_path):
    info, frames = _load(fixture_db_path)
    laps = split_into_laps(frames)
    deg = tyre_degradation(laps)
    assert len(deg.stints) == 1
    stint = deg.stints[0]
    # Fixture degrades by 0.2 s/lap; with the dirty lap 3 excluded the slope
    # should still come out positive.
    assert stint.slope_s_per_lap is not None
    assert stint.slope_s_per_lap > 0
