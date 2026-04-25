"""Golden-report test: structured report should be stable across runs.

We do NOT snapshot the LLM output (non-deterministic). We snapshot the
report dict the LLM would see — that's the actual contract.
"""

from __future__ import annotations

from racepace.agents.analyst import build_report
from racepace.features.laps import split_into_laps
from racepace.storage.session_store import SessionReader


def test_report_has_expected_top_level_shape(fixture_db_path):
    with SessionReader(fixture_db_path) as r:
        info, frames = r.load_session(r.latest_session_id())
    laps = split_into_laps(frames)
    report = build_report(info, frames, laps)

    assert set(report.keys()) == {
        "session", "lap_summary", "fuel", "tyre_degradation", "sector_focus", "notable_events",
    }


def test_report_lap_summary_matches_fixture(fixture_db_path):
    with SessionReader(fixture_db_path) as r:
        info, frames = r.load_session(r.latest_session_id())
    laps = split_into_laps(frames)
    report = build_report(info, frames, laps)

    ls = report["lap_summary"]
    assert ls["total_laps"] == 5
    # Lap 3 is the dirty one in the fixture; the rest are clean.
    assert ls["dirty_laps"] == 1
    # Best lap should be lap 1 (fastest) since later laps are degraded.
    assert ls["best_lap"]["lap_number"] == 1
    assert ls["best_lap"]["is_clean"] is True


def test_report_notable_events_includes_off_track(fixture_db_path):
    with SessionReader(fixture_db_path) as r:
        info, frames = r.load_session(r.latest_session_id())
    laps = split_into_laps(frames)
    report = build_report(info, frames, laps)

    types = {e["type"] for e in report["notable_events"]}
    assert "off_track_or_spin" in types


def test_report_session_metadata_carries_through(fixture_db_path):
    with SessionReader(fixture_db_path) as r:
        info, frames = r.load_session(r.latest_session_id())
    laps = split_into_laps(frames)
    report = build_report(info, frames, laps)

    s = report["session"]
    assert s["sim"] == "acc"
    assert s["track"] == "fixture-track"
    assert s["session_type"] == "practice"
    assert s["track_temp_c"] == 28.0


def test_report_is_json_serializable(fixture_db_path):
    """The LLM only ever sees JSON — verify the report serializes cleanly."""
    import json

    with SessionReader(fixture_db_path) as r:
        info, frames = r.load_session(r.latest_session_id())
    laps = split_into_laps(frames)
    report = build_report(info, frames, laps)

    s = json.dumps(report, default=str)
    assert len(s) > 0
