"""Coach fast-loop trigger timing using a mocked clip player."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from racepace.agents.coach import CoachAgent, CoachConfig
from racepace.features.track_map import extract_track_map
from racepace.schema import SessionInfo, TelemetryFrame
from racepace.storage.ringbuffer import RingBuffer
from racepace.tests.fixtures.build_track_fixtures import build_synthetic_lap
from racepace.voice.cache import ClipCache, default_phrase_list
from racepace.voice.live_tts import NullTTSBackend


def _build_coach():
    lap = build_synthetic_lap()
    tm = extract_track_map(lap, track="synthetic_test", sim="acc", car="fixture-car")
    cache = ClipCache()
    cache.preload(default_phrase_list(), NullTTSBackend())
    ring = RingBuffer(capacity_seconds=120, expected_hz=50)
    info = SessionInfo(
        session_id="test", sim="acc",
        started_at=datetime(2026, 4, 25, tzinfo=timezone.utc),
        track="synthetic_test", car="fixture-car",
        session_type="hotlap",
    )
    spoken: list[str] = []
    coach = CoachAgent(
        ringbuffer=ring,
        session_info=info,
        track_map=tm,
        clip_cache=cache,
        speak_clip=lambda phrase: spoken.append(phrase),
        speak_text=lambda msg: spoken.append(f"TEXT::{msg}"),
        config=CoachConfig(pace_notes_only=True),
    )
    return coach, ring, lap, spoken


def _drive_through(coach, ring, lap):
    """Push frames one at a time and tick the fast loop after each push."""
    for f in lap.frames:
        ring.push(f)
        coach.fast_tick()


def test_coach_announces_each_corner_at_least_once():
    coach, ring, lap, spoken = _build_coach()
    _drive_through(coach, ring, lap)
    # Each of 3 corners should produce at least one announce ("left/right N").
    direction_severity_phrases = [s for s in spoken if any(s.startswith(d + " ") for d in ("left", "right"))]
    # Expect roughly one announce per corner (3) — allow >=3 since exit_check
    # may produce noise if a "carry more speed" fires (it shouldn't here).
    assert len(direction_severity_phrases) >= 3, f"got {spoken}"


def test_coach_fires_action_callouts_for_braking_corners():
    coach, ring, lap, spoken = _build_coach()
    _drive_through(coach, ring, lap)
    # All 3 synthetic corners are braking corners → expect "brake" callouts.
    brake_calls = [s for s in spoken if s == "brake"]
    assert len(brake_calls) >= 1


def test_coach_silent_with_empty_buffer():
    coach, ring, lap, spoken = _build_coach()
    coach.fast_tick()
    assert spoken == []


def test_coach_pace_notes_only_mode_does_not_call_llm():
    """In pace-notes-only mode the slow loop must be a no-op (no LLM call)."""
    coach, ring, lap, spoken = _build_coach()
    # Stuff the buffer with a full lap and call slow_tick — should not append TEXT::
    for f in lap.frames:
        ring.push(f)
    coach.slow_tick()
    text_msgs = [s for s in spoken if s.startswith("TEXT::")]
    assert text_msgs == []


def test_callout_phrases_are_in_default_vocabulary():
    """Every phrase the coach can ask the cache for must be in the default phrase list."""
    coach, ring, lap, spoken = _build_coach()
    _drive_through(coach, ring, lap)
    vocab = set(default_phrase_list())
    for s in spoken:
        if s.startswith("TEXT::"):
            continue
        assert s in vocab, f"coach asked for {s!r} which isn't in the default vocabulary"
