"""Voice layer: player queue + clip cache."""

from __future__ import annotations

import time

import numpy as np

from racepace.voice.cache import ClipCache, default_phrase_list
from racepace.voice.live_tts import NullTTSBackend, RecordingTTSBackend
from racepace.voice.player import NullBackend, Player, RecordingBackend


def test_player_drops_stale_clips():
    """Inject a clip with an old timestamp so it's already stale on dequeue."""
    from racepace.voice.player import _QueuedClip

    rec = RecordingBackend()
    p = Player(backend=rec, max_age_s=0.05)
    samples = np.zeros(100, dtype=np.float32)
    fake_old = _QueuedClip(samples=samples, sample_rate=1000, enqueued_t=time.monotonic() - 10.0)
    p._q.put(fake_old)
    time.sleep(0.5)
    p.stop()
    assert len(rec.played) == 0


def test_player_plays_fresh_clip():
    rec = RecordingBackend()
    p = Player(backend=rec, max_age_s=1.0)
    samples = np.zeros(100, dtype=np.float32)
    p.play(samples, 1000, label="ok")
    time.sleep(0.5)  # let the worker drain
    p.stop()
    assert len(rec.played) == 1


def test_clip_cache_preload_with_recording_tts():
    tts = RecordingTTSBackend()
    cache = ClipCache()
    phrases = ["left", "right", "brake"]
    cache.preload(phrases, tts)
    assert tts.requested == phrases
    assert all(p in cache for p in phrases)


def test_default_phrase_list_includes_pace_notes():
    phrases = default_phrase_list()
    # Sanity: covers the must-haves.
    assert "left three" in phrases
    assert "right four" in phrases
    assert "brake" in phrases
    assert "flat" in phrases
    assert "brake in two" in phrases
