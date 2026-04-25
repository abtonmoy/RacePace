"""Pre-rendered clip cache for the coach's fast loop.

The coach's pace-note vocabulary is small and known up-front. Synthesize
once at startup, hold in RAM, play with zero IO on the hot path.

Default vocabulary covers:
- Directions: "left", "right"
- Severity: "one" through "six"
- Actions: "brake", "flat", "lift", "trail brake", "no lift", "easy"
- Numbers 1..9 for distance callouts ("brake in three")
- Common short phrases: "good", "tidy that up", "more apex"

You can also pre-render combined phrases ("left three", "brake in two")
as single clips to avoid mid-callout gaps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from racepace.voice.live_tts import (
    NullTTSBackend,
    OnDiskCachedTTS,
    TTSBackend,
    _read_wav_mono,
    _write_wav_mono,
    default_tts,
)


DIRECTIONS = ["left", "right"]
SEVERITY_WORDS = ["one", "two", "three", "four", "five", "six"]
NUMBER_WORDS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
ACTIONS = ["brake", "flat", "lift", "trail brake", "no lift", "easy"]
SHORT = ["good", "tidy that up", "more apex", "carry more speed"]


def default_phrase_list() -> list[str]:
    """Atomic phrases the coach is allowed to say."""
    out: list[str] = []
    out.extend(DIRECTIONS)
    out.extend(SEVERITY_WORDS)
    out.extend([f"{d} {s}" for d in DIRECTIONS for s in SEVERITY_WORDS])  # left three, right four, ...
    out.extend(ACTIONS)
    out.extend([f"brake in {n}" for n in NUMBER_WORDS[:5]])
    out.extend(SHORT)
    seen = set()
    deduped: list[str] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped


class ClipCache:
    """In-RAM mapping from phrase → (samples, sample_rate)."""

    def __init__(self) -> None:
        self._clips: dict[str, tuple[np.ndarray, int]] = {}

    def __contains__(self, phrase: str) -> bool:
        return phrase in self._clips

    def __len__(self) -> int:
        return len(self._clips)

    def get(self, phrase: str) -> tuple[np.ndarray, int] | None:
        return self._clips.get(phrase)

    def add(self, phrase: str, samples: np.ndarray, sample_rate: int) -> None:
        self._clips[phrase] = (samples, sample_rate)

    def preload(self, phrases: list[str], tts: TTSBackend) -> None:
        for p in phrases:
            try:
                self.add(p, *tts.synthesize(p))
            except Exception:
                # Don't let one bad phrase poison the cache; the coach's
                # caller knows to skip missing phrases.
                continue

    def load_dir(self, dir_path: str | Path) -> int:
        """Load every {phrase}.wav under `dir_path`. Returns count loaded."""
        d = Path(dir_path)
        n = 0
        if not d.exists():
            return 0
        for wav in d.glob("*.wav"):
            try:
                samples, sr = _read_wav_mono(wav)
            except Exception:
                continue
            phrase = wav.stem.replace("_", " ")
            self.add(phrase, samples, sr)
            n += 1
        return n

    def dump_dir(self, dir_path: str | Path) -> int:
        d = Path(dir_path)
        d.mkdir(parents=True, exist_ok=True)
        n = 0
        for phrase, (samples, sr) in self._clips.items():
            fname = phrase.replace(" ", "_") + ".wav"
            try:
                _write_wav_mono(d / fname, samples, sr)
                n += 1
            except Exception:
                continue
        return n


def build_default_cache(
    tts: TTSBackend | None = None,
    cache_dir: str | Path | None = None,
) -> ClipCache:
    """Synthesize (or load) the default phrase set.

    If `cache_dir` is given, an OnDiskCachedTTS is layered around the
    backend so the second startup is fast. If no TTS backend is configured
    (NullTTSBackend), the cache will hold short silent buffers — the coach
    still runs but says nothing audible.
    """
    if tts is None:
        tts = default_tts()
    if cache_dir is not None:
        tts = OnDiskCachedTTS(tts, cache_dir)
    cache = ClipCache()
    cache.preload(default_phrase_list(), tts)
    return cache
