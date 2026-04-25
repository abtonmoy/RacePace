"""Non-blocking audio queue for the coach's fast loop.

The fast loop calls `Player.play(samples, rate)` and returns immediately.
A worker thread drains the queue and hands samples to an `AudioBackend`.
Clips queued more than `max_age_s` ago are dropped — a stale callout is
worse than no callout.

Backends:
- `SoundDeviceBackend` — real audio via the `sounddevice` package (only
  imported on demand so the package is optional).
- `NullBackend` — no-op; sleeps the clip's duration so timing is realistic.
- `RecordingBackend` — captures every play into a list (used by tests).
"""

from __future__ import annotations

import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class _QueuedClip:
    samples: np.ndarray
    sample_rate: int
    enqueued_t: float
    label: str = ""


class AudioBackend(ABC):
    @abstractmethod
    def play(self, samples: np.ndarray, sample_rate: int) -> None:
        """Synchronously play one clip. The Player thread serializes calls."""

    def close(self) -> None:
        pass


class NullBackend(AudioBackend):
    def play(self, samples: np.ndarray, sample_rate: int) -> None:
        time.sleep(len(samples) / max(1, sample_rate))


class RecordingBackend(AudioBackend):
    def __init__(self) -> None:
        self.played: list[tuple[float, str]] = []
        self._t0 = time.monotonic()

    def play(self, samples: np.ndarray, sample_rate: int) -> None:
        # Tests inspect (relative_time, label_or_len). Sleep is omitted
        # to keep tests fast.
        self.played.append((time.monotonic() - self._t0, f"len={len(samples)}"))


class SoundDeviceBackend(AudioBackend):
    def __init__(self) -> None:
        import sounddevice as sd  # imported lazily; optional dep
        self._sd = sd

    def play(self, samples: np.ndarray, sample_rate: int) -> None:
        # Convert to float32 in [-1, 1] if needed.
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32) / max(1.0, np.iinfo(samples.dtype).max if np.issubdtype(samples.dtype, np.integer) else 1.0)
        self._sd.play(samples, sample_rate, blocking=True)


def default_backend() -> AudioBackend:
    """Pick SoundDeviceBackend if installed, else NullBackend."""
    try:
        return SoundDeviceBackend()
    except Exception:
        return NullBackend()


class Player:
    def __init__(
        self,
        backend: AudioBackend | None = None,
        max_age_s: float = 0.5,
        min_gap_s: float = 0.0,
    ) -> None:
        self.backend = backend if backend is not None else default_backend()
        self.max_age_s = max_age_s
        self.min_gap_s = min_gap_s   # caller-enforced gap between clips
        self._q: queue.Queue[_QueuedClip | None] = queue.Queue(maxsize=64)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="Player", daemon=True)
        self._last_played_t: float = 0.0
        self._lock = threading.Lock()
        self._thread.start()

    def play(self, samples: np.ndarray, sample_rate: int, label: str = "") -> bool:
        """Enqueue a clip; returns False if the queue is full (clip dropped)."""
        clip = _QueuedClip(samples=samples, sample_rate=sample_rate, enqueued_t=time.monotonic(), label=label)
        try:
            self._q.put_nowait(clip)
            return True
        except queue.Full:
            return False

    def stop(self, timeout: float = 1.0) -> None:
        self._stop.set()
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)
        self.backend.close()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                clip = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if clip is None:
                break
            now = time.monotonic()
            if now - clip.enqueued_t > self.max_age_s:
                continue   # stale; drop
            if self.min_gap_s > 0:
                gap = now - self._last_played_t
                if gap < self.min_gap_s:
                    time.sleep(self.min_gap_s - gap)
            try:
                self.backend.play(clip.samples, clip.sample_rate)
            except Exception:
                pass
            self._last_played_t = time.monotonic()
