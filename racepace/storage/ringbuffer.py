"""Bounded, thread-safe in-memory frame buffer.

Producer (adapter thread) calls .push(); consumer (agent thread) calls
.snapshot() / .latest() / .window(). One lock guards the deque.

Capacity is sized in seconds × expected_hz with a 50% headroom — under-
or over-rate adapters won't push frames out prematurely.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Iterable

from racepace.schema import TelemetryFrame


class RingBuffer:
    def __init__(self, capacity_seconds: float = 60.0, expected_hz: float = 30.0) -> None:
        self.capacity_seconds = float(capacity_seconds)
        self.expected_hz = float(expected_hz)
        max_len = max(1, int(capacity_seconds * expected_hz * 1.5))
        self._buf: deque[TelemetryFrame] = deque(maxlen=max_len)
        self._lock = threading.Lock()

    def push(self, frame: TelemetryFrame) -> None:
        with self._lock:
            self._buf.append(frame)

    def extend(self, frames: Iterable[TelemetryFrame]) -> None:
        with self._lock:
            self._buf.extend(frames)

    def snapshot(self) -> list[TelemetryFrame]:
        with self._lock:
            return list(self._buf)

    def latest(self) -> TelemetryFrame | None:
        with self._lock:
            return self._buf[-1] if self._buf else None

    def window(self, seconds: float) -> list[TelemetryFrame]:
        """Last `seconds` worth of frames, by timestamp_s."""
        with self._lock:
            if not self._buf:
                return []
            cutoff = self._buf[-1].timestamp_s - seconds
            # Walk from the right; deques are O(1) at either end but iter is O(n).
            out: list[TelemetryFrame] = []
            for f in reversed(self._buf):
                if f.timestamp_s < cutoff:
                    break
                out.append(f)
            out.reverse()
            return out

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()
