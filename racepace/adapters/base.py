"""Adapter base class.

Every sim-specific adapter implements this interface. Adapters know about
sims; agents do not. All sim-native units must be converted to the SI-style
units defined in `racepace.schema` before frames leave the adapter.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterator

from racepace.schema import SessionInfo, TelemetryFrame


class AbstractAdapter(ABC):
    """Base class for all telemetry adapters."""

    sim_name: str = ""

    target_hz: float = 30.0
    """Target emission rate. The adapter must downsample if the source is
    faster. If the source is slower, do not interpolate — emit at native
    rate and let consumers handle gaps."""

    @abstractmethod
    def connect(self) -> None:
        """Open the underlying transport (shared memory, UDP socket, file)."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close cleanly. Idempotent."""

    @abstractmethod
    def read_session_info(self) -> SessionInfo:
        """Return a fully populated SessionInfo for the active session.

        Called once at session start. Adapters that cannot determine
        session info up front should populate what they can and leave the
        rest as None — the storage layer will update it on close.
        """

    @abstractmethod
    def stream_frames(self) -> Iterator[TelemetryFrame]:
        """Yield TelemetryFrame instances until the session ends or the
        consumer stops iterating. Must respect target_hz."""

    def __enter__(self) -> AbstractAdapter:
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()


class RateLimiter:
    """Helper for adapters whose source runs faster than target_hz.

    Usage:
        limiter = RateLimiter(hz=30.0)
        for raw in source:
            if limiter.should_emit():
                yield convert(raw)

    The limiter uses a monotonic clock and admits a frame whenever at
    least 1/hz seconds have elapsed since the last emission. It does not
    sleep — drop-on-arrival keeps the source loop tight.
    """

    def __init__(self, hz: float) -> None:
        self._period = 1.0 / hz
        self._last_emit_t: float | None = None

    def should_emit(self) -> bool:
        now = time.monotonic()
        if self._last_emit_t is None or (now - self._last_emit_t) >= self._period:
            self._last_emit_t = now
            return True
        return False
