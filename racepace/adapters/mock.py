"""Replay-from-SQLite adapter.

Plays back a previously recorded session at real-time pace (or accelerated
via `speed`). Useful for cross-platform development without ACC, and as a
deterministic source for tests.

`speed=0.0` means as fast as possible — yields every frame back-to-back
with no sleeping, which is what the test fixtures use.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

from racepace.adapters.base import AbstractAdapter
from racepace.schema import SessionInfo, TelemetryFrame
from racepace.storage.session_store import SessionReader


class MockAdapter(AbstractAdapter):
    sim_name = "mock"

    def __init__(
        self,
        path: str | Path,
        session_id: str | None = None,
        speed: float = 1.0,
    ) -> None:
        self.path = Path(path)
        self.session_id = session_id
        self.speed = speed
        self._reader: SessionReader | None = None
        self._info: SessionInfo | None = None
        self._frames: list[TelemetryFrame] | None = None

    def connect(self) -> None:
        self._reader = SessionReader(self.path)
        sid = self.session_id or self._reader.latest_session_id()
        if sid is None:
            raise RuntimeError(f"No sessions in {self.path}.")
        self._info, self._frames = self._reader.load_session(sid)
        self.sim_name = self._info.sim

    def disconnect(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def read_session_info(self) -> SessionInfo:
        if self._info is None:
            raise RuntimeError("connect() before read_session_info()")
        return self._info

    def stream_frames(self) -> Iterator[TelemetryFrame]:
        if self._frames is None:
            raise RuntimeError("connect() before stream_frames()")
        if self.speed <= 0.0:
            yield from self._frames
            return

        start_wall = time.monotonic()
        start_sim = self._frames[0].timestamp_s if self._frames else 0.0
        for f in self._frames:
            target = (f.timestamp_s - start_sim) / self.speed
            elapsed = time.monotonic() - start_wall
            if elapsed < target:
                time.sleep(target - elapsed)
            yield f
