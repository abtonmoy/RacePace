"""Shared base for live agents (Phase 2 engineer, Phase 3 coach).

The agent runs its own thread, polling the ring buffer at `tick_interval_s`.
Subclasses override `should_speak()` and `generate_message()`.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from racepace.comms.text_out import TextOutput
    from racepace.features.situation import SituationReport
    from racepace.schema import SessionInfo
    from racepace.storage.ringbuffer import RingBuffer


class LiveAgent(ABC):
    tick_interval_s: float = 2.0

    def __init__(
        self,
        ringbuffer: "RingBuffer",
        session_info: "SessionInfo",
        output: "TextOutput",
    ) -> None:
        self.ringbuffer = ringbuffer
        self.session_info = session_info
        self.output = output
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @abstractmethod
    def build_report(self) -> "SituationReport | None":
        """Pull a snapshot from the ring buffer and build the structured report."""

    @abstractmethod
    def should_speak(self, report: "SituationReport") -> tuple[bool, str | None]:
        """Return (yes/no, trigger_reason). The gatekeeper. Default to silence."""

    @abstractmethod
    def generate_message(self, report: "SituationReport", trigger: str) -> str:
        """Produce the user-facing message. May call an LLM."""

    def start(self) -> threading.Thread:
        if self._thread and self._thread.is_alive():
            return self._thread
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=type(self).__name__, daemon=True)
        self._thread.start()
        return self._thread

    def stop(self, join_timeout_s: float = 5.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=join_timeout_s)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                report = self.build_report()
                if report is not None:
                    yes, trigger = self.should_speak(report)
                    if yes:
                        msg = self.generate_message(report, trigger or "")
                        if msg:
                            self.output.send(msg)
            except Exception as exc:  # keep the loop alive
                self.output.send(f"[agent error] {exc!r}")
            # Wait with cancellation; lets stop() interrupt without waiting full tick.
            self._stop.wait(self.tick_interval_s)
