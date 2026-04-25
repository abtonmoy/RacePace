"""Console + optional log-file output sink.

Phase 2 only. Voice (TTS) is Phase 3.
Thread-safe — multiple agents may share one TextOutput.
"""

from __future__ import annotations

import sys
import threading
from datetime import datetime
from pathlib import Path


class TextOutput:
    def __init__(self, log_path: str | Path | None = None, prefix: str = "ENGINEER") -> None:
        self.log_path = Path(log_path) if log_path else None
        self.prefix = prefix
        self._lock = threading.Lock()
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            # Truncate to start clean for this session.
            self.log_path.write_text("", encoding="utf-8")

    def send(self, msg: str) -> None:
        line = f"[{datetime.now():%H:%M:%S}] {self.prefix}: {msg}"
        with self._lock:
            print(line, file=sys.stdout, flush=True)
            if self.log_path:
                with self.log_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")


class CapturingOutput:
    """In-memory sink for tests."""

    def __init__(self) -> None:
        self.messages: list[str] = []
        self._lock = threading.Lock()

    def send(self, msg: str) -> None:
        with self._lock:
            self.messages.append(msg)
