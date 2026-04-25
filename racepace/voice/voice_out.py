"""Voice-enabled output sink.

Drop-in replacement for `racepace.comms.text_out.TextOutput`: same
`.send(msg)` signature, also synthesizes TTS and queues into a Player.

If TTS or audio is unavailable (no backend), VoiceOutput silently falls
back to text-only — never blocks the agent that called .send().
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path

from racepace.voice.live_tts import TTSBackend, default_tts
from racepace.voice.player import Player


class VoiceOutput:
    def __init__(
        self,
        log_path: str | Path | None = None,
        prefix: str = "ENGINEER",
        tts: TTSBackend | None = None,
        player: Player | None = None,
        speak: bool = True,
    ) -> None:
        self.log_path = Path(log_path) if log_path else None
        self.prefix = prefix
        self.tts = tts if tts is not None else default_tts()
        self.player = player if player is not None else Player()
        self.speak = speak
        self._lock = threading.Lock()
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text("", encoding="utf-8")

    def send(self, msg: str) -> None:
        line = f"[{datetime.now():%H:%M:%S}] {self.prefix}: {msg}"
        with self._lock:
            print(line, flush=True)
            if self.log_path:
                with self.log_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
        if not self.speak or not msg.strip():
            return
        # Synthesize off the caller's thread so a slow TTS never blocks the
        # agent loop. The Player itself is also non-blocking.
        threading.Thread(target=self._synth_and_queue, args=(msg,), daemon=True).start()

    def _synth_and_queue(self, msg: str) -> None:
        try:
            samples, sr = self.tts.synthesize(msg)
        except Exception:
            return
        self.player.play(samples, sr, label=msg[:40])

    def stop(self) -> None:
        try:
            self.player.stop()
        except Exception:
            pass
