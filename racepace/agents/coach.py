"""Driving coach with two loops.

FAST LOOP (~30Hz, no LLM): pace-note callouts via a prebuilt clip cache.
SLOW LOOP (per sector or per lap): one targeted coaching sentence via
gemini-2.5-flash, spoken only on a straight.

The fast loop's hot path: read latest frame → find next corner → check
trigger windows → enqueue cached clip. No IO, no LLM, no allocation
beyond a list comprehension. End-to-end latency target: <150ms.

If no reference lap is loaded, the coach runs in PACE-NOTES-ONLY mode:
direction + severity callouts, no comparative coaching. This is a
deliberate degraded mode — the coach starts even on a brand-new track.
"""

from __future__ import annotations

import bisect
import json
import os
import threading
from dataclasses import asdict, dataclass
from typing import Callable

from google import genai
from google.genai import types

from racepace.features.reference import (
    Reference,
    SectorDelta,
    live_delta,
    sector_delta,
)
from racepace.features.track_map import Corner, TrackMap
from racepace.schema import SessionInfo, TelemetryFrame
from racepace.storage.ringbuffer import RingBuffer
from racepace.voice.cache import ClipCache


_MODEL = "gemini-2.5-flash"

_SLOW_SYSTEM_PROMPT = """You are a calm, expert sim-racing driving coach. You speak between corners, not during them. You receive a structured comparison of a single sector against a recorded reference lap.

Style:
- ONE sentence, max 14 words. No preamble. No closing pleasantries.
- Be specific: name the corner (if known), say what the driver did vs what the reference did.
- Quote numbers verbatim from the input. Do not estimate or compute.
- Tone: encouraging but direct. Treat the driver as competent.

If the input describes nothing actionable (e.g. delta < 0.15s), output the literal token NO_COMMENT and nothing else."""


# Trigger windows (seconds-to-corner)
ANNOUNCE_WINDOW_S = 3.0
ACTION_WINDOW_S = 1.5

# Suppression: never two callouts within this gap
CALLOUT_MIN_GAP_S = 0.8

# Slow-loop trigger threshold
SECTOR_LOSS_TRIGGER_S = 0.15


@dataclass
class _FastTrigger:
    """One scheduled callout that the fast loop will fire when met."""
    corner_id: int
    kind: str             # "announce", "action", "exit_check"
    fired: bool = False


@dataclass
class CoachConfig:
    fast_tick_hz: float = 30.0
    sector_loss_trigger_s: float = SECTOR_LOSS_TRIGGER_S
    pace_notes_only: bool = False   # forces degraded mode even if reference exists


class CoachAgent:
    """Two-loop driving coach. Caller is responsible for start()/stop()."""

    def __init__(
        self,
        ringbuffer: RingBuffer,
        session_info: SessionInfo,
        track_map: TrackMap | None,
        clip_cache: ClipCache,
        speak_clip: Callable[[str], None],
        speak_text: Callable[[str], None],
        reference: Reference | None = None,
        config: CoachConfig | None = None,
        llm_call: Callable[[str], str] | None = None,
        api_key: str | None = None,
    ) -> None:
        self.ringbuffer = ringbuffer
        self.session_info = session_info
        self.track_map = track_map
        self.cache = clip_cache
        self.speak_clip = speak_clip
        self.speak_text = speak_text
        self.reference = reference
        self.config = config or CoachConfig()
        self._llm_call = llm_call or _default_llm_call(api_key)

        self._stop = threading.Event()
        self._fast_thread: threading.Thread | None = None
        self._slow_thread: threading.Thread | None = None

        # Fast-loop dedupe state: which (corner_id, kind) we've already fired this lap.
        self._fired_this_lap: set[tuple[int, str]] = set()
        self._current_lap: int = -1
        # Min-gap is on the session clock so this works under fast replay too.
        self._last_callout_session_t: float = float("-inf")
        self._latest_session_t: float = 0.0
        # Slow-loop dedupe
        self._slow_last_lap: int = -1
        self._slow_last_sector: int = -1

        # Sorted corner-brake-distance index for binary search.
        if track_map and track_map.corners:
            self._corners_sorted = sorted(track_map.corners, key=lambda c: c.brake_point_m)
            self._brake_points = [c.brake_point_m for c in self._corners_sorted]
        else:
            self._corners_sorted = []
            self._brake_points = []

    # --- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        if self._fast_thread and self._fast_thread.is_alive():
            return
        self._stop.clear()
        self._fast_thread = threading.Thread(target=self._fast_run, name="CoachFast", daemon=True)
        self._slow_thread = threading.Thread(target=self._slow_run, name="CoachSlow", daemon=True)
        self._fast_thread.start()
        self._slow_thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        for t in (self._fast_thread, self._slow_thread):
            if t:
                t.join(timeout=timeout)

    # --- fast loop ------------------------------------------------------------

    def _fast_run(self) -> None:
        period = 1.0 / max(1.0, self.config.fast_tick_hz)
        while not self._stop.is_set():
            try:
                self.fast_tick()
            except Exception:
                pass
            self._stop.wait(period)

    def fast_tick(self) -> None:
        """One iteration of the fast loop. Public for testing."""
        latest = self.ringbuffer.latest()
        if latest is None:
            return
        self._latest_session_t = latest.timestamp_s
        if latest.lap_number != self._current_lap:
            # New lap: reset dedupe.
            self._current_lap = latest.lap_number
            self._fired_this_lap.clear()
        if not self._corners_sorted:
            return
        if latest.lap_distance_m is None or latest.speed_kph is None or latest.speed_kph <= 1:
            return

        next_corner = self._next_corner(latest.lap_distance_m)
        if next_corner is None:
            return

        speed_mps = max(0.5, latest.speed_kph / 3.6)
        time_to_brake = (next_corner.brake_point_m - latest.lap_distance_m) / speed_mps

        # Three trigger windows
        if ANNOUNCE_WINDOW_S - 0.4 < time_to_brake < ANNOUNCE_WINDOW_S:
            self._fire_callout(next_corner, "announce")
        elif ACTION_WINDOW_S - 0.4 < time_to_brake < ACTION_WINDOW_S:
            self._fire_callout(next_corner, "action")
        elif latest.lap_distance_m > next_corner.exit_m:
            self._fire_callout(next_corner, "exit_check")

    def _next_corner(self, distance_m: float) -> Corner | None:
        idx = bisect.bisect_right(self._brake_points, distance_m)
        if idx >= len(self._corners_sorted):
            return None
        return self._corners_sorted[idx]

    def _fire_callout(self, corner: Corner, kind: str) -> None:
        key = (corner.id, kind)
        if key in self._fired_this_lap:
            return
        if (self._latest_session_t - self._last_callout_session_t) < CALLOUT_MIN_GAP_S:
            return

        phrase = self._phrase_for(corner, kind)
        if phrase is None:
            return
        self._fired_this_lap.add(key)
        if self.cache.get(phrase) is None:
            return  # phrase not cached; skip silently
        self.speak_clip(phrase)
        self._last_callout_session_t = self._latest_session_t

    def _phrase_for(self, corner: Corner, kind: str) -> str | None:
        sev_words = ["", "one", "two", "three", "four", "five", "six"]
        sev_word = sev_words[min(max(1, corner.severity), 6)]
        direction = "left" if corner.direction == "L" else "right"
        if kind == "announce":
            return f"{direction} {sev_word}"
        if kind == "action":
            if corner.flat:
                return "flat"
            if corner.severity <= 2:
                return "lift"
            return "brake"
        if kind == "exit_check":
            # Only meaningful when we have a reference and the driver is slow at apex.
            if self.reference is None or self.config.pace_notes_only:
                return None
            return self._exit_phrase_if_slow(corner)
        return None

    def _exit_phrase_if_slow(self, corner: Corner) -> str | None:
        # Sample apex speed from the latest snapshot's frames in this lap.
        # If apex speed is materially below the corner's target, suggest carrying more.
        snap = self.ringbuffer.snapshot()
        if not snap:
            return None
        apex_window = [f for f in snap if f.lap_number == self._current_lap and f.lap_distance_m and abs(f.lap_distance_m - corner.apex_m) < 30.0]
        if not apex_window:
            return None
        speeds = [f.speed_kph for f in apex_window if f.speed_kph is not None]
        if not speeds:
            return None
        actual_min = min(speeds)
        if actual_min < corner.target_min_speed_kph - 5.0:
            return "carry more speed"
        return None

    # --- slow loop ------------------------------------------------------------

    def _slow_run(self) -> None:
        # Cheaper to wake on a 1-second timer and check than to do per-frame.
        while not self._stop.is_set():
            try:
                self.slow_tick()
            except Exception:
                pass
            self._stop.wait(1.0)

    def slow_tick(self) -> None:
        """Sector-by-sector check; one coaching sentence at most per sector."""
        if self.reference is None or self.config.pace_notes_only:
            return
        snap = self.ringbuffer.snapshot()
        if not snap:
            return
        latest = snap[-1]
        if latest.current_sector is None:
            return

        # Trigger when the sector index advances (i.e. driver crossed into a new sector).
        if latest.lap_number == self._slow_last_lap and latest.current_sector == self._slow_last_sector:
            return

        # The sector that just COMPLETED is one back from current_sector
        completed_sector = latest.current_sector - 1
        completed_lap = latest.lap_number
        if completed_sector < 0:
            # Sector 0 just started; the just-completed sector is the last of the previous lap.
            completed_sector = 2
            completed_lap = latest.lap_number - 1
        if completed_lap < 1:
            self._slow_last_lap = latest.lap_number
            self._slow_last_sector = latest.current_sector
            return

        # Reconstruct the just-completed lap from the snapshot.
        from racepace.features.laps import split_into_laps
        laps = split_into_laps(snap)
        completed = next((l for l in laps if l.lap_number == completed_lap), None)
        if completed is None:
            self._slow_last_lap = latest.lap_number
            self._slow_last_sector = latest.current_sector
            return

        delta = sector_delta(self.reference, completed, completed_sector)
        if delta is None:
            self._slow_last_lap = latest.lap_number
            self._slow_last_sector = latest.current_sector
            return

        # Only speak if the loss is meaningful AND driver is currently on a straight.
        on_straight = self._is_on_straight(latest)
        if delta.delta_s >= self.config.sector_loss_trigger_s and on_straight:
            corner_name = self._nearest_corner_name(delta.worst_distance_m)
            payload = {
                "sector": delta.sector + 1,
                "delta_s": round(delta.delta_s, 3),
                "worst_corner": corner_name,
                "worst_distance_m": round(delta.worst_distance_m, 1) if delta.worst_distance_m else None,
                "worst_local_loss_s": round(delta.worst_local_delta_s, 3) if delta.worst_local_delta_s else None,
            }
            try:
                msg = self._llm_call(json.dumps(payload))
            except Exception:
                msg = ""
            if msg and msg.strip().upper() != "NO_COMMENT":
                self.speak_text(msg.strip())

        self._slow_last_lap = latest.lap_number
        self._slow_last_sector = latest.current_sector

    def _is_on_straight(self, frame: TelemetryFrame) -> bool:
        if not self._corners_sorted or frame.lap_distance_m is None:
            return True
        nxt = self._next_corner(frame.lap_distance_m)
        if nxt is None:
            return True
        speed_mps = max(0.5, (frame.speed_kph or 0) / 3.6)
        time_to_brake = (nxt.brake_point_m - frame.lap_distance_m) / speed_mps
        return time_to_brake > ANNOUNCE_WINDOW_S + 0.5

    def _nearest_corner_name(self, distance_m: float | None) -> str | None:
        if distance_m is None or not self._corners_sorted:
            return None
        # Find corner whose apex is closest.
        best = min(self._corners_sorted, key=lambda c: abs(c.apex_m - distance_m))
        return best.name or f"corner {best.id}"


# --- LLM ----------------------------------------------------------------------

def _default_llm_call(api_key: str | None) -> Callable[[str], str]:
    def _call(payload_json: str) -> str:
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY).")
        client = genai.Client(api_key=key)
        user_text = f"Sector comparison:\n```json\n{payload_json}\n```\n\nSpeak now."
        resp = client.models.generate_content(
            model=_MODEL,
            contents=user_text,
            config=types.GenerateContentConfig(
                system_instruction=_SLOW_SYSTEM_PROMPT,
                max_output_tokens=120,
                # Disable thinking — for one-sentence callouts we don't need the
                # model to deliberate; it eats the output budget and truncates.
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return (resp.text or "").strip()

    return _call
